"""Orchestration: query in, ranked qualified companies out.

    plan -> filter -> retrieve -> qualify -> escalate -> rank

Each stage is a pure-ish function over the previous stage's output, so any of
them can be swapped, skipped or tested alone. The pipeline object owns only
the things that are expensive to build -- the corpus embedding and the cache
-- so running twelve queries reuses them instead of rebuilding per query.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Optional

from . import config, filters, planner, qualifier, ranking, retrieval
from .cache import DiskCache
from .embeddings import CorpusIndex
from .llm import Gemini
from .loading import company_document, load_companies
from .schema import Company, QueryResult, QuerySpec, Usage


class QualificationPipeline:
    """Holds the corpus-level state shared across queries."""

    def __init__(
        self,
        companies: list[Company],
        cfg: Optional[config.PipelineConfig] = None,
        cache: Optional[DiskCache] = None,
    ) -> None:
        self.companies = companies
        self.cfg = cfg or config.DEFAULT
        self.cache = cache or DiskCache(config.CACHE_DIR / "iq.sqlite", self.cfg.use_cache)
        self.documents = [company_document(c) for c in companies]
        self.index: Optional[CorpusIndex] = None

    @classmethod
    def from_file(
        cls, path: str | Path, cfg: Optional[config.PipelineConfig] = None
    ) -> "QualificationPipeline":
        return cls(load_companies(path), cfg)

    async def warm(self, client: Gemini, rebuild: bool = False) -> None:
        """Build (or load) the corpus embedding. Idempotent."""
        if self.index is None or rebuild:
            self.index = await CorpusIndex.build(
                client, self.documents, self.cache, rebuild=rebuild
            )

    async def run(self, client: Gemini, query: str) -> QueryResult:
        await self.warm(client)
        assert self.index is not None
        cfg = self.cfg
        timings: dict[str, float] = {}
        counts: dict[str, int] = {"database": len(self.companies)}
        before = _snapshot(client.usage)

        # --- Stage 0: plan -------------------------------------------------
        t = time.perf_counter()
        spec = await planner.plan_query(client, query)
        timings["plan"] = time.perf_counter() - t

        # --- Stage 1: hard filters ----------------------------------------
        t = time.perf_counter()
        survivors, rejected_by = filters.filter_companies(self.companies, spec)
        # If the planner's numeric constraints wiped out almost everything, it
        # is far more likely that it over-read the query than that the answer
        # is genuinely empty. Retry once with the numeric gates demoted.
        relaxed = False
        if filters.relax(spec, len(survivors)):
            soft_spec = filters.soften(spec)
            soft_survivors, soft_rejected = filters.filter_companies(self.companies, soft_spec)
            if len(soft_survivors) > len(survivors):
                survivors, rejected_by, relaxed = soft_survivors, soft_rejected, True
        timings["filter"] = time.perf_counter() - t
        counts["after_filter"] = len(survivors)
        counts["filter_relaxed"] = int(relaxed)

        if not survivors:
            return QueryResult(
                query=query, spec=spec, companies=[],
                usage=_delta(client.usage, before), timings=timings, stage_counts=counts,
            )

        # --- Stage 2: cheap retrieval --------------------------------------
        t = time.perf_counter()
        probe = spec.ideal_profile or spec.role_statement or query
        query_vec = await self.index.embed_query(client, probe, self.cache)
        top_k = cfg.shortlist_by_complexity.get(spec.complexity, 55)
        candidates = retrieval.retrieve(
            survivors, spec, self.index.matrix, query_vec, self.documents, top_k, cfg
        )
        timings["retrieve"] = time.perf_counter() - t
        counts["shortlist"] = len(candidates)

        # --- Stage 3: batched qualification --------------------------------
        t = time.perf_counter()
        quals = await qualifier.qualify(client, candidates, spec, cfg)
        timings["qualify"] = time.perf_counter() - t

        # --- Stage 4: selective escalation ---------------------------------
        t = time.perf_counter()
        quals, escalated = await qualifier.escalate(client, candidates, quals, spec, cfg)
        timings["escalate"] = time.perf_counter() - t
        counts["escalated"] = escalated

        # --- Stage 5: rank --------------------------------------------------
        t = time.perf_counter()
        ranked = ranking.rank(candidates, quals, spec, cfg)
        timings["rank"] = time.perf_counter() - t
        counts["returned"] = len(ranked)
        counts["qualified"] = sum(
            1 for r in ranked if r.qualification.verdict.value == "QUALIFIED"
        )
        timings["total"] = sum(
            v for k, v in timings.items() if k != "total" and isinstance(v, float)
        )

        return QueryResult(
            query=query, spec=spec, companies=ranked,
            usage=_delta(client.usage, before), timings=timings, stage_counts=counts,
        )

    async def run_many(
        self, client: Gemini, queries: list[str], concurrent_queries: int = 2
    ) -> list[QueryResult]:
        """Run several queries, bounded so they do not fight for rate limit.

        Queries are already internally concurrent (batches fan out inside a
        single query), so stacking many queries on top of that just moves the
        contention into backoff. Two at a time was the sweet spot on this key.
        """
        await self.warm(client)
        gate = asyncio.Semaphore(concurrent_queries)

        async def one(q: str) -> QueryResult:
            async with gate:
                try:
                    return await self.run(client, q)
                except Exception as exc:  # noqa: BLE001 - isolate per query
                    # One query must never take the batch down with it. A
                    # benchmark that loses eleven good results because the
                    # twelfth hit a rate limit is worse than useless, because
                    # it also throws away everything already paid for.
                    return QueryResult(
                        query=q,
                        spec=planner.heuristic_spec(q),
                        companies=[],
                        usage=Usage(),
                        timings={"total": 0.0},
                        stage_counts={"database": len(self.companies), "failed": 1},
                        error=str(exc)[:300],
                    )

        return await asyncio.gather(*[one(q) for q in queries])


def _snapshot(usage: Usage) -> Usage:
    snap = Usage()
    snap.merge(usage)
    return snap


def _delta(now: Usage, before: Usage) -> Usage:
    out = Usage(
        calls=now.calls - before.calls,
        prompt_tokens=now.prompt_tokens - before.prompt_tokens,
        output_tokens=now.output_tokens - before.output_tokens,
        embed_tokens=now.embed_tokens - before.embed_tokens,
        cached_calls=now.cached_calls - before.cached_calls,
        cost_usd=now.cost_usd - before.cost_usd,
    )
    for model, count in now.by_model.items():
        diff = count - before.by_model.get(model, 0)
        if diff:
            out.by_model[model] = diff
    return out
