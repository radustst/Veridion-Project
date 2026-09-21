"""The two baselines from the brief, implemented properly so the comparison is fair.

Both are built from the same loaders, the same company rendering and the same
client as the real system. A comparison against a deliberately weakened
baseline proves nothing, so the only differences here are architectural:
Baseline A spends an LLM call on every company, and Baseline B spends none.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Sequence

import numpy as np

from iq import config
from iq.cache import DiskCache
from iq.embeddings import CorpusIndex, embed_texts
from iq.llm import Gemini, LLMError
from iq.loading import company_card
from iq.schema import Company, Usage

# ---------------------------------------------------------------------------
# Baseline A: one LLM call per company
# ---------------------------------------------------------------------------
A_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "match": {"type": "boolean"},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["match", "confidence", "reason"],
}

A_PROMPT = """Does this company match the user's query?

QUERY: {query}

COMPANY
{card}

Answer with match (true/false), your confidence 0-1, and a one-sentence reason."""


async def _ask_one(client: Gemini, query: str, company: Company) -> tuple[int, float, str]:
    try:
        data = await client.generate_json(
            config.QUALIFIER,
            A_PROMPT.format(query=query, card=company_card(company)),
            A_SCHEMA,
            namespace="baseline_a",
            thinking="low",
        )
    except LLMError as exc:
        return company.idx, 0.0, "error: {}".format(str(exc)[:80])
    score = float(bool(data.get("match"))) * max(
        0.0, min(1.0, float(data.get("confidence", 0.5) or 0.0))
    )
    return company.idx, score, str(data.get("reason", ""))


def project_full_corpus_cost(
    measured: dict[str, Any], n_companies: int, rpm: int = 15
) -> dict[str, Any]:
    """Extrapolate Baseline A over a whole corpus from a measured sample.

    Running one LLM call per company over 456 companies needs 456 requests,
    and the free tier caps this model at 15 requests per minute, so a single
    query takes over half an hour. Measuring it end to end for every benchmark
    query was not possible here, so the per-call cost and token counts are
    measured on a real sample and scaled. Scaling is legitimate for cost --
    each call is independent and near-identical in size -- and it is stated as
    a projection rather than presented as a measurement.
    """
    calls = max(1, measured.get("calls_measured", 0))
    cost = measured.get("cost_usd", 0.0)
    if not measured.get("calls_measured") or cost <= 0:
        # Every call was served from cache, so this run cost nothing and
        # scaling it would project zero. Say so instead of reporting a
        # confident $0.00 -- a projection from no measurement is not a
        # projection.
        return {
            "projected_calls": n_companies,
            "projected_cost_usd": None,
            "projected_minutes_at_free_tier": round(n_companies / max(1, rpm), 1),
            "note": "no uncached calls in this run; cost not measurable here",
        }
    return {
        "projected_calls": n_companies,
        "projected_cost_usd": round(
            measured.get("cost_usd", 0.0) / calls * n_companies, 6
        ),
        "projected_minutes_at_free_tier": round(n_companies / max(1, rpm), 1),
        "measured_on_calls": calls,
        "cost_per_call_usd": round(measured.get("cost_usd", 0.0) / calls, 8),
    }


async def baseline_a(
    client: Gemini, query: str, companies: Sequence[Company], top_k: int = 25
) -> dict[str, Any]:
    """Send every company to the LLM individually, then rank by confidence.

    This is the strategy the brief describes as accurate but unaffordable. It
    runs on the same cheap model the cascade uses for its bulk tier, so the
    comparison isolates *architecture* rather than model quality -- any cost
    or latency difference is purely the price of not filtering first.

    `companies` may be the whole database (the true Baseline A) or the judged
    pool (a cheaper variant that isolates one specific question: is
    adjudicating companies one at a time actually more accurate than
    adjudicating them eight to a prompt?). The caller labels which was run;
    the two must not be conflated, because the pooled variant says nothing
    about recall over the full corpus.
    """
    before_calls, before_cost = client.usage.calls, client.usage.cost_usd
    t0 = time.perf_counter()
    results = await asyncio.gather(*[_ask_one(client, query, c) for c in companies])
    elapsed = time.perf_counter() - t0

    by_idx = {c.idx: c for c in companies}
    matched = [(idx, score, reason) for idx, score, reason in results if score > 0]
    matched.sort(key=lambda r: -r[1])
    return {
        "system": "baseline_a_llm_per_company",
        "query": query,
        "companies_seen": len(companies),
        "calls_measured": client.usage.calls - before_calls,
        "ranked_idx": [idx for idx, _, _ in matched[:top_k]],
        "detail": [
            {"name": by_idx[i].name, "score": round(s, 3), "reason": r}
            for i, s, r in matched[:top_k]
        ],
        "llm_calls": client.usage.calls - before_calls,
        "cost_usd": round(client.usage.cost_usd - before_cost, 6),
        "seconds": round(elapsed, 2),
    }


# ---------------------------------------------------------------------------
# Baseline B: embedding similarity only
# ---------------------------------------------------------------------------
async def baseline_b(
    client: Gemini,
    query: str,
    companies: Sequence[Company],
    index: CorpusIndex,
    cache: DiskCache,
    top_k: int = 25,
) -> dict[str, Any]:
    """Embed the raw query, rank by cosine. No planning, no filtering, no LLM.

    Uses RETRIEVAL_QUERY task type, which is the correct and strongest setting
    for a genuine query-side embedding -- the point is to show that even a
    well-implemented similarity search misreads intent, not to beat a straw man.
    """
    before_cost = client.usage.cost_usd
    t0 = time.perf_counter()
    qvec = (await embed_texts(client, [query], cache, "RETRIEVAL_QUERY"))[0]
    sims = index.matrix @ qvec
    order = np.argsort(-sims)[:top_k]
    elapsed = time.perf_counter() - t0

    by_idx = {c.idx: c for c in companies}
    return {
        "system": "baseline_b_embedding_only",
        "query": query,
        "ranked_idx": [int(i) for i in order],
        "detail": [
            {"name": by_idx[int(i)].name, "score": round(float(sims[i]), 4)} for i in order
        ],
        "llm_calls": 0,
        "cost_usd": round(client.usage.cost_usd - before_cost, 6),
        "seconds": round(elapsed, 2),
    }
