#!/usr/bin/env python3
"""Measure run-to-run stability with the cache disabled.

  python -m evaluation.consistency --runs 3 --queries q01,q04,q08

"Inconsistent: borderline cases may produce different answers across runs" is
one of the four charges against the naive baseline, and it is the only one
that cannot be argued from architecture alone -- it has to be measured.

Caching makes repeat runs trivially identical, which proves nothing, so the
cache is bypassed here. What is left is the genuine variance of the model at
temperature 0, plus any variance introduced by batching: which companies
share a prompt depends on retrieval order, and a company's neighbours can
influence its verdict.

Reported per query:
  set Jaccard  -- do the same companies come back at all
  top-10 overlap -- do the same companies come back near the top
  verdict flips -- companies whose verdict changed between runs
"""
from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import statistics
from typing import Any

from evaluation.queries import BY_ID, QUERIES
from iq import config
from iq.cache import DiskCache
from iq.llm import Gemini
from iq.pipeline import QualificationPipeline


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


async def measure(args: argparse.Namespace) -> dict[str, Any]:
    ids = args.queries.split(",") if args.queries else ["q01", "q04", "q08"]
    queries = [BY_ID[i] for i in ids if i in BY_ID] or QUERIES[:3]

    cfg = config.PipelineConfig()
    cfg.use_cache = False  # the whole point
    pipeline = QualificationPipeline.from_file(args.data, cfg)
    # Embeddings stay cached: they are deterministic and re-buying them would
    # cost five minutes of rate limit without testing anything.
    warm_cache = DiskCache(config.CACHE_DIR / "iq.sqlite", True)
    pipeline.cache = warm_cache

    out: dict[str, Any] = {"runs": args.runs, "queries": {}}
    async with Gemini(warm_cache, config.PipelineConfig()) as warm_client:
        await pipeline.warm(warm_client)

    # A cache-disabled client for the actual measurement.
    cold_cache = DiskCache(config.CACHE_DIR / "consistency.sqlite", False)
    pipeline.cache = cold_cache
    async with Gemini(cold_cache, cfg) as client:
        for q in queries:
            runs: list[dict[str, str]] = []
            for _ in range(args.runs):
                result = await pipeline.run(client, q.text)
                runs.append({
                    rc.company.name: rc.qualification.verdict.value
                    for rc in result.companies
                })

            sets = [set(r) for r in runs]
            tops = [list(r)[:10] for r in runs]
            pair_jaccard = [jaccard(a, b) for a, b in itertools.combinations(sets, 2)]
            pair_top = [
                len(set(a) & set(b)) / max(1, min(len(a), len(b)))
                for a, b in itertools.combinations(tops, 2)
            ]

            everyone = set().union(*sets) if sets else set()
            flips = []
            for name in sorted(everyone):
                verdicts = {r.get(name, "ABSENT") for r in runs}
                if len(verdicts) > 1:
                    flips.append({"company": name, "verdicts": sorted(verdicts)})

            out["queries"][q.id] = {
                "query": q.text,
                "kind": q.kind,
                "returned_per_run": [len(s) for s in sets],
                "mean_set_jaccard": round(statistics.fmean(pair_jaccard), 4) if pair_jaccard else 1.0,
                "mean_top10_overlap": round(statistics.fmean(pair_top), 4) if pair_top else 1.0,
                "unstable_companies": len(flips),
                "stable_companies": len(everyone) - len(flips),
                "flips": flips[:15],
            }
            print("  {}: jaccard {:.3f}, top10 {:.3f}, {} unstable of {}".format(
                q.id, out["queries"][q.id]["mean_set_jaccard"],
                out["queries"][q.id]["mean_top10_overlap"], len(flips), len(everyone)))

    js = [v["mean_set_jaccard"] for v in out["queries"].values()]
    ts = [v["mean_top10_overlap"] for v in out["queries"].values()]
    out["overall"] = {
        "mean_set_jaccard": round(statistics.fmean(js), 4),
        "mean_top10_overlap": round(statistics.fmean(ts), 4),
    }
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(config.DATA_DIR / "companies.jsonl"))
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--queries", help="comma-separated ids, default q01,q04,q08")
    args = p.parse_args()
    report = asyncio.run(measure(args))
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.RESULTS_DIR / "consistency.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["overall"], indent=2))
    print("wrote {}".format(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
