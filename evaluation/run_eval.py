#!/usr/bin/env python3
"""Benchmark the cascade against both baselines with an LLM judge.

  python -m evaluation.run_eval                    # full run
  python -m evaluation.run_eval --skip-baseline-a  # skip the expensive one
  python -m evaluation.run_eval --baseline-a-queries q01,q04,q07

Method
------
1. Run the cascade, Baseline B (embedding only) and -- on a sampled subset,
   because it is the expensive one -- Baseline A (one LLM call per company).
2. Pool the top-N results from every system for each query.
3. Grade the whole pool with a judge that never sees which system produced
   what, nor the planner's interpretation of the query.
4. Score every system against those grades.

Pooling matters. Grading only what one system returned would make a system
that returns three results look perfect, so the pool is the union across
systems and every system is scored against the same labels. This is the
standard TREC pooling approach and it has the same known limitation: a
relevant company no system retrieved is invisible, so recall is measured
against the pool, not against the database. That is stated in the report
output rather than left for the reader to discover.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from typing import Any

from evaluation import baselines, judge, metrics
from evaluation.queries import QUERIES, BenchmarkQuery
from iq import config
from iq.cache import DiskCache
from iq.llm import Gemini
from iq.pipeline import QualificationPipeline

POOL_DEPTH = 20


async def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    cfg = config.PipelineConfig()
    pipeline = QualificationPipeline.from_file(args.data, cfg)
    cache = DiskCache(config.CACHE_DIR / "iq.sqlite", True)
    pipeline.cache = cache
    companies = pipeline.companies
    by_idx = {c.idx: c for c in companies}

    queries: list[BenchmarkQuery] = QUERIES
    if args.queries:
        wanted = set(args.queries.split(","))
        queries = [q for q in QUERIES if q.id in wanted] or QUERIES
    a_ids = (
        set(args.baseline_a_queries.split(","))
        if args.baseline_a_queries
        else {"q01", "q04", "q07"}
    )

    report: dict[str, Any] = {
        "pool_depth": POOL_DEPTH,
        "relevance_threshold": metrics.RELEVANT_AT,
        "queries": {},
        "systems": {},
        "cost": {},
    }

    async with Gemini(cache, cfg) as client:
        await pipeline.warm(client)

        # ---------------- system under test ----------------
        t0 = time.perf_counter()
        cascade = await pipeline.run_many(client, [q.text for q in queries], args.parallel)
        cascade_wall = time.perf_counter() - t0
        cascade_by_query = {q.id: r for q, r in zip(queries, cascade)}

        # ---------------- baseline B (all queries) ----------------
        assert pipeline.index is not None
        b_runs = {}
        t0 = time.perf_counter()
        for q in queries:
            b_runs[q.id] = await baselines.baseline_b(
                client, q.text, companies, pipeline.index, cache, POOL_DEPTH
            )
        b_wall = time.perf_counter() - t0

        # ---------------- pool ----------------
        pools: dict[str, list[int]] = {}
        for q in queries:
            pool = [rc.company.idx for rc in cascade_by_query[q.id].companies[:POOL_DEPTH]]
            pool += b_runs[q.id]["ranked_idx"][:POOL_DEPTH]
            pools[q.id] = list(dict.fromkeys(pool))

        # ---------------- baseline A, over the pool ----------------
        # One call per company over the full 456-row corpus needs 456 requests
        # per query, and this model is capped at 15 requests/minute on the free
        # tier -- over half an hour for a single query, which is not a budget
        # worth spending on a baseline. Running it over the judged pool instead
        # costs ~40 calls and still answers the question that matters
        # architecturally: does one-company-per-call adjudication beat
        # eight-per-call? It deliberately does NOT measure recall over the full
        # corpus, and is labelled "pooled" everywhere so the two are not
        # confused.
        a_runs: dict[str, Any] = {}
        a_wall = 0.0
        if not args.skip_baseline_a:
            t0 = time.perf_counter()
            for q in queries:
                if q.id not in a_ids:
                    continue
                subset = [by_idx[i] for i in pools[q.id]]
                print("  baseline A (pooled) on {}: {} llm calls...".format(q.id, len(subset)))
                a_runs[q.id] = await baselines.baseline_a(
                    client, q.text, subset, POOL_DEPTH
                )
            a_wall = time.perf_counter() - t0

        # ---------------- judge ----------------
        # A query the judge could not grade is dropped from the aggregates and
        # named in the report, rather than aborting the whole evaluation or --
        # worse -- being scored as all-zeros. Eight judged queries with the
        # other four declared is an honest partial result; twelve queries where
        # four silently scored 0.000 is a fabricated one.
        grades: dict[str, dict[int, tuple[int, str]]] = {}
        unjudged: list[str] = []
        for q in queries:
            unique = pools[q.id]
            print("  judging {} ({} unique companies)...".format(q.id, len(unique)))
            try:
                grades[q.id] = await judge.judge_pool(
                    client, q.text, [by_idx[i] for i in unique]
                )
            except judge.JudgeUnavailable as exc:
                print("    SKIPPED {}: {}".format(q.id, str(exc)[:140]))
                unjudged.append(q.id)
        report["unjudged_queries"] = unjudged
        queries = [q for q in queries if q.id in grades]
        if not queries:
            raise SystemExit(
                "the judge could not grade any query -- almost certainly an "
                "exhausted API quota. Re-run later; judged batches are cached."
            )

        # ---------------- score ----------------
        per_system: dict[str, list[dict[str, float]]] = {
            "cascade": [], "baseline_a": [], "baseline_b": []
        }
        for q in queries:
            g = grades[q.id]
            pool_grades = [v[0] for v in g.values()]

            def graded(indices: list[int]) -> list[int]:
                return [g.get(i, (0, ""))[0] for i in indices]

            cascade_idx = [rc.company.idx for rc in cascade_by_query[q.id].companies]
            row: dict[str, Any] = {
                "query": q.text,
                "kind": q.kind,
                "note": q.note,
                "complexity_detected": cascade_by_query[q.id].spec.complexity,
                "pool_size": len(g),
                "relevant_in_pool": sum(1 for x in pool_grades if x >= metrics.RELEVANT_AT),
                "cascade": metrics.summarise(graded(cascade_idx), pool_grades),
                "baseline_b": metrics.summarise(
                    graded(b_runs[q.id]["ranked_idx"]), pool_grades
                ),
            }
            per_system["cascade"].append(row["cascade"])
            per_system["baseline_b"].append(row["baseline_b"])
            if q.id in a_runs:
                row["baseline_a"] = metrics.summarise(
                    graded(a_runs[q.id]["ranked_idx"]), pool_grades
                )
                per_system["baseline_a"].append(row["baseline_a"])

            # keep the graded detail for error analysis
            row["graded_detail"] = [
                {
                    "name": by_idx[i].name,
                    "location": by_idx[i].location(),
                    "grade": g.get(i, (0, ""))[0],
                    "judge_says": g.get(i, (0, ""))[1],
                    "cascade_rank": (cascade_idx.index(i) + 1) if i in cascade_idx else None,
                    "cascade_verdict": next(
                        (rc.qualification.verdict.value
                         for rc in cascade_by_query[q.id].companies if rc.company.idx == i),
                        None,
                    ),
                    "cascade_reason": next(
                        (rc.qualification.reason
                         for rc in cascade_by_query[q.id].companies if rc.company.idx == i),
                        None,
                    ),
                    "baseline_b_rank": (
                        b_runs[q.id]["ranked_idx"].index(i) + 1
                        if i in b_runs[q.id]["ranked_idx"] else None
                    ),
                }
                for i in g
            ]
            row["graded_detail"].sort(key=lambda d: -d["grade"])
            report["queries"][q.id] = row

        def mean_of(rows: list[dict[str, float]], key: str) -> float:
            vals = [r[key] for r in rows if key in r]
            return round(statistics.fmean(vals), 4) if vals else 0.0

        for system, rows in per_system.items():
            if not rows:
                continue
            report["systems"][system] = {
                k: mean_of(rows, k)
                for k in ("P@5", "P@10", "P@20", "nDCG@10", "nDCG@20", "R@20", "MAP", "returned")
            }
            report["systems"][system]["queries_scored"] = len(rows)

        report["cost"] = {
            "cascade": {
                "queries": len(queries),
                "wall_seconds": round(cascade_wall, 1),
                "llm_calls": sum(r.usage.calls for r in cascade),
                "cost_usd": round(sum(r.usage.cost_usd for r in cascade), 6),
                "cache_hits": sum(r.usage.cached_calls for r in cascade),
            },
            "baseline_b": {
                "queries": len(queries),
                "wall_seconds": round(b_wall, 1),
                "llm_calls": 0,
                "cost_usd": round(sum(r["cost_usd"] for r in b_runs.values()), 6),
            },
        }
        if a_runs:
            first = next(iter(a_runs.values()))
            report["baseline_a_projection"] = baselines.project_full_corpus_cost(
                first, len(companies)
            )
            report["baseline_a_mode"] = (
                "pooled: run over the judged candidate pool, not the full corpus"
            )
            report["cost"]["baseline_a"] = {
                "queries": len(a_runs),
                "wall_seconds": round(a_wall, 1),
                "llm_calls": sum(r["llm_calls"] for r in a_runs.values()),
                "cost_usd": round(sum(r["cost_usd"] for r in a_runs.values()), 6),
                "per_query_cost_usd": round(
                    statistics.fmean([r["cost_usd"] for r in a_runs.values()]), 6
                ),
                "per_query_seconds": round(
                    statistics.fmean([r["seconds"] for r in a_runs.values()]), 1
                ),
            }
        report["judge_model"] = config.JUDGE.primary
        report["cache_stats"] = cache.stats()

    return report


def render_markdown(report: dict[str, Any]) -> str:
    lines = ["# Evaluation report", ""]
    lines.append("Judge: `{}` | pool depth {} | relevance threshold grade >= {}".format(
        report.get("judge_model"), report["pool_depth"], report["relevance_threshold"]))
    lines.append("")
    lines.append("## Systems (mean over queries)")
    lines.append("")
    lines.append("| System | Queries | P@5 | P@10 | P@20 | nDCG@10 | nDCG@20 | MAP | Avg returned |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    label = {
        "cascade": "**Cascade (ours)**",
        "baseline_a": "Baseline A (LLM per company, pooled)",
        "baseline_b": "Baseline B (embedding only)",
    }
    for key in ("cascade", "baseline_a", "baseline_b"):
        s = report["systems"].get(key)
        if not s:
            continue
        lines.append("| {} | {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.1f} |".format(
            label[key], s["queries_scored"], s["P@5"], s["P@10"], s["P@20"],
            s["nDCG@10"], s["nDCG@20"], s["MAP"], s["returned"]))
    lines.append("")

    lines.append("## Cost and latency")
    lines.append("")
    lines.append("| System | Queries | Wall seconds | LLM calls | Cost USD |")
    lines.append("|---|---:|---:|---:|---:|")
    for key, c in report["cost"].items():
        lines.append("| {} | {} | {} | {} | ${:.5f} |".format(
            label.get(key, key), c["queries"], c["wall_seconds"], c["llm_calls"], c["cost_usd"]))
    lines.append("")

    lines.append("## Per-query")
    lines.append("")
    lines.append("| Query | Kind | Detected | Pool | Relevant | Cascade P@10 | B P@10 | Cascade nDCG@10 | B nDCG@10 |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for qid, row in report["queries"].items():
        lines.append("| {} | {} | {} | {} | {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(
            row["query"][:56], row["kind"], row["complexity_detected"], row["pool_size"],
            row["relevant_in_pool"], row["cascade"]["P@10"], row["baseline_b"]["P@10"],
            row["cascade"]["nDCG@10"], row["baseline_b"]["nDCG@10"]))
    lines.append("")
    if report.get("unjudged_queries"):
        lines.append("> **{} of 12 queries could not be judged** and are excluded from every".format(
            len(report["unjudged_queries"])))
        lines.append("> number above: `{}`. The judge hit an exhausted API quota.".format(
            ", ".join(report["unjudged_queries"])))
        lines.append("")
    lines.append("*Recall is measured against the judged pool, not the whole database: a")
    lines.append("relevant company that no system retrieved cannot appear in these numbers.*")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(config.DATA_DIR / "companies.jsonl"))
    p.add_argument("--parallel", type=int, default=2)
    p.add_argument("--skip-baseline-a", action="store_true")
    p.add_argument("--queries", help="comma-separated query ids to evaluate (default: all 12)")
    p.add_argument("--baseline-a-queries", help="comma-separated query ids, default q01,q04,q07")
    args = p.parse_args()

    report = asyncio.run(evaluate(args))
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_name = "eval_report" if not args.queries else "eval_report_subset"
    (config.RESULTS_DIR / (out_name + ".json")).write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    md = render_markdown(report)
    (config.RESULTS_DIR / (out_name + ".md")).write_text(md, encoding="utf-8")
    print("\n" + md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
