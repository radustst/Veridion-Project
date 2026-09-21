#!/usr/bin/env python3
"""Check the LLM judge against hand-adjudicated labels.

  python -m evaluation.calibration --sample     # emit a blank labelling sheet
  python -m evaluation.calibration              # score labels against the judge

Why this exists
---------------
Every accuracy number in the writeup comes from an LLM judge, and the judge
is the same model family as the system it grades. Quoting those numbers
without checking them would be assuming the conclusion. So a stratified
sample of (query, company) pairs was labelled by hand, by reading the full
company record, and the judge is scored against those labels.

If judge-vs-human agreement is poor, the headline metrics are not
trustworthy and the writeup has to say so. That is the point: this module
exists to be able to find that out, not to confirm what we hope.

Sampling is stratified across the judge's own grade bands so the sheet is not
dominated by the obvious zeros, and it deliberately over-samples the 1/2
boundary, which is where the disagreements that matter actually live.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from evaluation import metrics
from iq import config
from iq.loading import company_card, load_companies

LABELS_PATH = config.ROOT / "evaluation" / "labels" / "human_labels.jsonl"
SHEET_PATH = config.ROOT / "evaluation" / "labels" / "labelling_sheet.md"

# Queries chosen to span the difficulty range: one pure structured filter, one
# supply-chain reasoning query, one multi-condition semantic query.
CALIBRATION_QUERIES = ["q01", "q04", "q07"]
PER_QUERY = 20


def build_sheet(report_path: Path, seed: int = 11) -> str:
    """Emit a Markdown sheet of pairs to label, with the judge's grade hidden."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    companies = {c.name: c for c in load_companies(config.DATA_DIR / "companies.jsonl")}
    rng = random.Random(seed)

    out = [
        "# Hand-labelling sheet",
        "",
        "Grade each pair 0-3 using the scale in `evaluation/judge.py`, reading only",
        "the record shown. The judge's grade is deliberately not displayed.",
        "Record answers in `evaluation/labels/human_labels.jsonl` as",
        '`{"query_id": ..., "company": ..., "grade": N, "note": "..."}`.',
        "",
    ]
    for qid in CALIBRATION_QUERIES:
        row = report["queries"].get(qid)
        if not row:
            continue
        detail = row["graded_detail"]
        # Stratify: sample from each judge band so the sheet spans the range,
        # weighting the contested 1/2 boundary most heavily.
        bands: dict[int, list[dict[str, Any]]] = {0: [], 1: [], 2: [], 3: []}
        for d in detail:
            bands[d["grade"]].append(d)
        quota = {3: 5, 2: 6, 1: 6, 0: 3}
        picked: list[dict[str, Any]] = []
        for grade, want in quota.items():
            pool = bands[grade]
            rng.shuffle(pool)
            picked.extend(pool[:want])
        # top up from anywhere if a band was thin
        if len(picked) < PER_QUERY:
            rest = [d for d in detail if d not in picked]
            rng.shuffle(rest)
            picked.extend(rest[: PER_QUERY - len(picked)])
        picked = picked[:PER_QUERY]
        rng.shuffle(picked)

        out.append("## {} — {}".format(qid, row["query"]))
        out.append("")
        for d in picked:
            company = companies.get(d["name"])
            out.append("### {}".format(d["name"]))
            out.append("")
            out.append("```")
            out.append(company_card(company) if company else "(record not found)")
            out.append("```")
            out.append("")
    return "\n".join(out)


def load_labels() -> list[dict[str, Any]]:
    if not LABELS_PATH.exists():
        return []
    rows = []
    for line in LABELS_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("//"):
            rows.append(json.loads(line))
    return rows


def score(report_path: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    labels = load_labels()
    if not labels:
        raise SystemExit(
            "no labels found at {}. Run with --sample first.".format(LABELS_PATH)
        )

    judge_grades: list[int] = []
    human_grades: list[int] = []
    disagreements: list[dict[str, Any]] = []

    for row in labels:
        qrow = report["queries"].get(row["query_id"])
        if not qrow:
            continue
        match = next(
            (d for d in qrow["graded_detail"] if d["name"] == row["company"]), None
        )
        if match is None:
            continue
        judge_grades.append(int(match["grade"]))
        human_grades.append(int(row["grade"]))
        if abs(int(match["grade"]) - int(row["grade"])) >= 1:
            disagreements.append({
                "query_id": row["query_id"],
                "company": row["company"],
                "judge": match["grade"],
                "human": row["grade"],
                "judge_says": match["judge_says"],
                "human_note": row.get("note", ""),
                "cascade_verdict": match.get("cascade_verdict"),
            })

    n = len(judge_grades)
    exact = sum(1 for a, b in zip(judge_grades, human_grades) if a == b) / n if n else 0.0
    within_one = (
        sum(1 for a, b in zip(judge_grades, human_grades) if abs(a - b) <= 1) / n if n else 0.0
    )
    jb = [1 if g >= metrics.RELEVANT_AT else 0 for g in judge_grades]
    hb = [1 if g >= metrics.RELEVANT_AT else 0 for g in human_grades]

    return {
        "pairs_labelled": n,
        "exact_agreement": round(exact, 4),
        "within_one_grade": round(within_one, 4),
        "kappa_graded": round(metrics.cohens_kappa(judge_grades, human_grades), 4),
        "kappa_binary": round(metrics.cohens_kappa(hb, jb), 4),
        "judge_vs_human_binary": metrics.binary_scores(hb, jb),
        "judge_mean_grade": round(sum(judge_grades) / n, 3) if n else 0.0,
        "human_mean_grade": round(sum(human_grades) / n, 3) if n else 0.0,
        "disagreements": sorted(
            disagreements, key=lambda d: -abs(d["judge"] - d["human"])
        ),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--report", default=str(config.RESULTS_DIR / "eval_report.json"))
    p.add_argument("--sample", action="store_true", help="write a blank labelling sheet")
    args = p.parse_args()
    report_path = Path(args.report)

    if args.sample:
        SHEET_PATH.parent.mkdir(parents=True, exist_ok=True)
        SHEET_PATH.write_text(build_sheet(report_path), encoding="utf-8")
        print("wrote {}".format(SHEET_PATH))
        return 0

    result = score(report_path)
    out = config.RESULTS_DIR / "calibration.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "disagreements"}, indent=2))
    print("\n{} disagreements (worst first):".format(len(result["disagreements"])))
    for d in result["disagreements"][:12]:
        print("  {:<6} {:<34} judge={} human={}  {}".format(
            d["query_id"], d["company"][:34], d["judge"], d["human"], d["human_note"][:60]))
    print("\nwrote {}".format(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
