# Intent Qualification

Given a natural-language query and a database of company profiles, decide
which companies *actually satisfy the user's intent* — not merely which ones
look similar to the query.

The design rationale, measured results, error analysis and scaling plan are in
**[WRITEUP.md](WRITEUP.md)**. This file is just how to run it.

## The short version

```
query ──▶ ① plan ──▶ ② filter ──▶ ③ retrieve ──▶ ④ qualify ──▶ ⑤ escalate ──▶ ⑥ rank ──▶ results
          1 LLM      free,        free,          cheap LLM,     stronger LLM,   free
          call       deterministic no LLM        batched        ~10% of shortlist
```

Cost per query is roughly flat in database size, because only stages ② and ③
touch every company and neither of them calls a model.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # then put your Google AI Studio key in it
```

The key needs access to `gemini-embedding-001` and the `gemini-3.x-flash`
family. Get one at https://aistudio.google.com/apikey.

## Run it

```bash
# one query, showing the query plan the system built
python solution.py --query "Logistic companies in Romania" --explain

# all 12 benchmark queries -> results/results.json + results/results.md
python solution.py --benchmark

# build the corpus embedding on its own (about 5 minutes on the free tier,
# rate-limited to 100 embeddings/minute; cached afterwards)
python solution.py --warm
```

First run embeds all 456 companies (477 rows, minus 21 duplicate records) and
caches them to `cache/`. Later runs reuse that, so they start instantly.

Useful flags: `--no-escalate` (cheap tier only), `--shortlist N` (override how
many candidates reach the LLM), `--no-cache`, `--limit N`, `--parallel N`.

## Evaluate it

```bash
python -m evaluation.run_eval                    # cascade vs both baselines
python -m evaluation.run_eval --skip-baseline-a  # skip the expensive baseline
python -m evaluation.calibration                 # judge vs hand-written labels
python -m pytest tests/ -q                       # 68 offline unit tests
```

`run_eval` scores all three systems against a pooled, blind LLM judge and
writes `results/eval_report.md`. `calibration` checks that judge against
hand-adjudicated labels — read that before believing any of the numbers.

## Layout

```
solution.py              CLI entry point
iq/
  config.py              models, thresholds, rate limits — every tunable knob
  schema.py              typed objects passed between stages
  loading.py             dataset parsing + the two text renderings
  geo.py                 deterministic region/country resolution
  planner.py             ① query -> QuerySpec (the one LLM call per query)
  filters.py             ② three-valued hard constraints
  retrieval.py           ③ BM25 + dense + NAICS, fused with RRF
  qualifier.py           ④⑤ batched qualification and selective escalation
  ranking.py             ⑥ final score fusion
  pipeline.py            orchestration
  llm.py                 Gemini client: retries, fallback, rate limiting
  cache.py               content-addressed sqlite cache
evaluation/
  queries.py             the 12 benchmark queries, tagged by difficulty
  baselines.py           Baseline A (LLM per company), Baseline B (embeddings)
  judge.py               blind LLM judge, graded relevance 0-3
  metrics.py             P@k, nDCG, MAP, Cohen's kappa
  run_eval.py            benchmark driver
  calibration.py         judge-vs-human agreement
  consistency.py         run-to-run stability with the cache disabled
  labels/                hand-adjudicated relevance labels
tests/                   offline unit tests for the deterministic components
```

## Notes

- Everything is cached to `cache/iq.sqlite` keyed by content hash, so a rerun
  of an unchanged configuration is free and byte-identical. Delete the file to
  force a cold run.
- Free-tier Gemini limits are tight and enforced client-side by a token
  bucket: 100 embedding requests/minute, where each *text* in a batch counts
  as one request.
- No vector database. At 456 companies a flat numpy matrix scan takes under a
  millisecond; WRITEUP.md covers when that stops being true.
- The free tier is tight: the larger `gemini-3.x-flash` models have a small
  daily allowance and return 429 once it is gone. Every model chain in
  `iq/config.py` therefore ends in a lite model, rate-limited models enter a
  cooldown, and a failed query plan degrades to a heuristic rather than
  aborting the run.   
