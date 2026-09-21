"""LLM-as-judge: graded relevance labels for (query, company) pairs.

Deliberately kept ignorant of the system under test. The judge sees the raw
user query and the company record -- never the planner's rubric, role
statement, disqualifiers, the pipeline's verdict, or which system surfaced
the company. If the judge were shown the planner's interpretation it would be
grading the pipeline against its own reading of the query, which is circular
and would make a mis-planned query look perfectly answered.

Two things this does NOT solve, both measured rather than glossed over in the
writeup:

  1. Same-family bias. The judge and the qualifier are both Gemini, so shared
     blind spots survive. The pro tier was unavailable on this key, so the
     judge is the strongest reachable model and merely a *different* one.
  2. Self-preference. An LLM judge tends to agree with LLM-written reasoning.
     This is why the hand-adjudicated calibration set exists: it measures how
     far the judge can be trusted before any of its numbers are quoted.
"""
from __future__ import annotations

import asyncio
import random
from typing import Any, Sequence

from iq import config
from iq.llm import Gemini, LLMError
from iq.loading import company_card
from iq.schema import Company

# "low" rather than "high": measured on this data the grades were
# indistinguishable, while high thinking produced ~1400 output tokens per
# batch and made the judge the slowest stage of the whole evaluation.
JUDGE_THINKING = "low"

JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "grades": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "grade": {"type": "integer"},
                    "justification": {"type": "string"},
                },
                "required": ["id", "grade", "justification"],
            },
        }
    },
    "required": ["grades"],
}

SYSTEM = """You are an exacting evaluator of business search results. You \
grade how well a company satisfies a user's search intent. You are not \
generous: topical adjacency is not a match, and serving an industry is not \
the same as belonging to it."""

PROMPT = """Grade each company against this search query.

QUERY: {query}

GRADING SCALE
3 = Strong match. Unambiguously what the user asked for. All stated
    constraints (location, size, revenue, ownership, industry, role) hold.
2 = Good match. Satisfies the intent, with one minor caveat -- e.g. a stated
    numeric constraint cannot be verified from the record but nothing
    contradicts it, or the company does the required thing among other things.
1 = Weak. Topically related but does not satisfy the intent. A supplier to
    the industry rather than in it, a software vendor serving the sector, a
    company in the wrong country, or one that plainly violates a stated
    numeric constraint.
0 = Not a match. Different business entirely.

RULES
- Judge only from the record shown. Do not use outside knowledge of the
  company beyond what a reasonable reader would infer from these fields.
- A field showing "unknown" is not evidence against the company. If every
  other signal fits and only an unverifiable field is in question, grade 2.
- If a stated numeric or location constraint is present in the record and is
  violated, the grade is at most 1, regardless of how good the fit otherwise is.
- justification: at most 20 words, naming the deciding evidence.

COMPANIES
{companies}

Return one grade per company using the id shown."""


class JudgeUnavailable(RuntimeError):
    """Raised when the judge could not grade anything at all.

    This exists because the first version swallowed the error and returned an
    empty dict, which propagated as an empty pool and a table of 0.000 scores
    that looked exactly like a real result showing nothing was relevant. A
    silent zero is worse than a crash: it is a confident wrong answer about
    the evaluation itself, which is the same failure mode the writeup warns
    about in the system under test.
    """


async def _grade_batch(
    client: Gemini, query: str, batch: Sequence[Company]
) -> dict[int, tuple[int, str]]:
    cards = "\n\n".join(
        "--- id {} ---\n{}".format(i, company_card(c)) for i, c in enumerate(batch, start=1)
    )
    try:
        data = await client.generate_json(
            config.JUDGE,
            PROMPT.format(query=query, companies=cards),
            JUDGE_SCHEMA,
            system=SYSTEM,
            namespace="judge",
            thinking=JUDGE_THINKING,
        )
    except LLMError as exc:
        raise JudgeUnavailable(str(exc)[:200]) from exc

    out: dict[int, tuple[int, str]] = {}
    for item in data.get("grades") or []:
        try:
            local = int(item.get("id", 0))
            grade = int(item.get("grade", 0))
        except (TypeError, ValueError):
            continue
        if 1 <= local <= len(batch):
            out[batch[local - 1].idx] = (
                max(0, min(3, grade)),
                str(item.get("justification", "")).strip(),
            )
    return out


async def judge_pool(
    client: Gemini,
    query: str,
    companies: Sequence[Company],
    batch_size: int = 8,
    seed: int = 17,
) -> dict[int, tuple[int, str]]:
    """Grade a pool of companies for one query.

    The pool is shuffled with a fixed seed before batching. Shuffling breaks
    the correlation between a company's position in a judging batch and the
    rank the system under test gave it, so the judge cannot infer "this one
    was ranked first" from context and reward it for that. Fixing the seed
    keeps the whole evaluation reproducible.
    """
    pool = list(companies)
    random.Random(seed).shuffle(pool)
    batches = [pool[i : i + batch_size] for i in range(0, len(pool), batch_size)]
    results = await asyncio.gather(
        *[_grade_batch(client, query, b) for b in batches], return_exceptions=True
    )
    merged: dict[int, tuple[int, str]] = {}
    failures = 0
    for r in results:
        if isinstance(r, BaseException):
            failures += 1
            continue
        merged.update(r)
    if not merged and pool:
        raise JudgeUnavailable(
            "judge graded 0 of {} companies for {!r} ({} batch failures)".format(
                len(pool), query[:60], failures
            )
        )
    if failures:
        print("    warning: {} of {} judge batches failed for this query".format(
            failures, len(batches)))
    return merged
