"""Stages 3 and 4 -- batched LLM qualification, then selective escalation.

Three mechanisms make this cheap without making it dumb:

1. Batching. The rubric, disqualifiers and instructions are the bulk of the
   prompt and are identical for every company in a query. Sending them once
   per batch of eight instead of once per company removes roughly 80% of the
   input tokens, which is where the money is.

2. Tiering. The cheap model sees every shortlisted company. The stronger model
   sees only the ones the cheap model was not sure about. Spending the whole
   budget uniformly is what makes Baseline A expensive; spending it where the
   decision is actually in doubt is what makes this affordable.

3. Determinism. temperature=0 plus a response schema plus a disk cache means
   the same question asked twice gives the same answer, which is the third
   complaint about the naive baseline and the one that is easiest to fix.

Batching does introduce a real risk -- position bias and cross-contamination
between companies sharing a prompt -- which is measured in the writeup rather
than waved away.
"""
from __future__ import annotations

import asyncio
from typing import Any, Sequence

from . import config
from .loading import company_card
from .llm import Gemini, LLMError
from .schema import Candidate, Qualification, QuerySpec, Verdict

QUALIFY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "verdict": {"type": "string", "enum": ["QUALIFIED", "PARTIAL", "REJECTED"]},
                    "confidence": {"type": "number"},
                    "reason": {"type": "string"},
                    "criteria_met": {"type": "array", "items": {"type": "string"}},
                    "criteria_failed": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["id", "verdict", "confidence", "reason",
                             "criteria_met", "criteria_failed"],
            },
        }
    },
    "required": ["results"],
}

SYSTEM = """You qualify companies against a buyer's search intent. You are \
strict: topical adjacency is not a match. You judge only from the record \
given, you say so when the record is insufficient, and you never invent \
facts about a company."""

PROMPT = """{header}

DECISION RULES
- QUALIFIED: the company clearly performs the required role and meets every
  must-have criterion.
- PARTIAL: plausibly relevant but something essential is unproven -- the
  record is thin, the role is adjacent, or a must-have cannot be confirmed
  from the fields given.
- REJECTED: the company does not perform the required role, or it definitively
  fails a must-have criterion.

- Judge the role the company PLAYS, not the industry it is merely near. A
  vendor selling software to an industry belongs to the software industry.
- {unknown_rule}
- confidence is your certainty in the verdict itself, 0.0 to 1.0. Use the
  middle of the range honestly when the record is genuinely ambiguous; do not
  default to 0.9.
- reason: one sentence, maximum 30 words, citing the specific evidence in the
  record that decided it. No hedging language.
- Judge each company only against the query. Do not compare them to each other
  or ration your verdicts -- it is fine for all of them to qualify, or none.

COMPANIES
{companies}

Return one result per company, using the id shown."""


def _header(spec: QuerySpec) -> str:
    """The query-specific preamble shared by every company in every batch."""
    lines = ["USER QUERY: {}".format(spec.query)]
    if spec.intent_summary:
        lines.append("INTENT: {}".format(spec.intent_summary))
    if spec.role_statement:
        lines.append("A MATCHING COMPANY MUST: {}".format(spec.role_statement))

    hard: list[str] = []
    if spec.countries:
        hard.append("located in {}".format(spec.country_phrase or ", ".join(spec.countries)))
    if spec.employees.is_set():
        hard.append("employees {}".format(spec.employees.describe()))
    if spec.revenue_usd.is_set():
        hard.append("revenue USD {}".format(spec.revenue_usd.describe()))
    if spec.founded_year.is_set():
        hard.append("founded {}".format(spec.founded_year.describe()))
    if spec.is_public is not None:
        hard.append("publicly traded" if spec.is_public else "privately held")
    if hard:
        lines.append(
            "STRUCTURAL REQUIREMENTS (already pre-filtered where the data exists; "
            "re-check only where a field is unknown): " + "; ".join(hard)
        )

    if spec.criteria:
        lines.append("CRITERIA:")
        for c in spec.criteria:
            lines.append(
                "  - {} [{}, weight {:.2f}]: {}".format(
                    c.name, "MUST-HAVE" if c.must_have else "nice-to-have", c.weight, c.description
                )
            )
    if spec.disqualifiers:
        lines.append("DO NOT BE FOOLED BY:")
        lines.extend("  - {}".format(d) for d in spec.disqualifiers)
    return "\n".join(lines)


def _unknown_rule(spec: QuerySpec) -> str:
    if not spec.has_hard_constraints():
        return (
            "If the record is too thin to judge the role confidently, return "
            "PARTIAL with low confidence rather than guessing."
        )
    return (
        "Some records show 'unknown' for employees, revenue, founding year or "
        "public status. Unknown is NOT a failure. Infer the likely value from "
        "the description and offerings where you reasonably can, and if you "
        "still cannot tell, return PARTIAL rather than REJECTED -- reserve "
        "REJECTED for companies you can affirmatively rule out."
    )


def _render_batch(batch: Sequence[Candidate], spec: QuerySpec) -> str:
    """Render companies, flagging exactly which fields could not be checked."""
    blocks = []
    for i, cand in enumerate(batch, start=1):
        card = company_card(cand.company)
        if cand.unknown_constraints:
            card += "\n[unverifiable from data: {}]".format(", ".join(cand.unknown_constraints))
        blocks.append("--- id {} ---\n{}".format(i, card))
    return "\n\n".join(blocks)


def _parse(
    data: dict[str, Any], batch: Sequence[Candidate], spec: QuerySpec, tier_name: str,
    escalated: bool = False,
) -> dict[int, Qualification]:
    """Map the model's per-id results back onto candidates by company index."""
    known = {c.name.lower() for c in spec.criteria}
    out: dict[int, Qualification] = {}
    for item in data.get("results") or []:
        try:
            local_id = int(item.get("id", 0))
        except (TypeError, ValueError):
            continue
        if not 1 <= local_id <= len(batch):
            continue
        company = batch[local_id - 1].company

        raw_verdict = str(item.get("verdict", "REJECTED")).upper()
        verdict = Verdict.__members__.get(raw_verdict, Verdict.REJECTED)
        try:
            conf = float(item.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        conf = min(1.0, max(0.0, conf))

        met: dict[str, bool] = {}
        for name in item.get("criteria_met") or []:
            if str(name).lower() in known or not known:
                met[str(name)] = True
        for name in item.get("criteria_failed") or []:
            if str(name).lower() in known or not known:
                met[str(name)] = False

        out[company.idx] = Qualification(
            verdict=verdict,
            confidence=conf,
            reason=str(item.get("reason", "")).strip(),
            criteria_met=met,
            escalated=escalated,
            tier=tier_name,
        )
    return out


async def _run_batch(
    client: Gemini, batch: Sequence[Candidate], spec: QuerySpec,
    tier: config.ModelTier, namespace: str, thinking: str | None, escalated: bool,
) -> dict[int, Qualification]:
    prompt = PROMPT.format(
        header=_header(spec),
        unknown_rule=_unknown_rule(spec),
        companies=_render_batch(batch, spec),
    )
    try:
        data = await client.generate_json(
            tier, prompt, QUALIFY_SCHEMA, system=SYSTEM,
            namespace=namespace, thinking=thinking,
        )
    except LLMError as exc:
        # A dead batch must not kill the query. Mark its companies PARTIAL at
        # zero confidence so they rank below anything actually adjudicated but
        # remain visible, and record why.
        return {
            c.company.idx: Qualification(
                verdict=Verdict.PARTIAL, confidence=0.0,
                reason="not adjudicated: {}".format(str(exc)[:120]),
                tier="error",
            )
            for c in batch
        }
    return _parse(data, batch, spec, tier.primary, escalated)


async def qualify(
    client: Gemini, candidates: list[Candidate], spec: QuerySpec, cfg: config.PipelineConfig
) -> dict[int, Qualification]:
    """Stage 3: adjudicate every candidate on the cheap tier, concurrently."""
    if not candidates:
        return {}
    size = cfg.qualify_batch_size
    batches = [candidates[i : i + size] for i in range(0, len(candidates), size)]
    results = await asyncio.gather(
        *[
            _run_batch(client, b, spec, config.QUALIFIER, "qualify", "low", False)
            for b in batches
        ]
    )
    merged: dict[int, Qualification] = {}
    for r in results:
        merged.update(r)
    return merged


def needs_escalation(
    cand: Candidate, qual: Qualification, spec: QuerySpec, cfg: config.PipelineConfig,
    retrieval_rank: int, total: int,
) -> bool:
    """Decide whether a second, stronger opinion is worth paying for.

    Four triggers, all of them evidence that the cheap verdict is unreliable
    rather than merely negative:

      a) stated confidence sits in the uncertain band;
      b) the verdict is PARTIAL, which is the model saying so explicitly;
      c) retrieval and the LLM disagree -- a company retrieval ranked in the
         top decile but the LLM rejected, or vice versa. Disagreement between
         two independent signals is the classic cheap uncertainty proxy;
      d) the company was accepted despite a constraint we could never verify,
         so the verdict rests on an inference rather than on data.
    """
    if cfg.escalate_conf_low <= qual.confidence <= cfg.escalate_conf_high:
        return True
    if qual.verdict is Verdict.PARTIAL:
        return True
    if qual.tier == "error":
        return True
    top_decile = max(1, total // 10)
    if retrieval_rank < top_decile and qual.verdict is Verdict.REJECTED:
        return True
    if retrieval_rank > total * 0.6 and qual.verdict is Verdict.QUALIFIED:
        return True
    if cand.unknown_constraints and qual.verdict is not Verdict.REJECTED:
        return True
    return False


async def escalate(
    client: Gemini,
    candidates: list[Candidate],
    quals: dict[int, Qualification],
    spec: QuerySpec,
    cfg: config.PipelineConfig,
) -> tuple[dict[int, Qualification], int]:
    """Stage 4: re-adjudicate only the uncertain candidates on a stronger model.

    The cap on escalated fraction is deliberate. Without it, a badly planned
    query where the cheap tier is unsure about everything would quietly cost
    as much as Baseline A -- the exact failure this architecture exists to
    avoid. Bounding the blast radius keeps the cost predictable even when the
    system is confused, and the bound being hit is itself a signal worth
    logging.
    """
    if not cfg.escalate_enabled or not candidates:
        return quals, 0

    total = len(candidates)
    flagged = [
        (rank, cand)
        for rank, cand in enumerate(candidates)
        if needs_escalation(
            cand, quals.get(cand.company.idx, Qualification()), spec, cfg, rank, total
        )
    ]
    # Prefer the most uncertain when the cap bites: closest to 0.5 first.
    flagged.sort(key=lambda rc: abs(quals.get(rc[1].company.idx, Qualification()).confidence - 0.5))
    budget = max(1, int(total * cfg.escalate_max_fraction))
    chosen = [cand for _, cand in flagged[:budget]]
    if not chosen:
        return quals, 0

    size = cfg.escalate_batch_size
    batches = [chosen[i : i + size] for i in range(0, len(chosen), size)]
    results = await asyncio.gather(
        *[
            _run_batch(client, b, spec, config.ESCALATION, "escalate", cfg.escalate_thinking, True)
            for b in batches
        ]
    )
    merged = dict(quals)
    for r in results:
        for idx, q in r.items():
            if q.tier == "error":
                continue  # keep the cheap verdict rather than losing it
            merged[idx] = q
    return merged, len(chosen)
