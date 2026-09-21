"""Stage 5 -- fuse every signal into one ordered list.

A verdict alone gives three buckets and no order inside them, which is
useless when forty companies all come back QUALIFIED. The final score is
mostly the verdict, modulated by how sure the model was, how much of the
rubric was actually satisfied, how much we had to take on trust because of
missing data, and -- only as a tiebreak -- how the cheap retrieval ranked it.

The bands are chosen so the ordering is monotonic and non-overlapping in the
important place: a confidently QUALIFIED company always outranks a
confidently PARTIAL one. Confidence re-orders within a verdict; it never
promotes across one. That keeps the ranking explainable, which matters more
here than squeezing out the last point of nDCG.
"""
from __future__ import annotations

from .schema import Candidate, Qualification, QuerySpec, RankedCompany, Verdict

# (floor, span) per verdict -- strictly non-overlapping, so the verdict alone
# determines the band and everything else only orders companies *within* it.
# This makes the invariant real rather than aspirational: no combination of
# confidence, rubric score or retrieval rank can lift a PARTIAL above a
# QUALIFIED, which keeps the ranking explainable to a user.
VERDICT_BANDS = {
    Verdict.QUALIFIED: (0.60, 0.40),
    Verdict.PARTIAL: (0.25, 0.30),
    Verdict.REJECTED: (0.00, 0.20),
}

# How the within-band position is composed. Confidence leads, but it cannot
# decide alone: the cheap model returns 1.0 for almost everything it accepts,
# which on one query tied fifty companies at exactly the same score. A list
# that is qualified but not *ranked* only does half the job, so retrieval
# strength and rubric coverage are given real weight to break those ties.
W_CONFIDENCE = 0.55
W_RETRIEVAL = 0.30
W_CRITERIA = 0.15


def criteria_satisfaction(qual: Qualification, spec: QuerySpec) -> float | None:
    """Weighted fraction of the rubric the model said was met.

    Returns None when the model reported nothing usable, so the caller can
    leave the score untouched rather than punish a company for the model's
    silence.
    """
    if not spec.criteria or not qual.criteria_met:
        return None
    by_name = {c.name.lower(): c for c in spec.criteria}
    total = matched = 0.0
    for name, met in qual.criteria_met.items():
        crit = by_name.get(str(name).lower())
        weight = crit.weight if crit else 0.5
        total += weight
        if met:
            matched += weight
    if total <= 0:
        return None
    return matched / total


def score_one(cand: Candidate, qual: Qualification, spec: QuerySpec, cfg, retrieval_norm: float) -> float:
    floor, span = VERDICT_BANDS[qual.verdict]

    # For a rejection, confidence counts against the company: a confident
    # reject sinks, an unsure one floats to the top of the reject band where
    # it stays visible in a debug dump.
    confidence_term = (
        1.0 - qual.confidence if qual.verdict is Verdict.REJECTED else qual.confidence
    )

    sat = criteria_satisfaction(qual, spec)
    # A company whose rubric coverage is unknown gets the neutral 0.5 rather
    # than a zero, so the model staying silent is not read as failure.
    criteria_term = 0.5 if sat is None else sat

    position = (
        W_CONFIDENCE * confidence_term
        + W_RETRIEVAL * max(0.0, min(1.0, retrieval_norm))
        + W_CRITERIA * criteria_term
    )

    # Each requirement we could never verify costs a little standing. This is
    # a discount on certainty, not a rejection: the company may well qualify,
    # we just cannot show that it does.
    position -= cfg.unknown_field_penalty * len(cand.unknown_constraints)

    # A verdict the stronger model confirmed is worth marginally more than one
    # only the cheap tier produced.
    if qual.escalated and qual.verdict is not Verdict.REJECTED:
        position += 0.03

    return floor + span * max(0.0, min(1.0, position))


def rank(
    candidates: list[Candidate],
    quals: dict[int, Qualification],
    spec: QuerySpec,
    cfg,
    include_rejected: bool = False,
) -> list[RankedCompany]:
    if not candidates:
        return []

    scores = [c.fused_score for c in candidates]
    lo, hi = min(scores), max(scores)
    spread = (hi - lo) or 1.0

    ranked: list[RankedCompany] = []
    for cand in candidates:
        qual = quals.get(cand.company.idx)
        if qual is None:
            # Never silently drop a candidate the LLM stage lost track of;
            # surface it at the bottom with an honest label instead.
            qual = Qualification(
                verdict=Verdict.PARTIAL, confidence=0.0, reason="no verdict returned",
                tier="missing",
            )
        norm = (cand.fused_score - lo) / spread
        ranked.append(
            RankedCompany(
                company=cand.company,
                score=score_one(cand, qual, spec, cfg, norm),
                qualification=qual,
                candidate=cand,
            )
        )

    ranked.sort(key=lambda r: (-r.score, r.company.name.lower()))
    if include_rejected:
        return ranked

    kept = [
        r for r in ranked
        if r.score >= cfg.min_score_to_return and r.qualification.verdict is not Verdict.REJECTED
    ]

    # Zero-result recovery. An empty list is almost never the honest answer
    # when hundreds of companies passed the hard filters -- it usually means
    # the planner over-narrowed the role and the qualifier dutifully rejected
    # everything. Returning the closest candidates, clearly relabelled as
    # near misses, is more useful than a blank page and makes the failure
    # visible instead of silent. The user can see the system found nothing it
    # would vouch for, and see what it nearly vouched for.
    if not kept and ranked and cfg.no_match_fallback > 0:
        for r in ranked[: cfg.no_match_fallback]:
            r.qualification.verdict = Verdict.PARTIAL
            r.qualification.reason = "closest available match; no company fully met the query: {}".format(
                r.qualification.reason
            )
            r.qualification.tier = (r.qualification.tier or "") + "+fallback"
        return ranked[: cfg.no_match_fallback]

    return kept
