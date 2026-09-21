"""Ranking metrics and inter-rater agreement. Pure functions, no I/O.

Graded relevance on a 0-3 scale throughout:

    3 strong match   -- unambiguously what the user asked for
    2 good match     -- satisfies the intent with a minor caveat
    1 weak/adjacent  -- related but does not really satisfy the intent
    0 not a match

Binary precision treats >= 2 as relevant. The threshold is stated rather than
assumed because it is doing real work: the interesting disagreements in this
task live exactly on the 1/2 boundary, and a system can be made to look much
better by quietly moving it.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

RELEVANT_AT = 2


def precision_at_k(grades: Sequence[int], k: int, threshold: int = RELEVANT_AT) -> float:
    """Fraction of the top k that are relevant.

    Divides by min(k, len) rather than k, so a system returning 4 results, all
    correct, scores 1.0 at k=10 instead of 0.4. Penalising a system for
    honestly returning fewer results than the cap would reward padding the
    list with junk, which is the opposite of what this task wants.
    """
    if not grades:
        return 0.0
    window = list(grades[:k])
    return sum(1 for g in window if g >= threshold) / len(window)


def recall_at_k(
    grades: Sequence[int], total_relevant: int, k: int, threshold: int = RELEVANT_AT
) -> float:
    if total_relevant <= 0:
        return 0.0
    return sum(1 for g in grades[:k] if g >= threshold) / total_relevant


def dcg(grades: Sequence[int], k: int) -> float:
    return sum(
        (2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(list(grades)[:k])
    )


def ndcg_at_k(grades: Sequence[int], ideal_pool: Sequence[int], k: int) -> float:
    """nDCG against the best achievable ordering of the judged pool.

    `ideal_pool` is every grade known for this query across all systems, so
    the denominator reflects what was actually findable in the database, not
    just what this system happened to return.
    """
    best = dcg(sorted(ideal_pool, reverse=True), k)
    return (dcg(grades, k) / best) if best > 0 else 0.0


def average_precision(grades: Sequence[int], threshold: int = RELEVANT_AT) -> float:
    hits = 0
    total = 0.0
    for i, g in enumerate(grades, start=1):
        if g >= threshold:
            hits += 1
            total += hits / i
    return total / hits if hits else 0.0


def cohens_kappa(a: Sequence[int], b: Sequence[int]) -> float:
    """Agreement between two raters, corrected for chance.

    Reported instead of raw agreement because on a skewed pool two raters who
    both say "not a match" most of the time will agree 80% of the time by
    luck alone. Kappa is the honest number.
    """
    if len(a) != len(b) or not a:
        return float("nan")
    labels = sorted(set(a) | set(b))
    n = len(a)
    observed = sum(1 for x, y in zip(a, b) if x == y) / n
    expected = sum(
        (sum(1 for x in a if x == l) / n) * (sum(1 for y in b if y == l) / n)
        for l in labels
    )
    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else 0.0
    return (observed - expected) / (1 - expected)


def binary_scores(truth: Sequence[int], pred: Sequence[int]) -> dict[str, float]:
    """Precision/recall/F1 for two aligned binary label vectors."""
    tp = sum(1 for t, p in zip(truth, pred) if t and p)
    fp = sum(1 for t, p in zip(truth, pred) if not t and p)
    fn = sum(1 for t, p in zip(truth, pred) if t and not p)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": precision, "recall": recall, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn,
    }


def summarise(
    grades: Sequence[int], ideal_pool: Sequence[int], ks: Iterable[int] = (5, 10, 20)
) -> dict[str, float]:
    total_relevant = sum(1 for g in ideal_pool if g >= RELEVANT_AT)
    out: dict[str, float] = {"returned": float(len(grades))}
    for k in ks:
        out["P@{}".format(k)] = round(precision_at_k(grades, k), 4)
        out["nDCG@{}".format(k)] = round(ndcg_at_k(grades, ideal_pool, k), 4)
        out["R@{}".format(k)] = round(recall_at_k(grades, total_relevant, k), 4)
    out["MAP"] = round(average_precision(grades), 4)
    out["relevant_found"] = float(sum(1 for g in grades if g >= RELEVANT_AT))
    out["relevant_in_pool"] = float(total_relevant)
    return out
