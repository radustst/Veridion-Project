"""Stage 2 -- cheap multi-signal candidate retrieval. No LLM calls.

Three independent, complementary signals, fused by rank rather than score:

  dense  -- cosine similarity against the planner's synthetic ideal company
            profile. Catches paraphrase and role similarity; blind to
            explicit codes and rare proper nouns.
  bm25   -- lexical match on the planner's keywords. Catches "Shopify",
            "airless pump", "customs brokerage" -- the exact tokens dense
            retrieval smooths away; blind to synonymy.
  naics  -- prefix overlap with the planner's predicted industry codes. A
            curated, human-auditable taxonomy signal that is completely
            independent of how the description happens to be written, and so
            rescues companies with a thin or missing description.

They are fused with Reciprocal Rank Fusion. RRF consumes *ranks*, not scores,
which matters because a cosine of 0.71 and a BM25 of 14.3 live on
incomparable scales and any attempt to normalise them into a weighted sum
needs per-query calibration that we have no labels to fit. RRF needs none, is
robust to one signal being uninformative, and is a single line of code.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, Sequence

import numpy as np

from .schema import Candidate, FilterOutcome, QuerySpec

TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    """a an and are as at be by for from has have in is it its of on or that the
    to was were will with company companies business businesses service services
    provide provides providing solution solutions product products""".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS and len(t) > 1]


class BM25:
    """Okapi BM25 over the company corpus.

    Implemented here rather than pulled from a package: it is thirty lines,
    it removes a dependency, and having it inline makes the k1/b choices and
    the tokenizer visible to anyone reading the retrieval logic.
    """

    def __init__(self, corpus: Sequence[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.docs: list[Counter[str]] = [Counter(tokenize(d)) for d in corpus]
        self.lengths = np.array([sum(d.values()) for d in self.docs], dtype=np.float32)
        self.avg_len = float(self.lengths.mean()) if len(self.lengths) else 0.0
        n = len(self.docs)
        df: Counter[str] = Counter()
        for d in self.docs:
            df.update(d.keys())
        # Robertson/Sparck-Jones idf with the +1 guard that keeps it positive
        # for terms appearing in more than half the corpus.
        self.idf = {
            term: math.log(1.0 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()
        }

    def score(self, query_terms: Iterable[str]) -> np.ndarray:
        terms = [t for t in query_terms if t in self.idf]
        out = np.zeros(len(self.docs), dtype=np.float32)
        if not terms or self.avg_len == 0:
            return out
        norm = self.k1 * (1 - self.b + self.b * self.lengths / self.avg_len)
        for term in terms:
            idf = self.idf[term]
            tf = np.array([d.get(term, 0) for d in self.docs], dtype=np.float32)
            out += idf * (tf * (self.k1 + 1)) / (tf + norm)
        return out


def naics_affinity(company_codes: Sequence[str], prefixes: Sequence[str]) -> float:
    """Graded prefix match in [0, 1], rewarding longer agreement.

    A 6-digit exact match is near-certain evidence; a shared 2-digit sector is
    weak evidence. Scoring by matched-prefix length rather than a boolean
    keeps that distinction instead of flattening it.
    """
    if not company_codes or not prefixes:
        return 0.0
    best = 0.0
    for code in company_codes:
        for prefix in prefixes:
            if code.startswith(prefix):
                # 2 digits -> 0.35, 3 -> 0.52, 4 -> 0.70, 5 -> 0.87, 6 -> 1.0
                best = max(best, min(1.0, 0.35 + 0.165 * (len(prefix) - 2)))
    return best


def rrf(ranked_indices: Sequence[Sequence[int]], weights: Sequence[float], k: int, size: int) -> np.ndarray:
    """Weighted Reciprocal Rank Fusion over several ranked index lists."""
    scores = np.zeros(size, dtype=np.float32)
    for indices, weight in zip(ranked_indices, weights):
        if weight <= 0:
            continue
        for rank, idx in enumerate(indices):
            scores[idx] += weight / (k + rank + 1)
    return scores


def retrieve(
    survivors: list[FilterOutcome],
    spec: QuerySpec,
    doc_embeddings: np.ndarray,
    query_embedding: np.ndarray,
    documents: list[str],
    top_k: int,
    cfg,
) -> list[Candidate]:
    """Rank the filtered companies and return the top_k as Candidates.

    `doc_embeddings` is the full corpus matrix; `survivors` indexes into it by
    Company.idx, so the expensive embedding work is done once for the whole
    database and reused by every query.
    """
    if not survivors:
        return []

    idxs = [o.company.idx for o in survivors]
    n = len(idxs)

    # --- dense ---
    sub = doc_embeddings[idxs]  # rows are already L2-normalised
    dense = sub @ query_embedding if query_embedding is not None else np.zeros(n, dtype=np.float32)

    # --- lexical ---
    bm25 = BM25([documents[i] for i in idxs])
    query_terms: list[str] = []
    for source in (spec.keywords, [spec.role_statement], [spec.ideal_profile]):
        for text in source:
            query_terms.extend(tokenize(text))
    lexical = bm25.score(query_terms)

    # --- taxonomy ---
    naics = np.array(
        [naics_affinity(o.company.naics_codes, spec.naics_prefixes) for o in survivors],
        dtype=np.float32,
    )

    order = lambda arr: list(np.argsort(-arr, kind="stable"))
    fused = rrf(
        [order(dense), order(lexical), order(naics)],
        [cfg.weight_dense, cfg.weight_bm25, cfg.weight_naics if spec.naics_prefixes else 0.0],
        cfg.rrf_k,
        n,
    )

    candidates = [
        Candidate(
            company=survivors[i].company,
            dense_score=float(dense[i]),
            bm25_score=float(lexical[i]),
            naics_score=float(naics[i]),
            fused_score=float(fused[i]),
            unknown_constraints=survivors[i].unknowns,
        )
        for i in range(n)
    ]
    candidates.sort(key=lambda c: -c.fused_score)
    return candidates[:top_k]
