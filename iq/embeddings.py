"""Corpus and query embeddings, computed once and cached to disk.

The corpus embedding is the only part of the system whose cost scales with
database size rather than with query complexity, and it is paid once for the
lifetime of the data, not once per query. At 477 companies it is about two
cents; at 100k it is about four dollars, still one-off.

Cache keying is content-addressed on the rendered document text, so editing
`company_document` invalidates exactly the rows whose text changed and keeps
the rest. That makes iterating on the document format cheap.
"""
from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np

from . import config
from .cache import DiskCache
from .llm import Gemini


def l2_normalise(matrix: np.ndarray) -> np.ndarray:
    """Row-normalise so a dot product is cosine similarity.

    Required, not cosmetic: at outputDimensionality < 3072 the Gemini
    embedding endpoint returns un-normalised vectors (measured ~0.59 L2 norm),
    so skipping this silently turns cosine into an unnormalised dot product
    that rewards whichever documents happen to have larger vectors.
    """
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def _text_key(text: str) -> str:
    return hashlib.sha256(
        "{}|{}|{}".format(config.EMBED_MODEL, config.EMBED_DIM, text).encode("utf-8")
    ).hexdigest()


async def embed_texts(
    client: Gemini,
    texts: Sequence[str],
    cache: DiskCache,
    task_type: str = "RETRIEVAL_DOCUMENT",
) -> np.ndarray:
    """Embed texts, hitting the cache per text and batching only the misses."""
    vectors: list[list[float] | None] = [None] * len(texts)
    missing: list[int] = []
    for i, text in enumerate(texts):
        hit = cache.get(_text_key(text) + ":" + task_type)
        if hit is not None:
            vectors[i] = hit
        else:
            missing.append(i)

    if missing:
        batches = [
            missing[i : i + config.EMBED_BATCH]
            for i in range(0, len(missing), config.EMBED_BATCH)
        ]

        async def run_batch(batch: list[int]) -> None:
            """Embed one batch and persist it immediately.

            Persisting per batch rather than after the whole gather matters:
            embedding the full corpus takes minutes behind the 100-per-minute
            rate limit, and writing only at the end means any single failure
            throws away every vector bought so far. Now an interrupted run
            resumes from wherever it stopped.
            """
            vecs = await client.embed_batch([texts[i] for i in batch], task_type)
            for i, vec in zip(batch, vecs):
                vectors[i] = vec
                cache.put(_text_key(texts[i]) + ":" + task_type, "embed", vec)

        await asyncio.gather(*[run_batch(b) for b in batches])

    matrix = np.array(
        [v if v is not None else [0.0] * config.EMBED_DIM for v in vectors], dtype=np.float32
    )
    return l2_normalise(matrix)


class CorpusIndex:
    """Embedded company corpus, persisted as a single .npz next to the data.

    A flat matrix with a brute-force dot product is the right call at this
    size: 477 x 768 is 1.4 MB and a full scan is well under a millisecond, so
    an ANN index would add a dependency, a build step and a recall cliff in
    exchange for nothing. The scaling section of the writeup describes the
    point at which that stops being true.
    """

    def __init__(self, documents: list[str], matrix: np.ndarray) -> None:
        self.documents = documents
        self.matrix = matrix

    @property
    def dim(self) -> int:
        return int(self.matrix.shape[1])

    @classmethod
    async def build(
        cls,
        client: Gemini,
        documents: list[str],
        cache: DiskCache,
        path: Path | None = None,
        rebuild: bool = False,
    ) -> "CorpusIndex":
        fingerprint = hashlib.sha256(
            "||".join(documents).encode("utf-8")
        ).hexdigest()[:16]
        path = path or (config.CACHE_DIR / "corpus-{}.npz".format(fingerprint))
        if path.exists() and not rebuild:
            with np.load(path) as store:
                return cls(documents, store["matrix"])
        matrix = await embed_texts(client, documents, cache, "RETRIEVAL_DOCUMENT")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, matrix=matrix)
        return cls(documents, matrix)

    async def embed_query(self, client: Gemini, text: str, cache: DiskCache) -> np.ndarray:
        """Embed one query-side text.

        Deliberately uses RETRIEVAL_DOCUMENT, not RETRIEVAL_QUERY. What we
        embed is the planner's synthetic *company profile*, which is a
        document in every respect that matters to the model; asking for a
        query-side encoding of it would place it in the wrong subspace and
        undo the benefit of generating it.
        """
        matrix = await embed_texts(client, [text], cache, "RETRIEVAL_DOCUMENT")
        return matrix[0]
