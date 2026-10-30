"""Chunked vector-RAG baseline: ~128-token chunks in an in-memory Qdrant (or numpy) index."""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from metrics import TokenCounter

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams
except ImportError:  # optional dependency; the numpy backend needs nothing extra
    QdrantClient = None

COLLECTION = "chunks"
_SENTENCE = re.compile(r"(?<=[.!?])\s+")

Embed = Callable[[str], np.ndarray]


@dataclass(frozen=True)
class Chunk:
    dataset_id: str
    text: str
    tokens: int


def chunk_document(dataset_id: str, text: str, counter: TokenCounter, max_tokens: int = 128) -> list[Chunk]:
    """Packs whole sentences into chunks of at most ``max_tokens``; overlong sentences are split by tokens."""
    pieces: list[str] = []
    for paragraph in filter(None, (p.strip() for p in text.split("\n\n"))):
        for sentence in filter(None, _SENTENCE.split(paragraph)):
            tokens = counter.encode(sentence)
            if len(tokens) <= max_tokens:
                pieces.append(sentence)
            else:
                pieces += [counter.decode(tokens[i : i + max_tokens]) for i in range(0, len(tokens), max_tokens)]
    chunks: list[Chunk] = []
    current: list[str] = []
    for piece in pieces:
        candidate = " ".join([*current, piece])
        if current and counter.count(candidate) > max_tokens:
            body = " ".join(current)
            chunks.append(Chunk(dataset_id, body, counter.count(body)))
            current = [piece]
        else:
            current.append(piece)
    if current:
        body = " ".join(current)
        chunks.append(Chunk(dataset_id, body, counter.count(body)))
    return chunks


class VectorBaseline:
    """Top-k cosine search over chunk embeddings."""

    def __init__(self, chunks: Sequence[Chunk], embed: Embed, *, backend: str = "qdrant") -> None:
        if backend == "qdrant" and QdrantClient is None:
            raise RuntimeError("qdrant-client is not installed; use --baseline-backend numpy")
        self.chunks = list(chunks)
        self.backend = backend
        vectors = np.stack([embed(c.text) for c in self.chunks]).astype(np.float32)
        self.dim = vectors.shape[1]
        if backend == "qdrant":
            self._qdrant = QdrantClient(":memory:")
            self._qdrant.create_collection(
                COLLECTION, vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE)
            )
            self._qdrant.upsert(
                COLLECTION,
                points=[PointStruct(id=i, vector=v.tolist(), payload={}) for i, v in enumerate(vectors)],
                wait=True,
            )
        else:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            self._matrix = vectors / np.where(norms == 0, 1, norms)

    def search(self, query: np.ndarray, k: int) -> list[Chunk]:
        if self.backend == "qdrant":
            hits = self._qdrant.query_points(COLLECTION, query=query.tolist(), limit=k).points
            return [self.chunks[int(h.id)] for h in hits]
        scores = self._matrix @ query.astype(np.float32)
        top = np.argpartition(-scores, min(k, len(scores)) - 1)[:k] if k < len(scores) else np.arange(len(scores))
        return [self.chunks[i] for i in top[np.argsort(-scores[top])]]

    def timed_search(self, query: np.ndarray, k: int) -> tuple[list[Chunk], int]:
        start = time.perf_counter_ns()
        hits = self.search(query, k)
        return hits, time.perf_counter_ns() - start


def render_chunks(chunks: Sequence[Chunk]) -> str:
    return "\n\n".join(c.text for c in chunks)
