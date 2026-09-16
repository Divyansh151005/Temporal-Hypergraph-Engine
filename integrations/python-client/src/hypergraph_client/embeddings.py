"""Embedding helpers.

:class:`HashingEmbedder` is a deterministic, dependency-free feature-hashing
embedder. It lets the one-line integrations work out of the box and keeps
tests and offline benchmarks reproducible; pass a real embedding model for
semantic retrieval quality.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")


@runtime_checkable
class Embedder(Protocol):
    """Anything with LangChain-style ``embed_documents`` / ``embed_query``."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class HashingEmbedder:
    """Signed feature hashing of word unigrams, bigrams and character trigrams."""

    def __init__(self, dim: int = 256) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    def _features(self, text: str) -> list[str]:
        words = _TOKEN.findall(text.lower())
        features = [f"w:{w}" for w in words]
        features += [f"b:{a}_{b}" for a, b in zip(words, words[1:])]
        for word in words:
            padded = f"#{word}#"
            features += [f"c:{padded[i:i + 3]}" for i in range(len(padded) - 2)]
        return features

    def embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        for feature in self._features(text):
            digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            sign = 1.0 if value & 1 else -1.0
            weight = 2.0 if feature.startswith("w:") else 1.0
            vector[(value >> 1) % self.dim] += sign * weight
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t).tolist() for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed(text).tolist()


def to_f32_bytes(vector: Sequence[float] | np.ndarray | None) -> bytes:
    """Packs ``vector`` as little-endian float32 (empty for ``None``)."""
    if vector is None:
        return b""
    return np.asarray(vector, dtype="<f4").tobytes()


def from_f32_bytes(raw: bytes) -> np.ndarray | None:
    """Unpacks little-endian float32 bytes (``None`` when empty)."""
    if not raw:
        return None
    return np.frombuffer(raw, dtype="<f4").copy()
