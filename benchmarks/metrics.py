"""Shared metrics: token counts, mention coverage, answer grading and latency percentiles."""

from __future__ import annotations

import re
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np
import tiktoken

ENCODING = "o200k_base"
UNKNOWN = "UNKNOWN"
VERDICTS = ("correct", "stale", "wrong", "abstained")


class TokenCounter:
    def __init__(self, encoding: str = ENCODING) -> None:
        self.name = encoding
        self._encoding = tiktoken.get_encoding(encoding)

    def count(self, text: str) -> int:
        return len(self._encoding.encode(text, disallowed_special=()))

    def encode(self, text: str) -> list[int]:
        return self._encoding.encode(text, disallowed_special=())

    def decode(self, tokens: Sequence[int]) -> str:
        return self._encoding.decode(list(tokens))


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def mentions(text: str, names: Iterable[str]) -> bool:
    """Whole-phrase, case-insensitive match of any of ``names`` in ``text``."""
    haystack = f" {normalize(text)} "
    return any(f" {n} " in haystack for n in map(normalize, names) if n)


@dataclass(frozen=True)
class Entity:
    label: str
    aliases: tuple[str, ...] = ()

    @property
    def names(self) -> tuple[str, ...]:
        return (self.label, *self.aliases)


def edge_covered(text: str, participants: Sequence[Entity]) -> bool:
    """A fact is covered when every participant is named in the context."""
    return all(mentions(text, p.names) for p in participants)


@dataclass
class Question:
    """One QA item resolved against its document's entity table."""

    id: str
    dataset_id: str
    text: str
    as_of: int
    hops: int
    answer: Entity
    stale: Entity | None
    gold_edges: list[list[Entity]]
    seed_labels: list[str]


def coverage(text: str, question: Question) -> float:
    if not question.gold_edges:
        return 1.0
    return sum(edge_covered(text, e) for e in question.gold_edges) / len(question.gold_edges)


def stale_contaminated(text: str, question: Question) -> bool:
    return question.stale is not None and mentions(text, question.stale.names)


def exact_verdict(answer: str, question: Question) -> str | None:
    """Grades by normalized name match; ``None`` when a judge is needed."""
    norm = normalize(answer)
    if not norm or norm == normalize(UNKNOWN):
        return "abstained"
    if any(norm == normalize(n) for n in question.answer.names):
        return "correct"
    if question.stale is not None and any(norm == normalize(n) for n in question.stale.names):
        return "stale"
    return None


def percentiles(samples_ns: Sequence[int]) -> dict[str, float]:
    if not samples_ns:
        return {"p50_ms": float("nan"), "p95_ms": float("nan"), "p99_ms": float("nan"), "mean_ms": float("nan"), "n": 0}
    ms = np.asarray(samples_ns, dtype=np.float64) / 1e6
    p50, p95, p99 = np.percentile(ms, [50, 95, 99])
    return {"p50_ms": float(p50), "p95_ms": float(p95), "p99_ms": float(p99), "mean_ms": float(ms.mean()), "n": len(ms)}


@dataclass
class ContenderStats:
    """Per-question observations for one retrieval strategy."""

    tokens: list[int] = field(default_factory=list)
    coverage: list[float] = field(default_factory=list)
    contaminated: list[bool] = field(default_factory=list)
    verdicts: list[str] = field(default_factory=list)

    def add(self, text: str, tokens: int, question: Question) -> None:
        self.tokens.append(tokens)
        self.coverage.append(coverage(text, question))
        self.contaminated.append(stale_contaminated(text, question))

    def summary(self) -> dict[str, float | int | None]:
        n = len(self.tokens)
        result: dict[str, float | int | None] = {
            "questions": n,
            "mean_tokens": statistics.fmean(self.tokens) if n else 0.0,
            "total_tokens": sum(self.tokens),
            "gold_fact_coverage": statistics.fmean(self.coverage) if n else 0.0,
            "full_coverage_rate": sum(c >= 1.0 for c in self.coverage) / n if n else 0.0,
            "stale_contamination": sum(self.contaminated) / n if n else 0.0,
        }
        if self.verdicts:
            counts = {v: self.verdicts.count(v) for v in VERDICTS}
            total = len(self.verdicts)
            result.update({f"{v}_rate": counts[v] / total for v in VERDICTS})
            result["hallucination_rate"] = (counts["stale"] + counts["wrong"]) / total
        else:
            result.update({f"{v}_rate": None for v in VERDICTS})
            result["hallucination_rate"] = None
        return result


def reduction(baseline: float | None, candidate: float | None) -> float | None:
    """Relative reduction ``(baseline - candidate) / baseline``; ``None`` when undefined."""
    if baseline is None or candidate is None or baseline == 0:
        return None
    return (baseline - candidate) / baseline
