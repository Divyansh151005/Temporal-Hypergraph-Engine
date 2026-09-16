"""Pytest fixtures; enable with ``pytest_plugins = ["hypergraph_client.testing"]``.

Tests using the server fixtures are skipped when no ``hypergraph-server``
build is available. :class:`FakeHypergraphClient` needs no server.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Iterator, Sequence

import numpy as np
import pytest

from .client import HypergraphClient
from .server import LocalServer, find_server_binary
from .types import BatchResult, Hyperedge, ScoredEdge, now


class FakeHypergraphClient:
    """In-memory stand-in for :class:`HypergraphClient` in unit tests.

    It mirrors the traversal semantics (validity at ``as_of``, ``k_hops``,
    event-type filter, chronological or relevance order) but none of the
    engine's conflict resolution or decay.
    """

    endpoint = "fake"

    def __init__(self) -> None:
        self.edges: dict[str, Hyperedge] = {}
        self._ids = itertools.count(1)

    def close(self) -> None:
        pass

    def insert(self, edge: Hyperedge, session_id: str = "") -> str:
        if edge.valid_to is not None and edge.valid_to <= edge.valid_from:
            raise ValueError("valid_to must be after valid_from")
        edge_id = f"fake-{next(self._ids)}"
        self.edges[edge_id] = edge
        return edge_id

    def insert_batch(self, edges: Iterable[Hyperedge], session_id: str = "") -> BatchResult:
        errors = []
        count = 0
        for index, edge in enumerate(edges):
            count += 1
            try:
                self.insert(edge, session_id)
            except ValueError as error:
                errors.append(f"item {index}: {error}")
        return BatchResult(count, tuple(errors))

    def _valid(self, edge: Hyperedge, as_of: int) -> bool:
        return edge.valid_from <= as_of and (edge.valid_to is None or as_of < edge.valid_to)

    def _scored(self, edge_id: str, embedding: Sequence[float] | None) -> ScoredEdge:
        edge = self.edges[edge_id]
        score = edge.confidence
        if embedding is not None:
            score *= _cosine(embedding, edge.embedding)
        return ScoredEdge(
            id=edge_id,
            event_type=edge.event_type,
            participants=tuple(edge.participants),
            valid_from=edge.valid_from,
            valid_to=edge.valid_to,
            is_ongoing=edge.valid_to is None,
            summary=edge.summary,
            confidence=edge.confidence,
            score=score,
            embedding=None if edge.embedding is None else np.asarray(edge.embedding, "<f4"),
        )

    def traverse(
        self,
        seeds: Sequence[str],
        *,
        k_hops: int = 1,
        as_of: int | None = None,
        embedding: Sequence[float] | None = None,
        top_k: int = 10,
        chronological: bool = False,
        event_types: Sequence[str] = (),
        decay_lambda: float = 0.0,
    ) -> list[ScoredEdge]:
        instant = now() if as_of is None else as_of
        live = {i: e for i, e in self.edges.items() if self._valid(e, instant)}
        frontier, seen_nodes, found = set(seeds), set(seeds), []
        for _ in range(max(k_hops, 1)):
            reached = [i for i, e in live.items() if i not in found and frontier & set(e.participants)]
            found += reached
            frontier = {p for i in reached for p in live[i].participants} - seen_nodes
            seen_nodes |= frontier
        if event_types:
            found = [i for i in found if live[i].event_type in event_types]
        hits = [self._scored(i, embedding) for i in found]
        if chronological:
            hits.sort(key=lambda h: (h.valid_from or 0, int(h.id.split("-")[1])), reverse=True)
        else:
            hits.sort(key=lambda h: (h.score, h.valid_from or 0), reverse=True)
        return hits[:top_k] if top_k else hits

    def query(
        self,
        embedding: Sequence[float],
        *,
        as_of: int | None = None,
        top_k: int = 10,
        decay_lambda: float = 0.0,
        session_id: str = "",
    ) -> list[ScoredEdge]:
        instant = now() if as_of is None else as_of
        hits = [
            self._scored(i, embedding)
            for i, e in self.edges.items()
            if e.embedding is not None and self._valid(e, instant)
        ]
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]

    def delete(self, hyperedge_ids: Iterable[str]) -> int:
        return sum(self.edges.pop(i, None) is not None for i in set(hyperedge_ids))

    def health(self) -> bool:
        return True


def _cosine(a: Sequence[float], b: Sequence[float] | None) -> float:
    if b is None:
        return 0.0
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(x @ y) / denominator if denominator else 0.0


@pytest.fixture(scope="session")
def hypergraph_server() -> Iterator[LocalServer]:
    """A shared server for the whole test session."""
    binary = find_server_binary()
    if binary is None:
        pytest.skip("hypergraph-server not built; run `cargo build --release -p grpc-server`")
    with LocalServer(binary) as server:
        yield server


@pytest.fixture
def hypergraph_client(hypergraph_server: LocalServer) -> Iterator[HypergraphClient]:
    with HypergraphClient(hypergraph_server.endpoint) as client:
        yield client
