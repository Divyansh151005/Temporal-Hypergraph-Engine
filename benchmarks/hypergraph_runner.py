"""Hypergraph contender: ingest graphs, link question entities, k-hop traverse at ``as_of``, render compact facts."""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from hypergraph_client import Hyperedge, HypergraphClient, to_f32_bytes
from hypergraph_client.server import LocalServer
from hypergraph_engine_native import NativeHypergraphEngine

from metrics import Entity, normalize

Embed = Callable[[str], np.ndarray]


@dataclass(frozen=True)
class GraphEdge:
    """A hyperedge keyed by participant labels, ready for ingestion."""

    event_type: str
    participants: tuple[str, ...]
    valid_from: int | None
    valid_to: int | None
    summary: str


@dataclass(frozen=True)
class Fact:
    """A retrieved hyperedge."""

    event_type: str
    participants: tuple[str, ...]
    valid_from: int | None
    valid_to: int | None
    summary: str
    score: float = 0.0


DAY_SECONDS = 86_400


def validity(valid_from: int | None, valid_to: int | None) -> tuple[int | None, int | None]:
    """Widens point events (``valid_to <= valid_from``) to one day; the engine stores half-open intervals."""
    if valid_from is not None and valid_to is not None and valid_to <= valid_from:
        return valid_from, valid_from + DAY_SECONDS
    return valid_from, valid_to


def graph_edges(graph: dict) -> list[GraphEdge]:
    """Converts an id-keyed ``{nodes, hyperedges}`` graph to label-keyed edges."""
    labels = {n["id"]: n["label"] for n in graph["nodes"]}
    edges = []
    for e in graph["hyperedges"]:
        valid_from, valid_to = validity(e.get("valid_from"), e.get("valid_to"))
        edges.append(GraphEdge(
            event_type=e["event_type"],
            participants=tuple(labels.get(p, p) for p in e["participants"]),
            valid_from=valid_from,
            valid_to=valid_to,
            summary=e.get("summary") or "",
        ))
    return edges


def iso_day(epoch: int | None, open_text: str) -> str:
    if not epoch:
        return open_text
    return time.strftime("%Y-%m-%d", time.gmtime(epoch))


def render_fact(fact: Fact) -> str:
    span = f"[{iso_day(fact.valid_from, '?')} → {iso_day(fact.valid_to, 'present')}]"
    return f"{span} {fact.event_type}({', '.join(fact.participants)}): {fact.summary}"


def render_facts(facts: Sequence[Fact]) -> str:
    return "\n".join(render_fact(f) for f in facts)


class EntityLinker:
    """Longest-first, whole-phrase linking of labels and aliases mentioned in a question."""

    def __init__(self, entities: Iterable[Entity]) -> None:
        index: dict[str, set[str]] = defaultdict(set)
        for entity in entities:
            for name in entity.names:
                if key := normalize(name):
                    index[key].add(entity.label)
        self._index = dict(index)
        self._max_words = max((k.count(" ") + 1 for k in self._index), default=0)

    def link(self, text: str) -> list[str]:
        words = normalize(text).split()
        found: list[str] = []
        i = 0
        while i < len(words):
            for width in range(min(self._max_words, len(words) - i), 0, -1):
                labels = self._index.get(" ".join(words[i : i + width]))
                if labels:
                    found += sorted(labels - set(found))
                    i += width
                    break
            else:
                i += 1
        return found


class NativeRunner:
    """In-process engine through the PyO3 bindings."""

    name = "pyo3"

    def __init__(self, embed: Embed, dim: int) -> None:
        self.embed = embed
        self.engine = NativeHypergraphEngine.in_memory(dim)

    def ingest(self, edges: Iterable[GraphEdge]) -> int:
        count = 0
        for edge in edges:
            self.engine.insert_hyperedge_fast(
                edge.event_type,
                list(edge.participants),
                edge.valid_from or 0,
                edge.valid_to,
                to_f32_bytes(self.embed(edge.summary)),
                1.0,
                edge.summary,
            )
            count += 1
        return count

    def retrieve(self, seeds: Sequence[str], as_of: int, query: np.ndarray, hops: int, top_k: int) -> tuple[list[Fact], int]:
        query_bytes = to_f32_bytes(query)
        start = time.perf_counter_ns()
        if seeds:
            rows = self.engine.traverse(list(seeds), hops, as_of, query_bytes, 0.0, top_k)
        else:
            rows = self.engine.query_temporal(query_bytes, as_of, 0.0, top_k)
        elapsed = time.perf_counter_ns() - start
        facts = [
            Fact(r["event_type"] or "", tuple(r["participants"]), r["valid_from"] or None, r["valid_to"],
                 r["summary"] or "", r["score"])
            for r in rows
        ]
        return facts, elapsed

    def close(self) -> None:
        pass


class GrpcRunner:
    """The same engine behind ``hypergraph-server`` on a Unix socket."""

    name = "grpc_uds"

    def __init__(self, embed: Embed, dim: int) -> None:
        self.embed = embed
        self.server = LocalServer(embedding_dim=dim).start()
        self.client = HypergraphClient(self.server.endpoint)

    def ingest(self, edges: Iterable[GraphEdge]) -> int:
        batch = [
            Hyperedge(
                event_type=e.event_type,
                participants=e.participants,
                valid_from=e.valid_from or 0,
                valid_to=e.valid_to,
                summary=e.summary,
                embedding=self.embed(e.summary),
            )
            for e in edges
        ]
        result = self.client.insert_batch(batch)
        if result.errors:
            raise RuntimeError(f"gRPC ingest errors: {result.errors[:3]}")
        return result.processed

    def retrieve(self, seeds: Sequence[str], as_of: int, query: np.ndarray, hops: int, top_k: int) -> tuple[list[Fact], int]:
        start = time.perf_counter_ns()
        if seeds:
            rows = self.client.traverse(seeds, k_hops=hops, as_of=as_of, embedding=query, top_k=top_k)
        else:
            rows = self.client.query(query, as_of=as_of, top_k=top_k)
        elapsed = time.perf_counter_ns() - start
        facts = [Fact(r.event_type, r.participants, r.valid_from or None, r.valid_to, r.summary, r.score) for r in rows]
        return facts, elapsed

    def close(self) -> None:
        self.client.close()
        self.server.stop()
