#!/usr/bin/env python3
"""Offline competitor benchmark: temporal hypergraph traversal vs property-graph BFS vs flat vector search.

Builds a dense, seeded temporal world (entities, N-ary facts with validity windows and
superseded versions, articles chunked for vector RAG) and 1,000+ multi-hop questions with
exact gold fact sets. Every contender is purely algorithmic (no LLM calls):

- ``hypergraph``: the engine through PyO3 (``traverse`` with a cosine re-rank).
- ``networkx_naive``: Neo4j-style property graph (facts reified as event nodes), k-hop BFS, no time filter.
- ``networkx_temporal``: the same BFS skipping event nodes not valid at ``as_of``.
- ``vector_faiss`` / ``vector_qdrant``: top-k cosine over ~128-token article chunks.

    python -m hypergraph_evaluation.competitor_bench --entities 12000 --facts 60000 --queries 1000
    python -m hypergraph_evaluation.competitor_bench --scale-sweep 10000,50000,100000 --embedder hashing
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import platform
import random
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import tiktoken
from hypergraph_client import HashingEmbedder, to_f32_bytes
from hypergraph_engine_native import NativeHypergraphEngine
from hypergraph_engine_native import __version__ as ENGINE_VERSION

try:
    import faiss
except ImportError:  # optional; --vector-backend qdrant works without it
    faiss = None

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams
except ImportError:  # optional; --vector-backend faiss works without it
    QdrantClient = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # optional; --embedder hashing works without it
    SentenceTransformer = None

log = logging.getLogger("competitor_bench")

REPO_ROOT = Path(__file__).resolve().parents[2]


def repo_path(path: str) -> Path:
    """Relative paths resolve against the repository root so results land in the gitignored ``results/``."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


ENCODING = "o200k_base"
START = int(datetime(2000, 1, 1, tzinfo=timezone.utc).timestamp())
END = int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp())
MONTH = 30 * 86_400
YEAR = 365 * 86_400

# --------------------------------------------------------------------------- world


@dataclass(frozen=True)
class Fact:
    """One N-ary temporal fact; ``valid_to`` is exclusive, ``None`` while ongoing."""

    id: int
    event_type: str
    participants: tuple[int, ...]
    valid_from: int
    valid_to: int | None
    text: str

    def valid_at(self, t: int) -> bool:
        return self.valid_from <= t and (self.valid_to is None or t < self.valid_to)


@dataclass(frozen=True)
class Chunk:
    text: str
    tokens: int
    fact_ids: tuple[int, ...]


@dataclass(frozen=True)
class Query:
    id: str
    template: str
    text: str
    seeds: tuple[int, ...]
    as_of: int
    gold: tuple[int, ...]
    answer: int
    event_types: tuple[str, ...]

    @property
    def hops(self) -> int:
        return len(self.gold)


@dataclass(frozen=True)
class Plan:
    """What a text-to-query planner hands a graph store: hop depth and relation types."""

    hops: int
    event_types: tuple[str, ...]


@dataclass
class World:
    names: list[str]
    kinds: list[str]
    facts: list[Fact]
    chunks: list[Chunk] = field(default_factory=list)
    articles: int = 0

    def label(self, entity: int) -> str:
        return self.names[entity]


FIRST = (
    "Ava Liam Noah Emma Mia Lucas Zoe Ethan Aria Leo Maya Omar Ines Kai Nora Ivan Lena Theo Rhea Hugo Sana Elio "
    "Yara Nils Tara Aiden Clara Felix Mila Ravi Anya Jonas Lara Mateo Selin Dario Freya Kenji Alma Tobias Nadia "
    "Rafael Isla Malik Esme Viktor Leila Arjun Greta Soren Imani Pavel Celia Hamza Wren Bruno Talia Emil Dalia"
).split()
SYLLABLES = (
    "ka lo ren vi sto mar del ion tra bel cor nis vel dra mon ten sar quo fin lux zar pel gor han wex tor bri "
    "cal dun ely fra gil hol ist jor kel lim nor ost pra qui ros sel tam ul ven wil xan yor zel"
).split()
COMPANY_SUFFIX = "Dynamics Systems Labs Holdings Industries Analytics Robotics Energy Logistics Biotech Networks Foods".split()
PRODUCT_LINE = "Orion Vega Atlas Nimbus Helix Quanta Pulse Vertex Nova Strata Echo Titan".split()
CITY_PREFIX = "Port New San North East West Lake Fort Mount Saint".split()


def _word(rng: random.Random, parts: int) -> str:
    return "".join(rng.choice(SYLLABLES) for _ in range(parts)).capitalize()


def _unique(make: Callable[[], str], used: set[str]) -> str:
    while True:
        name = make()
        if name not in used:
            used.add(name)
            return name


def month(t: int | None) -> str:
    return "now" if t is None else datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m")


def phrase(t: int) -> str:
    return datetime.fromtimestamp(t, tz=timezone.utc).strftime("%B %Y")


def _windows(rng: random.Random, start: int, count: int, *, open_end: bool = True) -> list[tuple[int, int | None]]:
    """``count`` consecutive half-open windows from ``start``; the last stays open when ``open_end``."""
    cuts = sorted(rng.randint(start + YEAR, END - YEAR) for _ in range(count - 1)) if count > 1 else []
    bounds = [start, *cuts]
    out: list[tuple[int, int | None]] = []
    for i, lo in enumerate(bounds):
        hi = bounds[i + 1] if i + 1 < len(bounds) else (None if open_end else rng.randint(lo + MONTH, END))
        if hi is not None and hi <= lo:
            continue
        out.append((lo, hi))
    return out


class WorldBuilder:
    """Seeded generator for a dense temporal world with power-law hubs."""

    MIX = {"person": 0.48, "company": 0.24, "product": 0.13, "city": 0.06, "drug": 0.09}

    def __init__(self, entities: int, facts: int, seed: int) -> None:
        self.rng = random.Random(seed)
        self.target_facts = facts
        self.names: list[str] = []
        self.kinds: list[str] = []
        self.by_kind: dict[str, list[int]] = defaultdict(list)
        self.facts: list[Fact] = []
        self.keys: set[tuple[str, frozenset[int]]] = set()
        used: set[str] = set()
        rng = self.rng
        makers = {
            "person": lambda: f"{rng.choice(FIRST)} {_word(rng, 2)}{_word(rng, 1).lower()}",
            "company": lambda: f"{_word(rng, 2)} {rng.choice(COMPANY_SUFFIX)}",
            "product": lambda: f"{rng.choice(PRODUCT_LINE)} {_word(rng, 1)}{rng.randint(10, 999)}",
            "city": lambda: f"{rng.choice(CITY_PREFIX)} {_word(rng, 2)}",
            "drug": lambda: f"{_word(rng, 3)}mab",
        }
        for kind, share in self.MIX.items():
            for _ in range(max(2, int(entities * share))):
                self.by_kind[kind].append(len(self.names))
                self.names.append(_unique(makers[kind], used))
                self.kinds.append(kind)
        # Zipf-like popularity so a few companies and cities become hubs.
        self.weights = {
            kind: [1.0 / (rank + 1) ** 0.9 for rank in range(len(ids))] for kind, ids in self.by_kind.items()
        }

    def pick(self, kind: str, *, skewed: bool = True) -> int:
        ids = self.by_kind[kind]
        if skewed:
            return self.rng.choices(ids, weights=self.weights[kind])[0]
        return self.rng.choice(ids)

    def add(self, event_type: str, participants: Sequence[int], window: tuple[int, int | None], text: str) -> bool:
        key = (event_type, frozenset(participants))
        if key in self.keys or len(set(participants)) != len(participants):
            return False
        self.keys.add(key)
        lo, hi = window
        self.facts.append(Fact(len(self.facts), event_type, tuple(participants), lo, hi, text))
        return True

    def n(self, entity: int) -> str:
        return self.names[entity]

    def build(self) -> World:
        rng = self.rng
        companies, people = self.by_kind["company"], self.by_kind["person"]
        founded = {c: rng.randint(START, START + 12 * YEAR) for c in companies}
        free_people = people.copy()
        rng.shuffle(free_people)

        for c in companies:
            for lo, hi in _windows(rng, founded[c], rng.randint(1, 4)):
                ceo = free_people.pop() if free_people else self.pick("person", skewed=False)
                end = f" and stepped down in {phrase(hi)}" if hi else ""
                self.add("CEO_OF", (ceo, c), (lo, hi), f"{self.n(ceo)} became CEO of {self.n(c)} in {phrase(lo)}{end}.")
            for lo, hi in _windows(rng, founded[c], rng.choice((1, 1, 2, 3))):
                city = self.pick("city")
                text = (
                    f"{self.n(c)} was headquartered in {self.n(city)} from {phrase(lo)} until {phrase(hi)}."
                    if hi
                    else f"{self.n(c)} has been headquartered in {self.n(city)} since {phrase(lo)}."
                )
                self.add("HEADQUARTERED_IN", (c, city), (lo, hi), text)

        for p in self.by_kind["product"]:
            launch = rng.randint(START + 2 * YEAR, END - 4 * YEAR)
            for lo, hi in _windows(rng, launch, rng.choice((1, 1, 2))):
                maker = self.pick("company")
                text = (
                    f"{self.n(maker)} manufactured {self.n(p)} from {phrase(lo)} until {phrase(hi)}, when production moved."
                    if hi
                    else f"{self.n(maker)} manufactures {self.n(p)}, in production since {phrase(lo)}."
                )
                self.add("MAKES", (maker, p), (lo, hi), text)

        targets = rng.sample(companies, k=len(companies) * 3 // 10)
        for target in targets:
            acquirer = self.pick("company")
            when = rng.randint(founded[target] + 2 * YEAR, END - YEAR)
            self.add(
                "ACQUIRED", (acquirer, target), (when, None),
                f"{self.n(acquirer)} completed its acquisition of {self.n(target)} in {phrase(when)}.",
            )

        for person in people:
            career = rng.randint(START, START + 15 * YEAR)
            for lo, hi in _windows(rng, career, rng.randint(1, 3)):
                employer, office = self.pick("company"), self.pick("city")
                end = f" and left in {phrase(hi)}" if hi else ""
                self.add(
                    "WORKS_AT", (person, employer, office), (lo, hi),
                    f"{self.n(person)} joined {self.n(employer)} at its {self.n(office)} office in {phrase(lo)}{end}.",
                )
            for lo, hi in _windows(rng, career, rng.randint(1, 2)):
                city = self.pick("city")
                end = f" and moved away in {phrase(hi)}" if hi else ""
                self.add("LIVES_IN", (person, city), (lo, hi), f"{self.n(person)} settled in {self.n(city)} in {phrase(lo)}{end}.")

        fillers = (self._supplies, self._invested, self._prescribed)
        attempts = 0
        while len(self.facts) < self.target_facts and attempts < self.target_facts * 20:
            rng.choice(fillers)()
            attempts += 1

        world = World(self.names, self.kinds, self.facts)
        return world

    def _window(self) -> tuple[int, int | None]:
        lo = self.rng.randint(START + YEAR, END - YEAR)
        hi = None if self.rng.random() < 0.6 else self.rng.randint(lo + MONTH, END)
        return lo, hi

    def _supplies(self) -> None:
        a, b, p = self.pick("company"), self.pick("company"), self.pick("product", skewed=False)
        lo, hi = self._window()
        end = f" until {phrase(hi)}" if hi else ""
        self.add("SUPPLIES", (a, b, p), (lo, hi), f"{self.n(a)} supplied {self.n(p)} components to {self.n(b)} from {phrase(lo)}{end}.")

    def _invested(self) -> None:
        person, company = self.pick("person", skewed=False), self.pick("company")
        lo, hi = self._window()
        end = f" and exited in {phrase(hi)}" if hi else ""
        self.add("INVESTED_IN", (person, company), (lo, hi), f"{self.n(person)} invested in {self.n(company)} in {phrase(lo)}{end}.")

    def _prescribed(self) -> None:
        doctor, patient = self.pick("person", skewed=False), self.pick("person", skewed=False)
        drug = self.pick("drug")
        lo, hi = self._window()
        end = f"; treatment ended in {phrase(hi)}" if hi else ""
        self.add(
            "PRESCRIBED", (doctor, patient, drug), (lo, hi),
            f"Dr. {self.n(doctor)} prescribed {self.n(drug)} to {self.n(patient)} in {phrase(lo)}{end}.",
        )


PRIMARY_ROLE = {"CEO_OF": 1, "HEADQUARTERED_IN": 0, "MAKES": 0, "ACQUIRED": 1, "WORKS_AT": 1, "LIVES_IN": 0,
                "SUPPLIES": 0, "INVESTED_IN": 1, "PRESCRIBED": 1}


def build_articles(world: World, counter: TokenCounter, rng: random.Random, max_tokens: int = 128) -> None:
    """Groups facts into per-entity profile articles and packs whole sentences into chunks."""
    by_anchor: dict[int, list[Fact]] = defaultdict(list)
    for fact in world.facts:
        by_anchor[fact.participants[PRIMARY_ROLE.get(fact.event_type, 0)]].append(fact)
    chunks: list[Chunk] = []
    articles = 0
    for anchor, facts in by_anchor.items():
        facts.sort(key=lambda f: f.valid_from)
        for start in range(0, len(facts), 12):
            section = facts[start : start + 12]
            if len(section) > 3:
                rng.shuffle(section)
            articles += 1
            sentences = [(f"Profile: {world.label(anchor)}.", None)] + [(f.text, f.id) for f in section]
            text_parts: list[str] = []
            ids: list[int] = []
            tokens = 0
            for sentence, fid in sentences:
                n = counter.count(sentence) + 1
                if text_parts and tokens + n > max_tokens:
                    chunks.append(Chunk(" ".join(text_parts), tokens, tuple(ids)))
                    text_parts, ids, tokens = [], [], 0
                text_parts.append(sentence)
                tokens += n
                if fid is not None:
                    ids.append(fid)
            if text_parts:
                chunks.append(Chunk(" ".join(text_parts), tokens, tuple(ids)))
    world.chunks = chunks
    world.articles = articles


# --------------------------------------------------------------------------- queries


class FactIndex:
    def __init__(self, world: World) -> None:
        self.by_role: dict[tuple[str, int, int], list[Fact]] = defaultdict(list)
        for fact in world.facts:
            for role, entity in enumerate(fact.participants):
                self.by_role[(fact.event_type, role, entity)].append(fact)

    def unique(self, event_type: str, role: int, entity: int, t: int) -> Fact | None:
        """The single fact of ``event_type`` with ``entity`` in ``role`` valid at ``t``, if exactly one."""
        hits = [f for f in self.by_role.get((event_type, role, entity), ()) if f.valid_at(t)]
        return hits[0] if len(hits) == 1 else None

    def superseded(self, event_type: str, role: int, entity: int, t: int) -> bool:
        return any(not f.valid_at(t) for f in self.by_role.get((event_type, role, entity), ()))


# (template, seed kind, steps, question). Each step is (event_type, role of the known entity, role of the next one).
TEMPLATES: dict[str, tuple[str, tuple[tuple[str, int, int], ...], str]] = {
    "ceo_of_maker": ("product", (("MAKES", 1, 0), ("CEO_OF", 1, 0)),
                     "As of {date}, who is the CEO of the company that makes {seed}?"),
    "hq_of_acquirer": ("company", (("ACQUIRED", 1, 0), ("HEADQUARTERED_IN", 0, 1)),
                       "As of {date}, in which city is the company that acquired {seed} headquartered?"),
    "ceo_of_employer": ("person", (("WORKS_AT", 0, 1), ("CEO_OF", 1, 0)),
                        "As of {date}, who is the CEO of the company {seed} works for?"),
    "ceo_of_employer_acquirer": ("person", (("WORKS_AT", 0, 1), ("ACQUIRED", 1, 0), ("CEO_OF", 1, 0)),
                                 "As of {date}, who is the CEO of the company that acquired {seed}'s employer?"),
    "hq_of_maker_acquirer": ("product", (("MAKES", 1, 0), ("ACQUIRED", 1, 0), ("HEADQUARTERED_IN", 0, 1)),
                             "As of {date}, where is the company that acquired the maker of {seed} headquartered?"),
    "home_of_ceo_of_maker": ("product", (("MAKES", 1, 0), ("CEO_OF", 1, 0), ("LIVES_IN", 0, 1)),
                             "As of {date}, in which city does the CEO of the company that makes {seed} live?"),
}


def generate_queries(world: World, n: int, max_hops: int, seed: int) -> list[Query]:
    """Multi-hop questions whose every step has exactly one fact valid at ``as_of``; prefers chains with stale versions."""
    rng = random.Random(seed)
    index = FactIndex(world)
    by_kind: dict[str, list[int]] = defaultdict(list)
    for entity, kind in enumerate(world.kinds):
        by_kind[kind].append(entity)
    templates = [name for name, (_, steps, _) in TEMPLATES.items() if len(steps) <= max_hops]
    queries: list[Query] = []
    attempts = 0
    while len(queries) < n and attempts < n * 400:
        attempts += 1
        name = rng.choice(templates)
        kind, steps, question = TEMPLATES[name]
        seed_entity = rng.choice(by_kind[kind])
        t = rng.randint(START + 3 * YEAR, END - MONTH)
        current, chain, stale = seed_entity, [], False
        for event_type, known_role, next_role in steps:
            fact = index.unique(event_type, known_role, current, t)
            if fact is None:
                break
            stale = stale or index.superseded(event_type, known_role, current, t)
            chain.append(fact)
            current = fact.participants[next_role]
        if len(chain) != len(steps):
            continue
        if not stale and rng.random() < 0.7:
            continue  # bias toward questions where an outdated fact is a live distractor
        queries.append(Query(
            id=f"q{len(queries):05d}", template=name,
            text=question.format(date=phrase(t), seed=world.label(seed_entity)),
            seeds=(seed_entity,), as_of=t, gold=tuple(f.id for f in chain), answer=current,
            event_types=tuple(dict.fromkeys(step[0] for step in steps)),
        ))
    if len(queries) < n:
        log.warning("only %d of %d queries could be generated", len(queries), n)
    return queries


# --------------------------------------------------------------------------- embedders


class TokenCounter:
    def __init__(self) -> None:
        self._encoding = tiktoken.get_encoding(ENCODING)

    def count(self, text: str) -> int:
        return len(self._encoding.encode(text, disallowed_special=()))


class Embedder:
    name: str
    dim: int

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError


class MiniLMEmbedder(Embedder):
    def __init__(self, model: str) -> None:
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers is not installed; pip install 'sentence-transformers>=3.0' or use --embedder hashing")
        self._model = SentenceTransformer(model)
        self.name = model
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._model.encode(
            list(texts), batch_size=256, normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=len(texts) > 5_000,
        )
        return np.asarray(vectors, dtype=np.float32)


class HashEmbedder(Embedder):
    def __init__(self, dim: int) -> None:
        self._impl = HashingEmbedder(dim)
        self.name = f"hashing-{dim}"
        self.dim = dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._impl.embed(t) for t in texts]).astype(np.float32)


def cached_encode(embedder: Embedder, texts: Sequence[str], cache_dir: Path | None, tag: str) -> np.ndarray:
    if cache_dir is None:
        return embedder.encode(texts)
    digest = hashlib.sha256()
    digest.update(embedder.name.encode())
    for text in texts:
        digest.update(text.encode())
        digest.update(b"\0")
    path = cache_dir / f"{tag}-{digest.hexdigest()[:20]}.npy"
    if path.exists():
        return np.load(path)
    started = time.perf_counter()
    vectors = embedder.encode(texts)
    log.info("embedded %d %s with %s in %.1fs", len(texts), tag, embedder.name, time.perf_counter() - started)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(path, vectors)
    return vectors


# --------------------------------------------------------------------------- contenders


@dataclass
class Retrieval:
    facts: list[int]
    context: str
    ns: int


def plan_for(query: Query, hops: int, planned: bool) -> Plan:
    if planned:
        return Plan(min(hops, query.hops), query.event_types)
    return Plan(hops, ())


def render_fact(world: World, fact: Fact) -> str:
    who = ", ".join(world.label(p) for p in fact.participants)
    return f"[{month(fact.valid_from)} -> {month(fact.valid_to)}] {fact.event_type}({who}): {fact.text}"


class HypergraphContender:
    name = "hypergraph"
    label = "Temporal Hypergraph (PyO3)"

    def __init__(self, world: World, fact_vectors: np.ndarray, hops: int, top_k: int, decay_lambda: float,
                 planned: bool) -> None:
        self.world, self.hops, self.top_k, self.decay_lambda, self.planned = world, hops, top_k, decay_lambda, planned
        self.engine = NativeHypergraphEngine.in_memory(fact_vectors.shape[1])
        self.ids: dict[str, int] = {}
        started = time.perf_counter()
        for fact, vector in zip(world.facts, fact_vectors):
            hid = self.engine.insert_hyperedge_fast(
                fact.event_type, [world.label(p) for p in fact.participants], fact.valid_from, fact.valid_to,
                to_f32_bytes(vector), 1.0, None,
            )
            self.ids[hid] = fact.id
        self.ingest_secs = time.perf_counter() - started
        stored = self.engine.stats()["hyperedges"]
        if stored != len(world.facts):
            raise RuntimeError(f"engine stored {stored} hyperedges for {len(world.facts)} facts")

    def retrieve(self, query: Query, qvec: np.ndarray) -> Retrieval:
        seeds = [self.world.label(s) for s in query.seeds]
        qbytes = to_f32_bytes(qvec)
        plan = plan_for(query, self.hops, self.planned)
        event_types = list(plan.event_types) or None
        start = time.perf_counter_ns()
        rows = self.engine.traverse(seeds, plan.hops, query.as_of, qbytes, self.decay_lambda, self.top_k,
                                    False, event_types)
        ns = time.perf_counter_ns() - start
        facts = [self.ids[r["id"]] for r in rows]
        return Retrieval(facts, "\n".join(render_fact(self.world, self.world.facts[f]) for f in facts), ns)


class NetworkXContender:
    """Neo4j-style property graph: entities and reified fact (event) nodes with validity properties."""

    def __init__(self, world: World, fact_vectors: np.ndarray, hops: int, top_k: int, temporal: bool,
                 planned: bool) -> None:
        self.world, self.vectors, self.hops, self.top_k, self.temporal = world, fact_vectors, hops, top_k, temporal
        self.planned = planned
        self.name = "networkx_temporal" if temporal else "networkx_naive"
        self.label = "NetworkX property graph (temporal filter)" if temporal else "NetworkX property graph (naive BFS)"
        self.offset = len(world.names)
        started = time.perf_counter()
        graph = nx.Graph()
        graph.add_nodes_from((e, {"kind": k, "name": n}) for e, (k, n) in enumerate(zip(world.kinds, world.names)))
        for fact in world.facts:
            node = self.offset + fact.id
            graph.add_node(node, kind="event", event_type=fact.event_type,
                           valid_from=fact.valid_from, valid_to=fact.valid_to)
            graph.add_edges_from((p, node, {"role": r}) for r, p in enumerate(fact.participants))
        self.graph = graph
        self.ingest_secs = time.perf_counter() - started

    def _reachable(self, seeds: Sequence[int], as_of: int, plan: Plan) -> list[int]:
        """Engine-equivalent BFS: results are filtered by event type, expansion is not."""
        graph, adj, nodes = self.graph, self.graph.adj, self.graph.nodes
        wanted = set(plan.event_types)
        seen_nodes = set(seeds)
        seen_events: set[int] = set()
        frontier = [s for s in seeds if s in graph]
        found: list[int] = []
        for hop in range(plan.hops):
            expand = hop + 1 < plan.hops
            nxt: list[int] = []
            for entity in frontier:
                for event in adj[entity]:
                    if event in seen_events:
                        continue
                    seen_events.add(event)
                    props = nodes[event]
                    if self.temporal:
                        end = props["valid_to"]
                        if not (props["valid_from"] <= as_of and (end is None or as_of < end)):
                            continue
                    if not wanted or props["event_type"] in wanted:
                        found.append(event - self.offset)
                    if expand:
                        for participant in adj[event]:
                            if participant not in seen_nodes:
                                seen_nodes.add(participant)
                                nxt.append(participant)
            if not nxt:
                break
            frontier = nxt
        return found

    def retrieve(self, query: Query, qvec: np.ndarray) -> Retrieval:
        plan = plan_for(query, self.hops, self.planned)
        start = time.perf_counter_ns()
        candidates = np.asarray(self._reachable(query.seeds, query.as_of, plan), dtype=np.int64)
        if len(candidates):
            scores = self.vectors[candidates] @ qvec
            k = min(self.top_k, len(candidates))
            top = np.argpartition(-scores, k - 1)[:k] if k < len(candidates) else np.arange(len(candidates))
            ranked = candidates[top[np.argsort(-scores[top], kind="stable")]].tolist()
        else:
            ranked = []
        ns = time.perf_counter_ns() - start
        return Retrieval(ranked, "\n".join(render_fact(self.world, self.world.facts[f]) for f in ranked), ns)


class VectorContender:
    def __init__(self, world: World, chunk_vectors: np.ndarray, backend: str, k: int) -> None:
        self.world, self.backend, self.k = world, backend, k
        self.name = f"vector_{backend}"
        started = time.perf_counter()
        vectors = chunk_vectors / np.maximum(np.linalg.norm(chunk_vectors, axis=1, keepdims=True), 1e-12)
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if backend == "faiss":
            if faiss is None:
                raise RuntimeError("faiss is not installed; pip install faiss-cpu or use --vector-backend qdrant")
            self.index = faiss.IndexFlatIP(vectors.shape[1])
            self.index.add(vectors)
        elif backend == "qdrant":
            if QdrantClient is None:
                raise RuntimeError("qdrant-client is not installed")
            self.client = QdrantClient(":memory:")
            self.client.create_collection("chunks", vectors_config=VectorParams(size=vectors.shape[1], distance=Distance.COSINE))
            for lo in range(0, len(vectors), 2_048):
                self.client.upsert("chunks", points=[
                    PointStruct(id=i, vector=vectors[i].tolist(), payload={}) for i in range(lo, min(lo + 2_048, len(vectors)))
                ], wait=True)
        else:
            raise ValueError(f"unknown vector backend {backend!r}")
        self.ingest_secs = time.perf_counter() - started

    def label_for(self, k: int, matched: bool) -> str:
        engine = "FAISS exact" if self.backend == "faiss" else "Qdrant in-memory"
        return f"Flat vector ({engine}, {'token-matched ' if matched else ''}k={k})"

    def search(self, qvec: np.ndarray, k: int) -> tuple[list[int], int]:
        q = np.ascontiguousarray(qvec[None, :], dtype=np.float32)
        start = time.perf_counter_ns()
        if self.backend == "faiss":
            _, idx = self.index.search(q, k)
            hits = [int(i) for i in idx[0] if i >= 0]
        else:
            hits = [int(p.id) for p in self.client.query_points("chunks", query=q[0].tolist(), limit=k).points]
        return hits, time.perf_counter_ns() - start

    def retrieve_k(self, qvec: np.ndarray, k: int) -> Retrieval:
        hits, ns = self.search(qvec, k)
        chunks = [self.world.chunks[i] for i in hits]
        facts = list(dict.fromkeys(f for c in chunks for f in c.fact_ids))
        return Retrieval(facts, "\n\n".join(c.text for c in chunks), ns)


# --------------------------------------------------------------------------- metrics


@dataclass
class Row:
    query: str
    contender: str
    latency_ms: float
    tokens: int
    retrieved: int
    precision: float
    recall: float
    f1: float
    stale_rate: float
    full_chain: bool
    final_hop: bool


def score(world: World, query: Query, contender: str, r: Retrieval, counter: TokenCounter) -> Row:
    retrieved = set(r.facts)
    gold = set(query.gold)
    hit = len(retrieved & gold)
    precision = hit / len(retrieved) if retrieved else 0.0
    recall = hit / len(gold) if gold else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    stale = sum(not world.facts[f].valid_at(query.as_of) for f in retrieved)
    return Row(
        query=query.id, contender=contender, latency_ms=r.ns / 1e6, tokens=counter.count(r.context),
        retrieved=len(retrieved), precision=precision, recall=recall, f1=f1,
        stale_rate=stale / len(retrieved) if retrieved else 0.0,
        full_chain=gold <= retrieved, final_hop=query.gold[-1] in retrieved,
    )


def pct(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else 0.0


def summarize(name: str, label: str, rows: Sequence[Row], **extra: Any) -> dict[str, Any]:
    lat = [r.latency_ms for r in rows]
    tok = [r.tokens for r in rows]
    mean = lambda xs: float(statistics.fmean(xs)) if xs else 0.0  # noqa: E731
    return {
        "name": name, "label": label, "queries": len(rows),
        "latency_ms": {"mean": mean(lat), "p50": pct(lat, 50), "p95": pct(lat, 95), "p99": pct(lat, 99)},
        "tokens": {"mean": mean(tok), "p50": pct(tok, 50), "p95": pct(tok, 95)},
        "precision": mean([r.precision for r in rows]),
        "recall": mean([r.recall for r in rows]),
        "f1": mean([r.f1 for r in rows]),
        "stale_rate": mean([r.stale_rate for r in rows]),
        "full_chain_recall": mean([float(r.full_chain) for r in rows]),
        "final_hop_recall": mean([float(r.final_hop) for r in rows]),
        "mean_retrieved_facts": mean([r.retrieved for r in rows]),
        **extra,
    }


# --------------------------------------------------------------------------- runner


def environment() -> dict[str, Any]:
    def run(*cmd: str) -> str:
        try:
            return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return ""

    sha = run("git", "rev-parse", "--short", "HEAD")
    dirty = bool(run("git", "status", "--porcelain"))
    return {
        "cpu": run("sysctl", "-n", "machdep.cpu.brand_string") or platform.processor(),
        "os": f"{platform.system()} {platform.machine()}",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "networkx": nx.__version__,
        "faiss": getattr(faiss, "__version__", None) if faiss else None,
        "engine": ENGINE_VERSION,
        "git_sha": f"{sha}-dirty" if dirty else sha,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def make_embedder(args: argparse.Namespace, kind: str | None = None) -> Embedder:
    kind = kind or args.embedder
    if kind == "minilm":
        return MiniLMEmbedder(args.model)
    return HashEmbedder(args.dim)


def timed_pass(queries: Sequence[Query], qvecs: np.ndarray, call: Callable[[Query, np.ndarray], Retrieval],
               warmup: int) -> list[Retrieval]:
    for q, v in zip(queries[:warmup], qvecs[:warmup]):
        call(q, v)
    return [call(q, v) for q, v in zip(queries, qvecs)]


def dataset_stats(world: World, queries: Sequence[Query]) -> dict[str, Any]:
    degree = Counter(p for f in world.facts for p in f.participants)
    degrees = list(degree.values())
    return {
        "entities": len(world.names),
        "entity_kinds": dict(Counter(world.kinds)),
        "facts": len(world.facts),
        "event_types": dict(Counter(f.event_type for f in world.facts)),
        "nary_facts": sum(len(f.participants) > 2 for f in world.facts),
        "closed_facts": sum(f.valid_to is not None for f in world.facts),
        "mean_entity_degree": float(statistics.fmean(degrees)) if degrees else 0.0,
        "max_entity_degree": max(degrees, default=0),
        "articles": world.articles,
        "chunks": len(world.chunks),
        "mean_chunk_tokens": float(statistics.fmean(c.tokens for c in world.chunks)) if world.chunks else 0.0,
        "queries": len(queries),
        "query_templates": dict(Counter(q.template for q in queries)),
        "mean_gold_facts": float(statistics.fmean(len(q.gold) for q in queries)) if queries else 0.0,
    }


def run_once(args: argparse.Namespace, entities: int, facts: int, n_queries: int, embedder: Embedder,
             nx_cap: int | None, backends: Sequence[str]) -> tuple[dict[str, Any], list[Row]]:
    counter = TokenCounter()
    started = time.perf_counter()
    world = WorldBuilder(entities, facts, args.seed).build()
    build_articles(world, counter, random.Random(args.seed + 1), args.chunk_tokens)
    queries = generate_queries(world, n_queries, args.hops, args.seed + 2)
    log.info("world: %d entities, %d facts, %d chunks, %d queries (%.1fs)",
             len(world.names), len(world.facts), len(world.chunks), len(queries), time.perf_counter() - started)

    cache = repo_path(args.cache_dir) if args.cache_dir else None
    fact_vecs = cached_encode(embedder, [f.text for f in world.facts], cache, "facts")
    chunk_vecs = cached_encode(embedder, [c.text for c in world.chunks], cache, "chunks")
    query_vecs = cached_encode(embedder, [q.text for q in queries], cache, "queries")

    contenders: list[dict[str, Any]] = []
    rows: list[Row] = []

    hg = HypergraphContender(world, fact_vecs, args.hops, args.top_k, args.decay_lambda, args.query_plan)
    log.info("hypergraph: ingested %d facts in %.1fs", len(world.facts), hg.ingest_secs)
    hg_results = timed_pass(queries, query_vecs, hg.retrieve, args.warmup)
    hg_rows = [score(world, q, hg.name, r, counter) for q, r in zip(queries, hg_results)]
    rows += hg_rows
    contenders.append(summarize(hg.name, hg.label, hg_rows, build_secs=hg.ingest_secs))
    del hg

    nx_queries, nx_vecs = queries, query_vecs
    if nx_cap and len(queries) > nx_cap:
        nx_queries, nx_vecs = queries[:nx_cap], query_vecs[:nx_cap]
    nx_temporal_results: list[Retrieval] = []
    for temporal in (False, True):
        g = NetworkXContender(world, fact_vecs, args.hops, args.top_k, temporal, args.query_plan)
        log.info("%s: built graph in %.1fs", g.name, g.ingest_secs)
        results = timed_pass(nx_queries, nx_vecs, g.retrieve, min(args.warmup, 10))
        g_rows = [score(world, q, g.name, r, counter) for q, r in zip(nx_queries, results)]
        rows += g_rows
        contenders.append(summarize(g.name, g.label, g_rows, build_secs=g.ingest_secs))
        if temporal:
            nx_temporal_results = results
        del g

    hg_tokens = statistics.median(r.tokens for r in hg_rows) if hg_rows else 0
    for backend in backends:
        v = VectorContender(world, chunk_vecs, backend, args.vector_k)
        log.info("%s: indexed %d chunks in %.1fs", v.name, len(world.chunks), v.ingest_secs)
        matched_k = args.vector_k
        for k in range(1, args.max_vector_k + 1):
            sample = [v.retrieve_k(vec, k) for vec in query_vecs[: min(200, len(queries))]]
            matched_k = k
            if statistics.median(counter.count(r.context) for r in sample) >= hg_tokens:
                break
        settings = [(args.vector_k, False)] + ([(matched_k, True)] if matched_k != args.vector_k else [])
        for k, matched in settings:
            name = f"{v.name}_k{k}"
            results = timed_pass(queries, query_vecs, lambda q, vec, k=k, v=v: v.retrieve_k(vec, k), args.warmup)
            v_rows = [score(world, q, name, r, counter) for q, r in zip(queries, results)]
            rows += v_rows
            contenders.append(summarize(name, v.label_for(k, matched), v_rows, k=k, token_matched=matched,
                                        build_secs=v.ingest_secs))
        del v

    parity = [
        len(set(a.facts) & set(b.facts)) / max(1, len(set(a.facts) | set(b.facts)))
        for a, b in zip(hg_results, nx_temporal_results)
    ]
    result = {
        "entities_requested": entities,
        "facts_requested": facts,
        "dataset": dataset_stats(world, queries),
        "contenders": contenders,
        "parity": {
            "hypergraph_vs_networkx_temporal_jaccard": float(statistics.fmean(parity)) if parity else None,
            "note": "Both rank the same reachable, valid facts by cosine; values near 1.0 confirm identical retrieval semantics.",
        },
        "networkx_queries": len(nx_queries),
    }
    return result, rows


def write_rows(path: Path, rows: Sequence[Row]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(Row.__dataclass_fields__.keys())
        for r in rows:
            writer.writerow([r.query, r.contender, f"{r.latency_ms:.4f}", r.tokens, r.retrieved, f"{r.precision:.4f}",
                             f"{r.recall:.4f}", f"{r.f1:.4f}", f"{r.stale_rate:.4f}", int(r.full_chain), int(r.final_hop)])


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--entities", type=int, default=12_000)
    p.add_argument("--facts", type=int, default=60_000)
    p.add_argument("--queries", type=int, default=1_000)
    p.add_argument("--hops", type=int, default=3, help="maximum k-hop depth for both graph contenders")
    p.add_argument("--query-plan", action=argparse.BooleanOptionalAction, default=True,
                   help="give both graph contenders the question's hop count and relation types, as a "
                        "text-to-query planner would (--no-query-plan: fixed --hops, all relation types)")
    p.add_argument("--top-k", type=int, default=16, help="fact budget for both graph contenders")
    p.add_argument("--vector-k", type=int, default=5, help="fixed chunk count for the vector contender")
    p.add_argument("--max-vector-k", type=int, default=40, help="upper bound when searching for the token-matched k")
    p.add_argument("--vector-backend", choices=("faiss", "qdrant", "both"), default="faiss")
    p.add_argument("--embedder", choices=("minilm", "hashing"), default="minilm")
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--dim", type=int, default=384, help="hashing embedder dimension")
    p.add_argument("--decay-lambda", type=float, default=1e-6,
                   help="engine decay per day; the tiny default disables decay so ranking is pure cosine, matching NetworkX")
    p.add_argument("--chunk-tokens", type=int, default=128)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--nx-query-cap", type=int, default=0, help="limit NetworkX contenders to the first N queries (0 = all)")
    p.add_argument("--scale-sweep", default="", help="comma-separated entity counts; facts = 5x entities")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--cache-dir", default="results/phase7/cache")
    p.add_argument("--out", default="results/phase7/competitor.json")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    args = parse_args(argv)
    out = repo_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    backends = ("faiss", "qdrant") if args.vector_backend == "both" else (args.vector_backend,)
    embedder = make_embedder(args)
    config = {k: v for k, v in vars(args).items()}
    config["embedder_name"] = embedder.name
    config["embedding_dim"] = embedder.dim
    report: dict[str, Any] = {"benchmark": "competitor_bench", "environment": environment(), "config": config}

    if args.scale_sweep:
        sizes = [int(s.replace("_", "")) for s in args.scale_sweep.split(",") if s]
        cap = args.nx_query_cap or 100
        report["sweep"] = []
        for entities in sizes:
            log.info("sweep: %d entities", entities)
            result, _ = run_once(args, entities, entities * 5, args.queries, embedder, cap, backends)
            report["sweep"].append(result)
            out.write_text(json.dumps(report, indent=2) + "\n")
    else:
        result, rows = run_once(args, args.entities, args.facts, args.queries, embedder, args.nx_query_cap or None, backends)
        report.update(result)
        write_rows(out.with_suffix(".csv"), rows)
        out.write_text(json.dumps(report, indent=2) + "\n")
        for c in result["contenders"]:
            log.info("%-26s p50=%8.3fms p99=%8.3fms tokens=%7.1f P=%.3f R=%.3f F1=%.3f stale=%.3f chain=%.3f",
                     c["name"], c["latency_ms"]["p50"], c["latency_ms"]["p99"], c["tokens"]["mean"],
                     c["precision"], c["recall"], c["f1"], c["stale_rate"], c["full_chain_recall"])
    log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
