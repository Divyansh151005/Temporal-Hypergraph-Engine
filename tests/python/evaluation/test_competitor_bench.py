"""Competitor benchmark: world/query generation, metric math and contender semantics (hashing embedder, no network)."""

import json
import random

import pytest

from hypergraph_evaluation.competitor_bench import (
    YEAR,
    Fact,
    HashEmbedder,
    HypergraphContender,
    NetworkXContender,
    Query,
    Retrieval,
    TokenCounter,
    World,
    WorldBuilder,
    build_articles,
    generate_queries,
    main,
    score,
)

T0 = 1_500_000_000


@pytest.fixture(scope="module")
def small_world():
    world = WorldBuilder(entities=400, facts=2_000, seed=3).build()
    build_articles(world, TokenCounter(), random.Random(4))
    return world


def test_gold_chains_are_valid_and_connected(small_world):
    queries = generate_queries(small_world, 40, max_hops=3, seed=5)
    assert len(queries) == 40
    for q in queries:
        chain = [small_world.facts[f] for f in q.gold]
        assert all(f.valid_at(q.as_of) for f in chain)
        assert q.seeds[0] in chain[0].participants
        for prev, nxt in zip(chain, chain[1:]):
            assert set(prev.participants) & set(nxt.participants)
        assert q.answer in chain[-1].participants
        assert set(q.event_types) == {f.event_type for f in chain}


def test_every_fact_lands_in_exactly_one_chunk(small_world):
    seen = [f for c in small_world.chunks for f in c.fact_ids]
    assert sorted(seen) == list(range(len(small_world.facts)))
    assert all(c.tokens <= 160 for c in small_world.chunks)


def toy_world():
    names = ["Acme Robotics", "Orion X1", "Ava Stone", "Leo Grant", "Port Vale"]
    kinds = ["company", "product", "person", "person", "city"]
    facts = [
        Fact(0, "MAKES", (0, 1), T0, None, "Acme Robotics manufactures Orion X1."),
        Fact(1, "CEO_OF", (3, 0), T0, T0 + YEAR, "Leo Grant became CEO of Acme Robotics and stepped down."),
        Fact(2, "CEO_OF", (2, 0), T0 + YEAR, None, "Ava Stone became CEO of Acme Robotics."),
        Fact(3, "HEADQUARTERED_IN", (0, 4), T0, None, "Acme Robotics has been headquartered in Port Vale."),
    ]
    query = Query("q0", "ceo_of_maker", "Who is the CEO of the company that makes Orion X1?", (1,),
                  T0 + 2 * YEAR, (0, 2), 2, ("MAKES", "CEO_OF"))
    return World(names, kinds, facts), query


@pytest.mark.parametrize("planned", [True, False])
def test_temporal_contenders_retrieve_the_current_chain_only(planned):
    world, query = toy_world()
    embedder = HashEmbedder(64)
    vectors = embedder.encode([f.text for f in world.facts])
    qvec = embedder.encode([query.text])[0]
    counter = TokenCounter()

    hg = HypergraphContender(world, vectors, hops=3, top_k=16, decay_lambda=1e-6, planned=planned)
    temporal = NetworkXContender(world, vectors, hops=3, top_k=16, temporal=True, planned=planned)
    naive = NetworkXContender(world, vectors, hops=3, top_k=16, temporal=False, planned=planned)

    hg_row = score(world, query, "hypergraph", hg.retrieve(query, qvec), counter)
    assert hg_row.recall == 1.0 and hg_row.full_chain and hg_row.stale_rate == 0.0
    assert set(hg.retrieve(query, qvec).facts) == set(temporal.retrieve(query, qvec).facts)

    naive_facts = set(naive.retrieve(query, qvec).facts)
    assert 1 in naive_facts  # the former CEO leaks without the temporal filter
    if planned:
        assert 3 not in naive_facts  # HEADQUARTERED_IN is outside the planned relation types


def test_score_math():
    world, query = toy_world()
    row = score(world, query, "x", Retrieval([0, 1, 3], "context text", 2_000_000), TokenCounter())
    assert row.precision == pytest.approx(1 / 3)
    assert row.recall == pytest.approx(1 / 2)
    assert row.f1 == pytest.approx(0.4)
    assert row.stale_rate == pytest.approx(1 / 3)
    assert not row.full_chain and not row.final_hop
    assert row.latency_ms == pytest.approx(2.0)

    empty = score(world, query, "x", Retrieval([], "", 0), TokenCounter())
    assert empty.precision == 0.0 and empty.recall == 0.0 and empty.f1 == 0.0


def test_end_to_end_writes_report(tmp_path):
    out = tmp_path / "competitor.json"
    assert main([
        "--entities", "300", "--facts", "1500", "--queries", "30", "--embedder", "hashing", "--dim", "64",
        "--vector-backend", "qdrant", "--warmup", "2", "--cache-dir", "", "--out", str(out),
    ]) == 0
    data = json.loads(out.read_text())
    names = [c["name"] for c in data["contenders"]]
    assert names[:3] == ["hypergraph", "networkx_naive", "networkx_temporal"]
    assert any(n.startswith("vector_qdrant") for n in names)
    assert data["parity"]["hypergraph_vs_networkx_temporal_jaccard"] == pytest.approx(1.0, abs=0.02)
    assert out.with_suffix(".csv").exists()
