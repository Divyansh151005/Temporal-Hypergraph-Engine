"""Client round trips against a real ``hypergraph-server``."""

import asyncio
import uuid

import numpy as np
import pytest

from hypergraph_client import (
    AsyncHypergraphClient,
    AuthenticationError,
    ChatRecord,
    HashingEmbedder,
    Hyperedge,
    HypergraphClient,
    HypergraphKVOffloadTierManager,
    Role,
    SessionMemory,
    record_node,
)
from hypergraph_client.server import LocalServer, find_server_binary


def unique(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def test_insert_traverse_query_stream_delete(hypergraph_client: HypergraphClient):
    a, b, c, d = (unique(x) for x in "abcd")
    vectors = {(a, b): [0.0, 1.0], (b, c): [1.0, 0.0], (c, d): [0.6, 0.8]}
    ids = {
        pair: hypergraph_client.insert(
            Hyperedge("LINK", pair, 100, summary=f"{pair[0]}->{pair[1]}", embedding=vec)
        )
        for pair, vec in vectors.items()
    }
    assert hypergraph_client.health()

    hop1 = hypergraph_client.traverse([a], k_hops=1, as_of=200)
    assert [e.id for e in hop1] == [ids[(a, b)]]
    assert hop1[0].summary == f"{a}->{b}"
    assert hop1[0].participants == (a, b)
    assert hop1[0].valid_from == 100 and hop1[0].valid_to is None and hop1[0].is_ongoing

    ranked = hypergraph_client.traverse([a], k_hops=3, as_of=200, embedding=[1.0, 0.0], top_k=0)
    assert [e.id for e in ranked] == [ids[(b, c)], ids[(c, d)], ids[(a, b)]]
    np.testing.assert_allclose(ranked[0].embedding, [1.0, 0.0])

    global_hits = hypergraph_client.query([1.0, 0.0], as_of=200, top_k=50)
    assert ids[(b, c)] in [e.id for e in global_hits]
    streamed = list(hypergraph_client.stream([1.0, 0.0], as_of=200, top_k=2))
    assert len(streamed) == 2

    assert hypergraph_client.delete([ids[(a, b)], ids[(a, b)]]) == 1
    assert hypergraph_client.traverse([a], as_of=200) == []
    assert hypergraph_client.delete([]) == 0


def test_batch_insert_reports_item_errors(hypergraph_client: HypergraphClient):
    good = Hyperedge("E", [unique("x")], 1)
    bad = Hyperedge("E", [unique("y")], 10, valid_to=5)
    result = hypergraph_client.insert_batch([good, bad, good])
    assert result.processed == 3
    assert not result.ok
    assert len(result.errors) == 1 and result.errors[0].startswith("item 1")


def test_session_memory_orders_scopes_and_clears(hypergraph_server: LocalServer):
    client = HypergraphClient(hypergraph_server.endpoint)
    memory = SessionMemory(unique("s"), client, embedder=HashingEmbedder(64))
    other = SessionMemory(unique("s"), client)
    memory.add(
        [
            ChatRecord(Role.HUMAN, "my card was charged twice", entities=("billing",)),
            ChatRecord(Role.AI, "I refunded the duplicate charge"),
            ChatRecord(Role.HUMAN, "thanks, also update my address"),
        ]
    )
    other.add([ChatRecord(Role.HUMAN, "unrelated")])

    history = memory.history()
    assert [r.content for r in history] == [
        "my card was charged twice",
        "I refunded the duplicate charge",
        "thanks, also update my address",
    ]
    assert [r.role for r in history] == [Role.HUMAN, Role.AI, Role.HUMAN]
    assert history[0].entities == ("billing",)
    assert [r.content for r in memory.history(limit=1)] == ["thanks, also update my address"]

    top = memory.relevant("duplicate charge refund", k=1)
    assert top[0].summary == "I refunded the duplicate charge"

    assert memory.clear() == 3
    assert memory.history() == []
    assert [r.content for r in other.history()] == ["unrelated"]
    memory.add([ChatRecord(Role.HUMAN, f"m{i}") for i in range(5)])
    assert [r.content for r in memory.history()] == [f"m{i}" for i in range(5)]


def test_async_client_and_session_memory(hypergraph_server: LocalServer):
    async def scenario():
        async with AsyncHypergraphClient(hypergraph_server.endpoint) as client:
            node = unique("n")
            edge_id = await client.insert(Hyperedge("E", [node, record_node()], 5, summary="hi"))
            [hit] = await client.traverse([node], as_of=10)
            assert hit.id == edge_id and hit.summary == "hi"
            assert await client.health()
            batch = await client.insert_batch([Hyperedge("E", [node, record_node()], 6)])
            assert batch.ok
            streamed = [e async for e in client.stream([1.0], as_of=10, top_k=1)]
            assert len(streamed) <= 1
            assert await client.delete([edge_id]) == 1

        memory = SessionMemory(unique("s"), endpoint=hypergraph_server.endpoint)
        await memory.aadd([ChatRecord(Role.HUMAN, "a"), ChatRecord(Role.AI, "b")])
        assert [r.content for r in await memory.ahistory()] == ["a", "b"]
        assert await memory.aclear() == 2
        assert await memory.ahistory() == []

    asyncio.run(scenario())


def test_api_key_enforcement():
    binary = find_server_binary()
    if binary is None:
        pytest.skip("hypergraph-server not built")
    with LocalServer(binary, api_key="s3cret") as server:
        with HypergraphClient(server.endpoint, api_key="s3cret") as client:
            assert client.health()
        with HypergraphClient(server.endpoint, api_key="wrong") as client:
            with pytest.raises(AuthenticationError):
                client.health()


def test_kv_tier_manager_round_trips_blocks(hypergraph_server: LocalServer):
    tier = HypergraphKVOffloadTierManager({"grpc_endpoint": hypergraph_server.endpoint})
    try:
        chunk = np.random.default_rng(7).integers(0, 256, 64 * 1024, dtype=np.uint8).tobytes()
        block = uuid.uuid4().int
        assert tier.submit_load(block) is None
        assert tier.submit_store(block, chunk)
        assert tier.submit_load(block) == chunk
        assert tier.client.load_kv_block("0" * 64) is None
    finally:
        tier.close()
