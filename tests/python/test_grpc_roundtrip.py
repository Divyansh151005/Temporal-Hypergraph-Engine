"""Python gRPC client against the `hypergraph-server` binary over a Unix socket.

Uses the committed stubs and server launcher from `hypergraph-client`
(`pip install -e integrations/python-client`). Build the server first:
`cargo build --release -p grpc-server`.
"""

import numpy as np
import pytest

grpc = pytest.importorskip("grpc")
pytest.importorskip("hypergraph_client")

from hypergraph_client._proto import pb as messages  # noqa: E402
from hypergraph_client._proto import pb_grpc as services  # noqa: E402
from hypergraph_client.server import LocalServer, find_server_binary  # noqa: E402
from hypergraph_client.types import channel_options  # noqa: E402

DIM = 1536


@pytest.fixture(scope="session")
def pb():
    return messages, services


@pytest.fixture(scope="module")
def stub():
    binary = find_server_binary()
    if binary is None:
        pytest.skip("hypergraph-server not built; run `cargo build --release -p grpc-server`")
    with LocalServer(binary) as server:
        channel = grpc.insecure_channel(server.endpoint, options=channel_options(server.endpoint))
        grpc.channel_ready_future(channel).result(timeout=10)
        yield services.HypergraphEngineServiceStub(channel)
        channel.close()


def embedding(i: int) -> np.ndarray:
    return np.random.default_rng(i).standard_normal(DIM).astype(np.float32)


def insert_request(pb, label: str, valid_from: int, vec: np.ndarray):
    messages, _ = pb
    return messages.InsertHyperedgeRequest(
        session_id="pytest",
        hyperedge=messages.HyperedgePayload(
            event_type="OBSERVED",
            participants=[
                messages.Participant(entity_id=label, role="subject"),
                messages.Participant(entity_id="hub", role="object"),
            ],
            temporal_boundary=messages.TemporalBoundary(
                valid_from_epoch=valid_from, is_ongoing=True
            ),
            confidence_score=0.8,
            embedding_raw=vec.tobytes(),
        ),
    )


def test_health(pb, stub):
    messages, _ = pb
    health = stub.HealthCheck(messages.HealthCheckRequest())
    assert health.healthy


def test_thousand_edges_serialize_round_trip(pb, stub):
    messages, _ = pb
    count = 1_000
    vectors = {f"r{i}": embedding(i) for i in range(count)}
    batch = stub.InsertHyperedgeBatch(
        insert_request(pb, label, i, vec) for i, (label, vec) in enumerate(vectors.items())
    )
    assert batch.processed_count == count
    assert batch.all_succeeded, list(batch.errors)

    response = stub.QueryTemporalContext(
        messages.QueryTemporalRequest(
            query_embedding_raw=vectors["r0"].tobytes(),
            target_timestamp_epoch=10**9,
            top_k=count,
        )
    )
    assert len(response.matched_edges) == count
    assert response.matched_edges[0].participants[0].entity_id == "r0"
    for edge in response.matched_edges:
        decoded = np.frombuffer(edge.embedding_raw, dtype=np.float32)
        np.testing.assert_array_almost_equal(decoded, vectors[edge.participants[0].entity_id])

    streamed = list(
        stub.StreamTemporalContext(
            messages.QueryTemporalRequest(
                query_embedding_raw=vectors["r1"].tobytes(),
                target_timestamp_epoch=10**9,
                top_k=10,
            )
        )
    )
    assert len(streamed) == 10
    assert streamed[0].participants[0].entity_id == "r1"


def test_kv_block_over_grpc(pb, stub):
    messages, _ = pb
    block = np.random.default_rng(0).standard_normal((2, 32, 16, 32, 128)).astype(np.float16)
    assert block.nbytes == 8_388_608
    block_hash = "1f" * 32
    stored = stub.StoreKVBlocks(
        messages.StoreKVBlocksRequest(
            block_hash=block_hash,
            kv_chunk_bytes=block.tobytes(),
            layout=messages.KvBlockLayout(
                num_layers=32,
                num_heads=32,
                head_dim=128,
                block_size=16,
                precision=messages.KV_PRECISION_FP16,
            ),
        )
    )
    assert stored.success and not stored.deduplicated
    loaded = stub.LoadKVBlocks(messages.LoadKVBlocksRequest(block_hash=block_hash))
    assert loaded.hit
    restored = np.frombuffer(loaded.kv_chunk_bytes, dtype=np.float16).reshape(block.shape)
    np.testing.assert_array_equal(restored, block)


def test_misaligned_embedding_is_rejected(pb, stub):
    request = insert_request(pb, "bad", 0, np.zeros(0, dtype=np.float32))
    request.hyperedge.embedding_raw = b"\x00" * 7
    with pytest.raises(grpc.RpcError) as info:
        stub.InsertHyperedge(request)
    assert info.value.code() == grpc.StatusCode.INVALID_ARGUMENT
