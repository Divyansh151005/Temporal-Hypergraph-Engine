"""Client helpers that need no server."""

import filecmp
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from hypergraph_client import (
    DEFAULT_SESSION,
    ChatRecord,
    HashingEmbedder,
    Hyperedge,
    Role,
    SessionMemory,
    TemporalHypergraph,
    block_key,
    entities_of,
    from_f32_bytes,
    record_node,
    record_of,
    scoped_participants,
    session_node,
    to_f32_bytes,
)
from hypergraph_client.testing import FakeHypergraphClient
from hypergraph_client.types import channel_options, resolve_endpoint

ROOT = Path(__file__).resolve().parents[3]
PROTO_OUT = ROOT / "integrations/python-client/src/hypergraph_client/_proto"


def test_session_and_record_nodes():
    assert session_node("abc") == "session:abc"
    with pytest.raises(ValueError):
        session_node("")
    records = [record_node() for _ in range(100)]
    assert records == sorted(records)
    assert len(set(records)) == 100
    participants = scoped_participants("s", ["Acme", "", "Acme", records[0]])
    assert participants == ["session:s", "Acme", records[0]]
    assert record_of(participants) == records[0]
    assert entities_of(participants) == ["Acme"]
    assert scoped_participants(None, ["a"]) == ["a"]


def test_hashing_embedder_is_deterministic_and_semantic_ish():
    embedder = HashingEmbedder(dim=128)
    a = np.asarray(embedder.embed_query("Microsoft acquired Activision"))
    b = np.asarray(embedder.embed_documents(["Microsoft acquired Activision"])[0])
    c = np.asarray(embedder.embed_query("The weather in Paris is mild"))
    d = np.asarray(embedder.embed_query("Activision was acquired by Microsoft"))
    np.testing.assert_allclose(a, b)
    assert abs(np.linalg.norm(a) - 1.0) < 1e-5
    assert a @ d > a @ c
    assert not np.any(embedder.embed(""))


def test_f32_packing_round_trip():
    vector = [0.5, -1.25, 3.0]
    raw = to_f32_bytes(vector)
    assert len(raw) == 12
    np.testing.assert_array_equal(from_f32_bytes(raw), np.asarray(vector, np.float32))
    assert to_f32_bytes(None) == b""
    assert from_f32_bytes(b"") is None


def test_hyperedge_to_pb():
    ongoing = Hyperedge("E", ["a", "b"], 10, summary="s").to_pb()
    assert ongoing.temporal_boundary.is_ongoing
    assert not ongoing.temporal_boundary.HasField("valid_to_epoch")
    bounded = Hyperedge("E", ["a"], 10, valid_to=20).to_pb()
    assert bounded.temporal_boundary.valid_to_epoch == 20
    assert not bounded.temporal_boundary.is_ongoing


def test_roles_map_to_event_types():
    assert Role.HUMAN.event_type == "MESSAGE_HUMAN"
    assert Role.from_event_type("MESSAGE_AI") is Role.AI
    assert Role.from_event_type("ACQUISITION") is None
    assert Role.from_event_type("MESSAGE_UNKNOWN") is None


def test_endpoint_resolution(monkeypatch):
    monkeypatch.delenv("HYPERGRAPH_ENDPOINT", raising=False)
    assert resolve_endpoint(None) == "localhost:50051"
    assert resolve_endpoint("tcp://10.0.0.1:9000") == "10.0.0.1:9000"
    monkeypatch.setenv("HYPERGRAPH_ENDPOINT", "unix:///tmp/x.sock")
    target = resolve_endpoint(None)
    assert target == "unix:///tmp/x.sock"
    assert ("grpc.default_authority", "localhost") in channel_options(target)


@pytest.mark.skipif(importlib.util.find_spec("grpc_tools") is None, reason="grpcio-tools missing")
def test_committed_stubs_match_proto(tmp_path):
    from grpc_tools import protoc

    proto_dir = ROOT / "proto"
    status = protoc.main(
        [
            "grpc_tools.protoc",
            f"-I{proto_dir}",
            f"--python_out={tmp_path}",
            f"--pyi_out={tmp_path}",
            f"--grpc_python_out={tmp_path}",
            str(proto_dir / "hypergraph_service.proto"),
        ]
    )
    assert status == 0
    grpc_stub = tmp_path / "hypergraph_service_pb2_grpc.py"
    grpc_stub.write_text(
        grpc_stub.read_text().replace(
            "import hypergraph_service_pb2 as", "from . import hypergraph_service_pb2 as"
        )
    )
    for name in ("hypergraph_service_pb2.py", "hypergraph_service_pb2.pyi", grpc_stub.name):
        assert filecmp.cmp(tmp_path / name, PROTO_OUT / name, shallow=False), (
            f"{name} is stale; run scripts/gen_protos.sh python"
        )


def test_temporal_hypergraph_one_liner_defaults_to_a_session():
    fake = FakeHypergraphClient()
    memory = TemporalHypergraph(api_key="k", client=fake)
    assert memory.session_id == DEFAULT_SESSION and memory.client is fake
    memory.add([ChatRecord(Role.HUMAN, "remember me")])
    assert [r.content for r in SessionMemory(DEFAULT_SESSION, fake).history()] == ["remember me"]
    assert TemporalHypergraph(session_id="s9", client=fake).node == session_node("s9")


def test_block_key_normalizes_vllm_hashes():
    digest = bytes(range(32))
    assert block_key(digest) == digest.hex()
    assert block_key(digest.hex().upper()) == digest.hex()
    for other in (12345, -7, "prefix-block", b"short"):
        key = block_key(other)
        assert len(key) == 64 and int(key, 16) >= 0
        assert key == block_key(other)
    assert block_key(1) != block_key(2)
