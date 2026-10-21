"""PyO3 in-process bindings: alignment safety, round trips, KV FFI and latency."""

import time

import numpy as np
import pytest

import hypergraph_engine_native as native

DIM = 1536


def emb(seed: int, dim: int = DIM) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(dim).astype(np.float32)


def test_insert_and_query_round_trip_bit_exact(mem_engine):
    vectors = {f"e{i}": emb(i) for i in range(50)}
    ids = {
        label: mem_engine.insert_hyperedge_fast(
            "OBSERVED", [label, "hub"], 1_000 + i, None, bytes(vec.data)
        )
        for i, (label, vec) in enumerate(vectors.items())
    }
    results = mem_engine.query_temporal(bytes(vectors["e7"].data), 10_000, 0.0, 50)
    assert len(results) == 50
    assert results[0]["id"] == ids["e7"]
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)
    for result in results:
        label = result["participants"][0]
        assert result["participants"][1] == "hub"
        assert result["event_type"] == "OBSERVED"
        decoded = np.frombuffer(result["embedding"], dtype=np.float32)
        np.testing.assert_array_equal(decoded, vectors[label])


@pytest.mark.parametrize("length", [1, 3, 7, 6146])
def test_misaligned_embedding_raises_value_error(mem_engine, length):
    with pytest.raises(ValueError, match="alignment"):
        mem_engine.insert_hyperedge_fast("X", ["a"], 0, None, b"\x00" * length)
    with pytest.raises(ValueError, match="alignment"):
        mem_engine.query_temporal(b"\x00" * length, 0)
    assert mem_engine.stats()["hyperedges"] == 0


def test_dimension_mismatch_and_bad_ids():
    engine = native.NativeHypergraphEngine.in_memory(embedding_dim=4)
    with pytest.raises(ValueError, match="dimension"):
        engine.insert_hyperedge_fast("X", ["a"], 0, None, bytes(emb(0, 3).data))
    edge = engine.insert_hyperedge_fast("X", ["a"], 0, None, bytes(emb(0, 4).data))
    with pytest.raises(ValueError):
        engine.soft_delete_hyperedge("not-an-id")
    engine.soft_delete_hyperedge(edge)
    assert engine.stats()["hyperedges"] == 0


def test_empty_embedding_means_none(mem_engine):
    mem_engine.insert_hyperedge_fast("X", ["a", "b"], 0, 10, b"", confidence=0.5)
    assert mem_engine.stats()["hyperedges"] == 1
    assert mem_engine.query_temporal(bytes(emb(0, 4).data), 5) == []


def test_persistence_across_reopen(tmp_path):
    path = str(tmp_path / "db")
    engine = native.NativeHypergraphEngine(path, sync="buffered")
    for i in range(10):
        engine.insert_hyperedge_fast("E", [f"n{i}", "hub"], i, None, bytes(emb(i, 8).data))
    block = bytes(range(256)) * 4
    engine.store_kv_block(b"\x01" * 32, block)
    seq = engine.checkpoint()
    assert seq == engine.stats()["last_sequence"]
    del engine

    reopened = native.NativeHypergraphEngine(path, sync="buffered")
    assert reopened.stats()["hyperedges"] == 10
    assert reopened.load_kv_block(b"\x01" * 32) == block


def test_kv_raw_pointer_round_trip(tmp_engine):
    layout = (2, 2, 4, 1, 1)  # 2*2*1*2*4 elements * 2 bytes = 64 bytes
    src = np.arange(32, dtype=np.float16)
    key = native.blake3_token_hash(np.arange(16, dtype=np.int32).tobytes())
    assert tmp_engine.store_kv_block_raw(key, src.ctypes.data, src.nbytes, layout=layout)
    assert tmp_engine.kv_block_layout(key) == layout

    dst = np.zeros_like(src)
    assert tmp_engine.load_kv_block_raw(key, dst.ctypes.data, dst.nbytes)
    np.testing.assert_array_equal(dst, src)

    short = np.zeros(4, dtype=np.float16)
    with pytest.raises(ValueError, match="destination"):
        tmp_engine.load_kv_block_raw(key, short.ctypes.data, short.nbytes)
    assert not tmp_engine.load_kv_block_raw(b"\x00" * 32, dst.ctypes.data, dst.nbytes)
    with pytest.raises(ValueError, match="null"):
        tmp_engine.load_kv_block_raw(key, 0, 64)


def test_kv_validation_and_dedup(mem_engine):
    with pytest.raises(ValueError, match="32 bytes"):
        mem_engine.store_kv_block(b"short", b"data")
    with pytest.raises(ValueError, match="layout requires"):
        mem_engine.store_kv_block(b"\x02" * 32, b"\x00" * 10, layout=(2, 2, 4, 1, 1))
    with pytest.raises(ValueError, match="precision"):
        mem_engine.store_kv_block(b"\x02" * 32, b"\x00" * 64, layout=(2, 2, 4, 1, 9))

    payload = np.arange(64, dtype=np.uint8)
    mem_engine.store_kv_block(b"\x03" * 32, payload)
    mem_engine.store_kv_block(b"\x04" * 32, memoryview(payload.tobytes()))
    stats = mem_engine.stats()
    assert (stats["kv_blocks"], stats["kv_records"], stats["kv_dedup_hits"]) == (2, 1, 1)
    assert mem_engine.remove_kv_block(b"\x03" * 32)
    assert not mem_engine.has_kv_block(b"\x03" * 32)
    assert mem_engine.load_kv_block(b"\x04" * 32) == payload.tobytes()


def test_blake3_helpers():
    tokens = np.array([101, 2023, 2003], dtype=np.int32).tobytes()
    root = native.blake3_token_hash(tokens)
    assert len(root) == 32
    assert native.blake3_token_hash(tokens) == root
    assert native.blake3_token_hash(tokens, root) != root
    with pytest.raises(ValueError):
        native.blake3_token_hash(b"\x00" * 5)
    assert len(native.blake3_content_hash(b"abc")) == 32


def test_ffi_latency_and_ingest_rate(mem_engine):
    count = 20_000
    payloads = [bytes(emb(i % 64).data) for i in range(count)]
    labels = [[f"n{i}", "hub"] for i in range(count)]
    insert = mem_engine.insert_hyperedge_fast

    start = time.perf_counter()
    for i in range(count):
        insert("BENCH", labels[i], i, None, payloads[i])
    elapsed = time.perf_counter() - start
    rate = count / elapsed
    per_call_ms = elapsed / count * 1e3
    print(f"\nPyO3 ingest: {rate:,.0f} ops/s, {per_call_ms * 1e3:.1f} us/insert")
    assert rate > 20_000
    assert per_call_ms < 0.05

    probe = mem_engine.has_kv_block
    key = b"\x00" * 32
    start = time.perf_counter()
    for _ in range(100_000):
        probe(key)
    ffi_ms = (time.perf_counter() - start) / 100_000 * 1e3
    print(f"PyO3 FFI round trip: {ffi_ms * 1e3:.2f} us")
    assert ffi_ms < 0.01


def test_summary_round_trips_and_survives_reopen(tmp_path):
    engine = native.NativeHypergraphEngine(str(tmp_path / "db"), sync="buffered")
    edge = engine.insert_hyperedge_fast(
        "MESSAGE_HUMAN", ["session:1", "record:a"], 10, None, b"", summary="héllo"
    )
    engine.insert_hyperedge_fast("FOUNDED", ["Ada", "Acme"], 5, None, b"")
    del engine
    engine = native.NativeHypergraphEngine(str(tmp_path / "db"), sync="buffered")
    [hit] = engine.traverse(["session:1"], 1, 100)
    assert hit["id"] == edge
    assert hit["summary"] == "héllo"
    assert hit["embedding"] is None
    [plain] = engine.traverse(["Ada"], 1, 100)
    assert plain["summary"] is None


def test_traverse_hops_order_filters_and_rerank(mem_engine):
    for i, text in enumerate(["hi", "hello", "bye"]):
        mem_engine.insert_hyperedge_fast(
            "MESSAGE_AI" if i % 2 else "MESSAGE_HUMAN",
            ["session:x", f"record:{i}"],
            100 + i,
            None,
            b"",
            summary=text,
        )
    history = mem_engine.traverse(
        ["session:x"], 1, 1_000, top_k=0, chronological=True,
        event_types=["MESSAGE_HUMAN", "MESSAGE_AI"],
    )
    assert [h["summary"] for h in history] == ["bye", "hello", "hi"]
    assert len(mem_engine.traverse(["session:x"], 1, 1_000, top_k=2)) == 2
    assert mem_engine.traverse(["missing"], 1, 1_000) == []

    vectors = {"bc": [1.0, 0.0], "cd": [0.6, 0.8], "ab": [0.0, 1.0], "de": [1.0, 0.0]}
    for name, vec in vectors.items():
        mem_engine.insert_hyperedge_fast(
            "LINK", [name[0], name[1]], 0, None, np.asarray(vec, np.float32).tobytes(), summary=name
        )
    reached = mem_engine.traverse(["a"], 3, 10, top_k=0)
    assert sorted(r["summary"] for r in reached) == ["ab", "bc", "cd"]
    query = np.asarray([1.0, 0.0], np.float32).tobytes()
    ranked = mem_engine.traverse(["a"], 3, 10, query_bytes=query, top_k=0, with_embeddings=True)
    assert [r["summary"] for r in ranked] == ["bc", "cd", "ab"]
    assert ranked[0]["embedding"] is not None
