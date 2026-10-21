"""HypergraphKVOffloader: hardware fallbacks, bit-exact round trips, dedup and eviction."""

import threading

import numpy as np
import pytest
import torch

import hypergraph_engine_native as native
from hypergraph_offloader import (
    LLAMA3_8B_FP16,
    HypergraphKVOffloader,
    KvLayout,
    detect_device,
    detect_hardware,
)

SMALL = KvLayout(num_layers=2, block_size=4, num_heads=2, head_dim=8)  # 512 bytes
DAY = 86_400

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
needs_mps = pytest.mark.skipif(
    not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
    reason="Apple MPS not available",
)


def random_block(layout: KvLayout = SMALL, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(layout.shape, generator=generator).to(layout.dtype)


def same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(a.cpu().view(torch.int16), b.cpu().view(torch.int16))


@pytest.fixture
def offloader(mem_engine):
    return HypergraphKVOffloader(mem_engine, SMALL)


def test_layouts():
    assert LLAMA3_8B_FP16.shape == (2, 32, 16, 32, 128)
    assert LLAMA3_8B_FP16.expected_bytes == 8_388_608
    assert LLAMA3_8B_FP16.as_native() == (32, 32, 128, 16, 1)
    assert KvLayout(dtype=torch.bfloat16).precision_id == 2
    assert KvLayout.from_tensor(random_block()) == SMALL
    with pytest.raises(ValueError):
        KvLayout(dtype=torch.float32)


def test_hardware_detection_matches_torch():
    hardware = detect_hardware()
    assert hardware.has_cuda == torch.cuda.is_available()
    assert hardware.supports_pinned_memory == hardware.has_cuda
    assert detect_device() == hardware.device
    assert hardware.device in {"cuda", "mps", "cpu"}


@pytest.mark.parametrize("engine_fixture", ["mem_engine", "tmp_engine"])
def test_split_offload_evict_prefetch_bit_exact(request, engine_fixture):
    engine = request.getfixturevalue(engine_fixture)
    offloader = HypergraphKVOffloader(engine, SMALL)
    original = random_block()
    key = offloader.hash_block_tokens(range(16))
    assert offloader.offload_kv_block(key, original[0], original[1])

    target = original.clone()
    target.zero_()  # "evict" the device copy
    assert offloader.prefetch_kv_block(key, target)
    assert same_bits(target, original)


def test_async_prefetch_pipeline_is_bit_exact_and_ordered(offloader):
    blocks = [random_block(seed=s) for s in range(8)]
    keys = [offloader.hash_block_tokens([s, s + 1]) for s in range(8)]
    for key, block in zip(keys, blocks):
        assert offloader.offload_kv_tensor(key, block)
    targets = [torch.zeros_like(b) for b in blocks]
    futures = [offloader.prefetch_kv_block_async(k, t) for k, t in zip(keys, targets)]
    miss = offloader.prefetch_kv_block_async(offloader.hash_block_tokens([999]), torch.zeros_like(blocks[0]))
    assert [f.result(timeout=10) for f in futures] == [True] * len(blocks)
    assert miss.result(timeout=10) is False
    assert all(same_bits(t, b) for t, b in zip(targets, blocks))
    with pytest.raises(ValueError):
        offloader.prefetch_kv_block_async(keys[0], torch.zeros(3))
    offloader.close()
    offloader.close()


def test_combined_tensor_offload_is_zero_copy_for_contiguous_cpu(offloader):
    original = random_block(seed=1)
    key = offloader.hash_block_tokens([1, 2, 3])
    assert offloader.offload_kv_tensor(key, original)
    target = torch.empty_like(original)
    assert offloader.prefetch_kv_block(key, target)
    assert same_bits(target, original)


def test_non_contiguous_inputs_are_made_contiguous(offloader):
    base = random_block(seed=2)
    wide_k = torch.randn(*SMALL.kv_shape[:-1], SMALL.head_dim * 2).to(SMALL.dtype)
    key_half = wide_k[..., ::2]
    assert not key_half.is_contiguous()
    key = offloader.hash_block_tokens([7])
    assert offloader.offload_kv_block(key, key_half, base[1])

    restored = torch.empty(SMALL.shape, dtype=SMALL.dtype)
    assert offloader.prefetch_kv_block(key, restored)
    assert same_bits(restored[0], key_half)
    assert same_bits(restored[1], base[1])

    permuted = random_block(seed=3).permute(4, 3, 2, 1, 0).contiguous().permute(4, 3, 2, 1, 0)
    assert not permuted.is_contiguous()
    key2 = offloader.hash_block_tokens([8])
    assert offloader.offload_kv_tensor(key2, permuted)
    non_contiguous_target = torch.zeros(SMALL.shape[::-1], dtype=SMALL.dtype).permute(4, 3, 2, 1, 0)
    assert offloader.prefetch_kv_block(key2, non_contiguous_target)
    assert same_bits(non_contiguous_target, permuted)


def test_miss_leaves_target_untouched(offloader):
    target = random_block(seed=4)
    before = target.clone()
    assert not offloader.prefetch_kv_block(b"\x00" * 32, target)
    assert same_bits(target, before)


def test_identical_blocks_are_deduplicated(mem_engine, offloader):
    block = random_block(seed=5)
    offloader.offload_kv_tensor(offloader.hash_block_tokens([1]), block)
    offloader.offload_kv_tensor(offloader.hash_block_tokens([2]), block.clone())
    stats = mem_engine.stats()
    assert (stats["kv_blocks"], stats["kv_records"], stats["kv_dedup_hits"]) == (2, 1, 1)


def test_shape_dtype_and_size_validation(mem_engine, offloader):
    with pytest.raises(ValueError, match="shape"):
        offloader.offload_kv_block(b"\x01" * 32, torch.zeros(3), torch.zeros(3))
    with pytest.raises(ValueError, match="bfloat16|float16"):
        offloader.offload_kv_tensor(b"\x01" * 32, random_block().to(torch.bfloat16))
    with pytest.raises(ValueError, match="shape"):
        offloader.prefetch_kv_block(b"\x01" * 32, torch.zeros(2, 2))

    data = np.zeros(100, dtype=np.uint8)
    with pytest.raises(ValueError, match="layout requires"):
        mem_engine.store_kv_block_raw(b"\x01" * 32, data.ctypes.data, data.nbytes, layout=SMALL.as_native())


def test_token_hash_chain(offloader):
    root = offloader.hash_block_tokens(range(16))
    child = offloader.hash_block_tokens(range(16, 32), root)
    assert root == native.blake3_token_hash(np.arange(16, dtype="<i4").tobytes())
    assert child == native.blake3_token_hash(np.arange(16, 32, dtype="<i4").tobytes(), root)
    assert root != child


def test_eviction_plan_follows_hyperedge_decay(mem_engine):
    evicted: list[bytes] = []
    offloader = HypergraphKVOffloader(mem_engine, SMALL, on_evict=evicted.append)
    now = 400 * DAY
    fresh = mem_engine.insert_hyperedge_fast("CHAT", ["user", "topic:new"], now - DAY, None, b"")
    stale = mem_engine.insert_hyperedge_fast("CHAT", ["user", "topic:old"], now - 300 * DAY, None, b"")
    fresh_key = offloader.hash_block_tokens([1])
    stale_key = offloader.hash_block_tokens([2])
    orphan_key = offloader.hash_block_tokens([3])
    offloader.offload_kv_tensor(fresh_key, random_block(seed=10), hyperedge_id=fresh)
    offloader.offload_kv_tensor(stale_key, random_block(seed=11), hyperedge_id=stale)
    orphan = mem_engine.insert_hyperedge_fast("CHAT", ["user", "topic:gone"], now - DAY, None, b"")
    offloader.offload_kv_tensor(orphan_key, random_block(seed=12), hyperedge_id=orphan)
    mem_engine.soft_delete_hyperedge(orphan)

    plan = offloader.plan_evictions(0.5, now=now)
    assert plan == [orphan_key, stale_key]
    assert evicted == plan
    assert offloader.has_block(stale_key)


def test_concurrent_offloads_share_staging_safely(mem_engine):
    offloader = HypergraphKVOffloader(mem_engine, SMALL)
    blocks = {offloader.hash_block_tokens([i]): random_block(seed=100 + i) for i in range(32)}
    items = list(blocks.items())

    def worker(chunk):
        for key, block in chunk:
            assert offloader.offload_kv_block(key, block[0], block[1])

    threads = [threading.Thread(target=worker, args=(items[i::4],)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    target = torch.empty(SMALL.shape, dtype=SMALL.dtype)
    for key, block in items:
        assert offloader.prefetch_kv_block(key, target)
        assert same_bits(target, block)


def test_full_size_block_round_trip(tmp_engine):
    offloader = HypergraphKVOffloader(tmp_engine)
    block = random_block(LLAMA3_8B_FP16, seed=42)
    assert block.nbytes == 8_388_608
    key = offloader.hash_block_tokens(range(16))
    assert offloader.offload_kv_tensor(key, block)
    target = torch.zeros_like(block)
    assert offloader.prefetch_kv_block(key, target)
    assert same_bits(target, block)


@needs_mps
def test_mps_tensors_use_cpu_fallback(offloader):
    original = random_block(seed=20)
    on_mps = original.to("mps")
    key = offloader.hash_block_tokens([20])
    assert offloader.offload_kv_block(key, on_mps[0], on_mps[1])
    target = torch.zeros(SMALL.shape, dtype=SMALL.dtype, device="mps")
    assert offloader.prefetch_kv_block(key, target)
    assert same_bits(target, original)


@needs_cuda
def test_cuda_pinned_staging_round_trip(offloader):
    original = random_block(seed=30).cuda()
    key = offloader.hash_block_tokens([30])
    assert offloader._staging.is_pinned()
    assert offloader.offload_kv_block(key, original[0], original[1])
    target = torch.zeros_like(original)
    assert offloader.prefetch_kv_block(key, target)
    assert torch.equal(target, original)
