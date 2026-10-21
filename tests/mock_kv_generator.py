#!/usr/bin/env python3
"""Mock vLLM PagedAttention KV traffic through the PyO3 FFI, no LLM or GPU needed.

Generates FP16/BF16 blocks of shape [2, 32, 16, 32, 128] (exactly 8,388,608
bytes, the Llama-3-8B 16-token block), offloads them into the native engine
through ``HypergraphKVOffloader`` (raw ``data_ptr`` hand-off), prefetches them
back, and reports latency and throughput. Sampled blocks are checked
bit-for-bit.

    python tests/mock_kv_generator.py --blocks 64 --in-memory   # quick smoke run
    python tests/mock_kv_generator.py --blocks 1000             # 8 GB on disk
"""

import argparse
import shutil
import sys
import tempfile
import time
from typing import Optional

import torch

import hypergraph_engine_native as native
from hypergraph_offloader import HypergraphKVOffloader, KvLayout, detect_hardware

BLOCK_SHAPE = (2, 32, 16, 32, 128)
BLOCK_BYTES = 8_388_608
TOKENS_PER_BLOCK = 16
DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--blocks", type=int, default=1000, help="blocks to offload (8 MiB each)")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="fp16")
    parser.add_argument("--data-dir", help="engine directory (default: a fresh temporary directory)")
    parser.add_argument("--in-memory", action="store_true", help="use a volatile engine (RAM only)")
    parser.add_argument("--keep", action="store_true", help="keep the data directory afterwards")
    parser.add_argument("--verify-every", type=int, default=50, help="bit-check every Nth block")
    parser.add_argument("--pool", type=int, default=16, help="distinct random tensors to cycle through")
    parser.add_argument(
        "--mode",
        choices=["tensor", "split"],
        default="tensor",
        help="offload combined [2,...] tensors (zero-copy) or separate K/V halves",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q / 100 * (len(ordered) - 1))))]


def report(label: str, seconds: list[float]) -> None:
    ms = [s * 1e3 for s in seconds]
    gbps = len(seconds) * BLOCK_BYTES / sum(seconds) / 1e9
    print(
        f"{label:<9} p50 {percentile(ms, 50):7.3f} ms   p99 {percentile(ms, 99):7.3f} ms   "
        f"max {max(ms):7.3f} ms   {gbps:6.2f} GB/s"
    )


def stamp(block: torch.Tensor, index: int) -> None:
    """Writes `index` into the block's first elements so every block's bytes are unique."""
    bits = block.view(torch.int16).view(-1)
    bits[0] = index % 0x3C00
    bits[1] = index // 0x3C00


def open_engine(args: argparse.Namespace) -> tuple[native.NativeHypergraphEngine, Optional[str]]:
    if args.in_memory:
        if args.blocks * BLOCK_BYTES > 4 << 30:
            print(f"warning: --in-memory holds {args.blocks * BLOCK_BYTES / 1e9:.1f} GB in RAM")
        return native.NativeHypergraphEngine.in_memory(), None
    data_dir = args.data_dir or tempfile.mkdtemp(prefix="hg-mock-kv-")
    print(f"engine directory: {data_dir} (needs ~{args.blocks * BLOCK_BYTES / 1e9:.1f} GB of disk)")
    return native.NativeHypergraphEngine(data_dir, sync="buffered"), data_dir


def main() -> int:
    args = parse_args()
    dtype = DTYPES[args.dtype]
    layout = KvLayout(dtype=dtype)
    assert layout.shape == BLOCK_SHAPE and layout.expected_bytes == BLOCK_BYTES

    hardware = detect_hardware()
    print(f"device: {hardware.device} (cuda={hardware.has_cuda}, mps={hardware.has_mps}); dtype {dtype}")
    engine, data_dir = open_engine(args)
    offloader = HypergraphKVOffloader(engine, layout)

    generator = torch.Generator().manual_seed(args.seed)
    pool = [
        torch.randn(BLOCK_SHAPE, generator=generator, dtype=torch.float32).to(dtype)
        for _ in range(max(1, args.pool))
    ]
    for block in pool:
        assert block.nbytes == BLOCK_BYTES, block.nbytes

    keys: list[bytes] = []
    expected: dict[int, torch.Tensor] = {}
    store_seconds: list[float] = []
    parent: Optional[bytes] = None
    try:
        for i in range(args.blocks):
            block = pool[i % len(pool)]
            stamp(block, i)
            key = offloader.hash_block_tokens(range(i * TOKENS_PER_BLOCK, (i + 1) * TOKENS_PER_BLOCK), parent)
            start = time.perf_counter()
            if args.mode == "tensor":
                ok = offloader.offload_kv_tensor(key, block)
            else:
                ok = offloader.offload_kv_block(key, block[0], block[1])
            store_seconds.append(time.perf_counter() - start)
            assert ok
            if i % args.verify_every == 0 or i == args.blocks - 1:
                expected[i] = block.clone()
            keys.append(key)
            parent = key

        target = torch.empty(BLOCK_SHAPE, dtype=dtype)
        load_seconds: list[float] = []
        mismatches = 0
        for i, key in enumerate(keys):
            start = time.perf_counter()
            hit = offloader.prefetch_kv_block(key, target)
            load_seconds.append(time.perf_counter() - start)
            if not hit:
                print(f"block {i}: miss")
                mismatches += 1
            elif i in expected and not torch.equal(target.view(torch.int16), expected[i].view(torch.int16)):
                print(f"block {i}: bytes differ after round trip")
                mismatches += 1

        stats = engine.stats()
        total_gb = args.blocks * BLOCK_BYTES / 1e9
        print(f"\n{args.blocks} blocks x 8 MiB = {total_gb:.2f} GB via {args.mode} offload")
        report("offload", store_seconds)
        report("prefetch", load_seconds)
        print(
            f"verified {len(expected)} blocks bit-exact; mismatches: {mismatches}; "
            f"records {stats['kv_records']}, dedup hits {stats['kv_dedup_hits']}, "
            f"segments {stats['kv_segments']}"
        )
        if stats["kv_records"] != args.blocks:
            print("error: expected one record per unique block")
            mismatches += 1
        return 0 if mismatches == 0 else 1
    finally:
        del offloader, engine
        if data_dir is not None and not args.keep and args.data_dir is None:
            shutil.rmtree(data_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
