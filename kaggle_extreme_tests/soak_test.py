"""End-to-end soak: 500 vLLM generations with periodic KV offload/restore through
the Rust engine, sampling process RSS and device VRAM to detect leaks.

Run with VLLM_ENABLE_V1_MULTIPROCESSING=0 so vLLM lives in this process and the
RSS / CUDA readings cover both vLLM and the offloader.

The KV store appends records to mmap'd segment files and `remove_kv_block` does
not reclaim record space, so file-backed RSS grows by the bytes appended. The
leak check therefore uses anonymous RSS (heap) and reports file-backed growth
against the bytes the store appended.
"""

import json
import os
import sys
import tempfile
import time

import psutil
import pynvml
import torch
from vllm import LLM, SamplingParams

import hypergraph_engine_native as native
from hypergraph_offloader import HypergraphKVOffloader, KvLayout

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ITERATIONS = 500
OFFLOAD_EVERY = 10
SAMPLE_EVERY = 50
BASELINE_ITERATION = 50
BLOCKS_PER_CYCLE = 32
MAX_TOKENS = 64
RSS_GROWTH_LIMIT = 256 << 20
VRAM_GROWTH_LIMIT = 64 << 20
SUMMARY_PATH = "/kaggle/working/soak_summary.json" if os.path.isdir("/kaggle/working") else "soak_summary.json"

# Qwen2.5-1.5B: 28 layers, 2 KV heads, head_dim 128. T4 has no bfloat16.
QWEN_1_5B_FP16 = KvLayout(num_layers=28, block_size=16, num_heads=2, head_dim=128, dtype=torch.float16)

PROMPTS = [
    "What is a temporal graph?",
    "Explain the difference between a graph and a hypergraph.",
    "What does a KV cache store in a transformer?",
    "Why do language models use attention?",
    "What is the capital of France?",
    "Summarize the causes of the French Revolution in two sentences.",
    "How does a hash function differ from encryption?",
    "What is the time complexity of binary search?",
    "Explain what a write-ahead log is.",
    "What is exponential decay?",
    "Name three uses of graph databases.",
    "What is retrieval-augmented generation?",
    "How does paged attention reduce memory fragmentation?",
    "What is the difference between RAM and VRAM?",
    "Explain eventual consistency in one paragraph.",
    "What is a lock-free data structure?",
    "Why is fp16 faster than fp32 on GPUs?",
    "What does BLAKE3 hashing provide?",
    "Who wrote Pride and Prejudice?",
    "What is photosynthesis?",
    "Explain the CAP theorem briefly.",
    "What is a memory leak?",
    "How do vector databases find similar items?",
    "What is multi-hop reasoning?",
    "Describe a use case for time-aware knowledge graphs.",
]


class Monitor:
    def __init__(self):
        pynvml.nvmlInit()
        self._gpu = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
        self._proc = psutil.Process()

    @staticmethod
    def _proc_status_bytes() -> dict:
        """RssAnon / RssFile / RssShmem from /proc/self/status (Linux)."""
        fields = {"RssAnon": "rss_anon_bytes", "RssFile": "rss_file_bytes", "RssShmem": "rss_shmem_bytes"}
        values = dict.fromkeys(fields.values())
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    key, _, rest = line.partition(":")
                    if key in fields:
                        values[fields[key]] = int(rest.split()[0]) * 1024
        except OSError:
            pass
        return values

    def snapshot(self) -> dict:
        return {
            "rss_bytes": self._proc.memory_info().rss,
            **self._proc_status_bytes(),
            "vram_used_bytes": pynvml.nvmlDeviceGetMemoryInfo(self._gpu).used,
            "cuda_allocated_bytes": torch.cuda.memory_allocated(),
            "cuda_reserved_bytes": torch.cuda.memory_reserved(),
        }


class OffloadCycler:
    """Offloads, restores, verifies and removes BLOCKS_PER_CYCLE CUDA blocks."""

    def __init__(self):
        self.engine = native.NativeHypergraphEngine(tempfile.mkdtemp(prefix="thg-soak-"), sync="buffered")
        self.offloader = HypergraphKVOffloader(self.engine, layout=QWEN_1_5B_FP16)
        self.cycles = 0
        self.blocks_moved = 0
        self.mismatches = 0
        self.leftover_blocks = 0
        self.offload_secs = 0.0
        self.restore_secs = 0.0

    def cycle(self, iteration: int) -> None:
        edge_id = self.engine.insert_hyperedge_fast(
            "KV_PREFIX", [f"session:soak-{iteration}", f"model:{MODEL}"], int(time.time()), None, b""
        )
        blocks = []
        parent = None
        for i in range(BLOCKS_PER_CYCLE):
            start_token = (iteration * BLOCKS_PER_CYCLE + i) * QWEN_1_5B_FP16.block_size
            key = self.offloader.hash_block_tokens(range(start_token, start_token + QWEN_1_5B_FP16.block_size), parent)
            parent = key
            blocks.append((key, torch.randn(QWEN_1_5B_FP16.shape, dtype=torch.float16, device="cuda")))

        start = time.perf_counter()
        for key, kv in blocks:
            if not self.offloader.offload_kv_tensor(key, kv, hyperedge_id=edge_id):
                raise RuntimeError(f"offload rejected at iteration {iteration}")
        self.offload_secs += time.perf_counter() - start

        start = time.perf_counter()
        for key, kv in blocks:
            restored = torch.empty_like(kv)
            if not self.offloader.prefetch_kv_block(key, restored) or not torch.equal(restored, kv):
                self.mismatches += 1
        self.restore_secs += time.perf_counter() - start

        for key, _ in blocks:
            self.engine.remove_kv_block(key)
            if self.engine.has_kv_block(key):
                self.leftover_blocks += 1
        self.engine.soft_delete_hyperedge(edge_id)
        self.cycles += 1
        self.blocks_moved += len(blocks)

    def throughput(self) -> dict:
        mib = self.blocks_moved * QWEN_1_5B_FP16.expected_bytes / 2**20
        return {
            "offload_mib_s": mib / self.offload_secs if self.offload_secs else 0.0,
            "restore_mib_s": mib / self.restore_secs if self.restore_secs else 0.0,
        }


def mib(n: int) -> str:
    return f"{n / 2**20:9.1f} MiB"


def main() -> int:
    if not torch.cuda.is_available():
        print("SOAK: FAIL (no CUDA device)")
        return 1
    if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") != "0":
        print("warning: VLLM_ENABLE_V1_MULTIPROCESSING is not 0; RSS/VRAM readings exclude the vLLM engine process")
    print(f"GPU: {torch.cuda.get_device_name(0)} (torch {torch.__version__}), native engine {native.__version__}", flush=True)

    monitor = Monitor()
    cycler = OffloadCycler()
    llm = LLM(model=MODEL, gpu_memory_utilization=0.5, dtype="half", max_model_len=2048)
    params = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.7)

    samples = []
    tokens = 0
    gen_secs = 0.0
    soak_start = time.perf_counter()

    def record(iteration: int) -> None:
        prev = samples[-1] if samples else {"tokens": 0, "generation_secs": 0.0}
        window_secs = gen_secs - prev["generation_secs"]
        sample = {
            "iteration": iteration,
            "elapsed_secs": time.perf_counter() - soak_start,
            "tokens": tokens,
            "generation_secs": gen_secs,
            "tokens_per_sec": tokens / gen_secs if gen_secs else 0.0,
            "window_tokens_per_sec": (tokens - prev["tokens"]) / window_secs if window_secs else 0.0,
            "offload_cycles": cycler.cycles,
            "kv_bytes_appended": cycler.blocks_moved * QWEN_1_5B_FP16.expected_bytes,
            **cycler.throughput(),
            **monitor.snapshot(),
        }
        samples.append(sample)
        anon = sample["rss_anon_bytes"]
        file_backed = sample["rss_file_bytes"]
        split = f" (anon {mib(anon)}, file {mib(file_backed)})" if anon is not None else ""
        print(
            f"[iter {iteration:3d}] RSS {mib(sample['rss_bytes'])}{split} | VRAM used {mib(sample['vram_used_bytes'])} | "
            f"torch alloc {mib(sample['cuda_allocated_bytes'])} reserved {mib(sample['cuda_reserved_bytes'])} | "
            f"{sample['window_tokens_per_sec']:5.1f} tok/s | cycles {cycler.cycles} | "
            f"offload {sample['offload_mib_s']:6.0f} MiB/s restore {sample['restore_mib_s']:6.0f} MiB/s",
            flush=True,
        )

    record(0)
    for iteration in range(1, ITERATIONS + 1):
        messages = [{"role": "user", "content": PROMPTS[(iteration - 1) % len(PROMPTS)]}]
        start = time.perf_counter()
        output = llm.chat(messages, params, use_tqdm=False)[0].outputs[0]
        gen_secs += time.perf_counter() - start
        tokens += len(output.token_ids)
        if iteration % OFFLOAD_EVERY == 0:
            cycler.cycle(iteration)
        if iteration % SAMPLE_EVERY == 0:
            record(iteration)
    cycler.offloader.close()

    baseline = next(s for s in samples if s["iteration"] == BASELINE_ITERATION)
    final = samples[-1]
    rss_growth = final["rss_bytes"] - baseline["rss_bytes"]
    vram_growth = final["vram_used_bytes"] - baseline["vram_used_bytes"]
    has_split = final["rss_anon_bytes"] is not None and baseline["rss_anon_bytes"] is not None
    anon_growth = final["rss_anon_bytes"] - baseline["rss_anon_bytes"] if has_split else None
    file_growth = final["rss_file_bytes"] - baseline["rss_file_bytes"] if has_split else None
    appended = final["kv_bytes_appended"] - baseline["kv_bytes_appended"]
    leak_metric = anon_growth if has_split else rss_growth
    checks = {
        "heap_growth_ok": leak_metric < RSS_GROWTH_LIMIT,
        "vram_growth_ok": vram_growth < VRAM_GROWTH_LIMIT,
        "no_mismatches": cycler.mismatches == 0,
        "no_leftover_blocks": cycler.leftover_blocks == 0,
    }
    passed = all(checks.values())
    summary = {
        "model": MODEL,
        "iterations": ITERATIONS,
        "max_tokens": MAX_TOKENS,
        "offload_every": OFFLOAD_EVERY,
        "blocks_per_cycle": BLOCKS_PER_CYCLE,
        "block_bytes": QWEN_1_5B_FP16.expected_bytes,
        "gpu": torch.cuda.get_device_name(0),
        "total_tokens": tokens,
        "generation_secs": gen_secs,
        "offload_cycles": cycler.cycles,
        "blocks_moved": cycler.blocks_moved,
        "mismatches": cycler.mismatches,
        "leftover_blocks": cycler.leftover_blocks,
        "baseline_iteration": BASELINE_ITERATION,
        "rss_growth_bytes": rss_growth,
        "rss_anon_growth_bytes": anon_growth,
        "rss_file_growth_bytes": file_growth,
        "kv_bytes_appended_since_baseline": appended,
        "leak_metric": "rss_anon" if has_split else "rss",
        "vram_growth_bytes": vram_growth,
        "rss_growth_limit_bytes": RSS_GROWTH_LIMIT,
        "vram_growth_limit_bytes": VRAM_GROWTH_LIMIT,
        "checks": checks,
        "passed": passed,
        "samples": samples,
    }
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    split = (f" (anon {anon_growth / 2**20:+.1f} MiB, file-backed {file_growth / 2**20:+.1f} MiB "
             f"vs {appended / 2**20:.1f} MiB appended to KV segments)") if has_split else ""
    print(
        f"\n{cycler.cycles} offload cycles, {cycler.blocks_moved} blocks, {cycler.mismatches} mismatches, "
        f"{cycler.leftover_blocks} leftover | RSS growth {rss_growth / 2**20:+.1f} MiB{split}, "
        f"VRAM growth {vram_growth / 2**20:+.1f} MiB (iter {BASELINE_ITERATION} -> {ITERATIONS}) | "
        f"{tokens} tokens at {tokens / gen_secs:.1f} tok/s"
    )
    print(f"SOAK: {'PASS' if passed else 'FAIL'} {checks}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
