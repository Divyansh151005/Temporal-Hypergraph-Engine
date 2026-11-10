"""vLLM + HypergraphKVOffloader smoke test for a single CUDA GPU (Kaggle T4)."""

import sys
import tempfile
import time

import torch
from vllm import LLM, SamplingParams

import hypergraph_engine_native as native
from hypergraph_offloader import HypergraphKVOffloader, KvLayout

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
PROMPT = "Explain temporal graphs."
NUM_BLOCKS = 8

# Qwen2.5-1.5B: 28 layers, 2 KV heads, head_dim 128. T4 has no bfloat16.
QWEN_1_5B_FP16 = KvLayout(num_layers=28, block_size=16, num_heads=2, head_dim=128, dtype=torch.float16)


def offloader_round_trip() -> None:
    engine = native.NativeHypergraphEngine(tempfile.mkdtemp(prefix="thg-"), sync="buffered")
    offloader = HypergraphKVOffloader(engine, layout=QWEN_1_5B_FP16)
    print(f"native engine {native.__version__}, cuda={offloader.has_cuda}, "
          f"block={QWEN_1_5B_FP16.expected_bytes / 2**20:.2f} MiB")

    edge_id = engine.insert_hyperedge_fast(
        "KV_PREFIX", ["session:kaggle", f"model:{MODEL}"], int(time.time()), None, b""
    )

    blocks = []
    parent = None
    for i in range(NUM_BLOCKS):
        key = offloader.hash_block_tokens(range(i * 16, (i + 1) * 16), parent)
        parent = key
        blocks.append((key, torch.randn(QWEN_1_5B_FP16.shape, dtype=torch.float16, device="cuda")))

    start = time.perf_counter()
    for i, (key, kv) in enumerate(blocks):
        if not offloader.offload_kv_tensor(key, kv, hyperedge_id=edge_id):
            raise RuntimeError(f"offload of block {i} was rejected")
    offload_s = time.perf_counter() - start

    start = time.perf_counter()
    for i, (key, kv) in enumerate(blocks):
        restored = torch.empty_like(kv)
        if not offloader.prefetch_kv_block(key, restored):
            raise RuntimeError(f"block {i} missing after offload")
        if not torch.equal(restored, kv):
            raise RuntimeError(f"block {i} differs after round trip")
    restore_s = time.perf_counter() - start
    offloader.close()

    total_mib = NUM_BLOCKS * QWEN_1_5B_FP16.expected_bytes / 2**20
    print(f"KV round trip OK: {NUM_BLOCKS} blocks linked to hyperedge {edge_id}, {total_mib:.1f} MiB, "
          f"offload {total_mib / offload_s:.0f} MiB/s, restore {total_mib / restore_s:.0f} MiB/s")


def vllm_generate() -> str:
    llm = LLM(
        model=MODEL,
        gpu_memory_utilization=0.4,
        dtype="half",
        max_model_len=2048,
        enforce_eager=True,
    )
    messages = [{"role": "user", "content": PROMPT}]
    outputs = llm.chat(messages, SamplingParams(max_tokens=128, temperature=0.7))
    return outputs[0].outputs[0].text


def main() -> int:
    if not torch.cuda.is_available():
        print("RESULT: FAIL (no CUDA device)")
        return 1
    print(f"GPU: {torch.cuda.get_device_name(0)} (torch {torch.__version__}, CUDA {torch.version.cuda})")
    offloader_round_trip()
    text = vllm_generate()
    print(f"\nPrompt: {PROMPT}\nResponse:\n{text}\n")
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
