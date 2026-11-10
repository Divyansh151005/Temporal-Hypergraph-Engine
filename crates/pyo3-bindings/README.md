# hypergraph-engine-native

In-process Python bindings for the [Temporal Hypergraph Engine](https://github.com/Divyansh151005/Temporal-Hypergraph-Engine):
a lock-free, Rust temporal hypergraph database with a content-addressed KV-cache
block store built for vLLM.

The module runs the engine inside your Python process, with no server and no
network hop. Facts are stored as time-bounded N-ary hyperedges, so a multi-hop
traversal at any point in time only follows facts that were true then.

```bash
pip install hypergraph-engine-native
```

Prebuilt wheels cover CPython 3.10 to 3.13 on Linux (x86_64, aarch64), macOS and
Windows.

## Temporal graph

```python
from hypergraph_engine_native import NativeHypergraphEngine

engine = NativeHypergraphEngine.in_memory()      # or NativeHypergraphEngine("./thg-data")

engine.insert_hyperedge_fast("CEO_OF", ["person:ballmer", "org:microsoft"], 946684800, 1391212800, b"")
engine.insert_hyperedge_fast("CEO_OF", ["person:nadella", "org:microsoft"], 1391212800, None, b"")

# Who ran Microsoft on 1 Jan 2010? Only facts valid at `as_of` are returned or expanded.
# decay_lambda is per day; 0 selects the engine default (30-day half-life).
for edge in engine.traverse(["org:microsoft"], k_hops=2, as_of=1262304000, decay_lambda=1e-6):
    print(edge["event_type"], edge["participants"], edge["score"])
```

`embedding_buffer` takes little-endian `float32` bytes (for example
`np.asarray(vec, dtype=np.float32).tobytes()`); pass `embedding_dim` to the
constructor and `query_bytes` to `query_temporal` / `traverse` to rank by
similarity.

## KV-cache block store

The same engine stores vLLM-style KV blocks in 64-byte-aligned, append-only
`mmap` segments. `store_kv_block_raw` and `load_kv_block_raw` take a raw pointer
and length (for example `torch.Tensor.data_ptr()`) and copy straight between the
tensor's memory and the segment, with no intermediate Python `bytes`.

```python
import torch
from hypergraph_engine_native import NativeHypergraphEngine, blake3_content_hash

engine = NativeHypergraphEngine("./thg-data", sync="buffered")
block = torch.randn(16, 2, 128, dtype=torch.float16).contiguous()
key = blake3_content_hash(block.numpy().tobytes())

engine.store_kv_block_raw(key, block.data_ptr(), block.numel() * block.element_size())

restored = torch.empty_like(block)
engine.load_kv_block_raw(key, restored.data_ptr(), restored.numel() * restored.element_size())
assert torch.equal(block, restored)
```

The raw-pointer calls are unsafe by contract: the pointer must address that many
live, contiguous bytes. For a higher-level, tensor-aware API (pinned-memory
staging on CUDA, layout checks, eviction planning), use
[`hypergraph-offloader`](https://pypi.org/project/hypergraph-offloader/).

## Related packages

| Package | Purpose |
|---|---|
| [`hypergraph-client`](https://pypi.org/project/hypergraph-client/) | gRPC client for the standalone server |
| [`langchain-hypergraph`](https://pypi.org/project/langchain-hypergraph/) | LangChain chat memory and vector store |
| [`llama-index-hypergraph`](https://pypi.org/project/llama-index-hypergraph/) | LlamaIndex retriever and chat memory |
| [`hypergraph-offloader`](https://pypi.org/project/hypergraph-offloader/) | KV-cache block offloader for PyTorch / vLLM |
| [`@temporal-hypergraph/sdk`](https://www.npmjs.com/package/@temporal-hypergraph/sdk) | TypeScript client and Vercel AI SDK memory |

## License

Licensed under the [Server Side Public License, version 1](https://github.com/Divyansh151005/Temporal-Hypergraph-Engine/blob/main/LICENSE)
(SSPL-1.0). Free to run, modify and embed in your own applications; offering the
engine to third parties as a managed service requires releasing your service
stack under the SSPL. Versions below 2.0.0 are deprecated and unsupported.

Source, benchmarks and issue tracker: <https://github.com/Divyansh151005/Temporal-Hypergraph-Engine>
