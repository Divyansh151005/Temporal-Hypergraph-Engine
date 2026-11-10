# hypergraph-offloader

Python KV-cache block offloader for the Temporal Hypergraph Engine. It moves
KV blocks between PyTorch tensors and the engine's content-addressed,
mmap-backed block store through a raw pointer, so each transfer copies the
bytes once. On CUDA hosts blocks are staged through page-locked memory; on
CPU and Apple Silicon hosts tensors are made contiguous and passed directly.

Verified on a Tesla T4 alongside vLLM 0.33: engine and vLLM share CUDA memory
side-by-side; blocks copied from GPU into the engine and back are bit-for-bit
identical. It is not yet registered as a vLLM KV connector; you call it from
your own code.

```bash
pip install hypergraph-offloader
```

```python
import torch
import hypergraph_engine_native as native
from hypergraph_offloader import HypergraphKVOffloader, KvLayout

layout = KvLayout(num_layers=28, block_size=16, num_heads=2, head_dim=128, dtype=torch.float16)
engine = native.NativeHypergraphEngine("./thg-data", sync="buffered")
offloader = HypergraphKVOffloader(engine, layout=layout)

key = offloader.hash_block_tokens(range(16))
block = torch.randn(layout.shape, dtype=torch.float16)
offloader.offload_kv_tensor(key, block)

restored = torch.empty_like(block)
assert offloader.prefetch_kv_block(key, restored) and torch.equal(restored, block)
offloader.close()
```

See [docs/04-vllm-kv-offloader.txt](https://github.com/Divyansh151005/Temporal-Hypergraph-Engine/blob/main/docs/04-vllm-kv-offloader.txt)
for the block format and eviction design.
