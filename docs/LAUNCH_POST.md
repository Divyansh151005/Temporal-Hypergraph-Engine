Title: Show HN: Temporal Hypergraph Engine – A lock-free temporal graph in Rust

URL: https://temporal-hypergraph-web.onrender.com/

---

Hi HN,

I built Temporal Hypergraph Engine, a lock-free temporal hypergraph database written in Rust, with a zero-copy CUDA memory offloader for LLM KV-cache blocks. It stores facts as time-bounded N-ary hyperedges and answers "what was true at time T?" during the traversal itself, not as a filter afterwards.

Live site (interactive demo): https://temporal-hypergraph-web.onrender.com/
Code: https://github.com/Divyansh151005/Temporal-Hypergraph-Engine

I am a solo developer. I built the engine primarily on my M3 Apple Silicon laptop, and leaned on Kaggle's free T4 GPUs for the heavy-VRAM vLLM offloader soak testing that my laptop could not run.

Why I built it

- LLM KV-caches grow endlessly. Long-running agents and serving stacks keep accumulating context, and GPU memory is the most expensive place to keep cold blocks. I wanted a tier that could take blocks off the GPU and restore them bit-exact, and decide what to evict using the same notion of "staleness" as the memory layer.
- Traditional graph libraries like NetworkX are too slow for temporal, point-in-time traversals. If every hop has to check a validity window, doing that in Python per edge adds up fast. If you skip it and filter at the end, the traversal has already walked through superseded facts and pulled in neighbours it should never have reached.
- Existing vector databases lose graph structure. Cosine similarity cannot tell you which of two near-identical facts was true in March, and it cannot follow a three-hop chain. In my benchmarks, exact FAISS returned stale facts more than half the time.

The engineering

- Readers never take a lock. The graph is published as an immutable snapshot behind an ArcSwap. Writers group-commit a batch to a WAL, apply it to a copy-on-write paged arena, and publish the new snapshot with a single atomic pointer swap.
- Temporal validity is checked inside the traversal loop. A hyperedge whose [valid_from, valid_to) window does not contain the query's as_of time is neither returned nor expanded.
- Ranking uses a SIMD exponential time decay (8 lanes via wide::f32x8, AVX on x86_64 and NEON on aarch64).
- The PyO3 bindings release the GIL on every call and take embeddings through the buffer protocol. The KV offloader copies straight between a tensor's memory (via data_ptr()) and 64-byte-aligned append-only mmap segments, with no intermediate Python buffers.

The numbers

- Ingestion: 10 million hyperedges in 85.6 s (~117K edges/sec sustained, 227K/s peak) on a 4-core CPU, peaking at just 3.2 GiB of RAM. This run was on a Kaggle instance's CPU, because my 8 GiB laptop starts paging around 3M edges.
- Latency: an 8x to 9.5x median latency advantage over NetworkX for 3-hop queries (100K entities, 500K facts, 1,000 questions; 2.95 ms p50 against 23.7 ms for temporally filtered NetworkX and 27.8 ms for naive BFS), with 0% stale facts.
- Soak test: 500 vLLM generations of Qwen2.5-1.5B on a T4, with a 32-block offload/restore every 10 generations. All 1,600 blocks came back bit-exact, VRAM grew by 0 B, and anonymous heap stayed flat (about 0.2 MiB of growth). Resident memory does grow with file-backed pages that track the KV data appended to the segments, which is why file-backed segment compaction is my immediate roadmap priority for v2.1.0.

What is not done yet, so you do not have to find it: the offloader is not yet registered as a vLLM KV connector (you call it from your own serving code), the benchmark data is synthetic, and the read p99 is worse than flat vector search because of snapshot reclamation on the read path. All of this, plus the full methodology, is in the README and the whitepaper in the repo.

The website

The site has a bespoke "Transit Diagram" UI. Facts are drawn as lines and stations like a transit map, and you can scrub a timeline to watch a point-in-time traversal change: superseded edges drop out, and the path the query is allowed to take redraws as of the moment you pick.

The ask

- Try the SDKs: pip install hypergraph-client (or hypergraph-engine-native for in-process use), and npm install @temporal-hypergraph/sdk for TypeScript. A prebuilt server is one docker run away.
- Review the Rust core, especially the ArcSwap snapshot publishing in engine-core and the WAL recovery logic. That is where I would most like a second pair of eyes.
- Tell me what you think of the architecture. If a benchmark baseline looks unfair, or you know a workload where this design falls over, I would rather hear it now.

License note: v2.0.0 is under SSPL-1.0. You can use, modify and self-host it for free; the restriction is on offering it as a competing managed database service. Versions 1.0.1 and earlier remain MIT OR Apache-2.0.

Thanks for reading. I will be in the comments.
