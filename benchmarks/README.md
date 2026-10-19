# Benchmarks: hypergraph traversal vs vector RAG

Head-to-head on synthetic multi-hop temporal QA (`python/evaluation`). Both contenders use the same embedder.

- **Vector RAG baseline** (`baseline_vector.py`): articles are split into sentence-packed chunks of up to 128 tokens. They are indexed in `QdrantClient(":memory:")` (or `--baseline-backend numpy`), and the top-k chunks become the context.
- **Hypergraph** (`hypergraph_runner.py`): graphs are ingested through the in-process PyO3 engine, gold by default or from `--graphs` for extracted ones. For each question:
  - Entity names and aliases in the question are linked to seed labels.
  - `traverse(seeds, k_hops, as_of)` runs with an embedding re-rank.
  - Each fact is rendered as `[from → to] EVENT(participants): summary`.

## Run

```bash
pip install -r benchmarks/requirements.txt
(cd crates/pyo3-bindings && maturin develop --release)

# No API key: template articles, LLM-free proxies only
python benchmarks/run_benchmark.py --offline-corpus --skip-llm

# Full run on an LLM-written dataset, also timing gRPC over a Unix socket
cargo build --release -p grpc-server
python benchmarks/run_benchmark.py --dataset datasets/synthetic/corp.jsonl --hops 3 --grpc
```

The answer and judge steps use separate provider chains from the router in `python/evaluation/llm.py`:

- **Answerer:** `ANSWER_LLM` or `--answer-llm`; default Groq `qwen/qwen3.8-27b`, then xAI `grok-4.7`.
- **Judge:** `JUDGE_LLM` or `--judge-llm`; same default chain as the answerer, so the judge grades its own model family. Pass `--judge-llm xai:grok-4.7` for an independent judge. The judge only sees answers that don't match the gold or stale entity by name, so the judgement count is usually far below the answer count.

Keys come from `GROQ_API_KEY` and `XAI_API_KEY`. A model that hits its rate limit or quota fails over to the next one in its chain. `results.json` (`config.llm`) and `summary.md` record which models actually served the answers and judgements. Pass `--assert-targets` to exit nonzero when a target is missed. Each run writes `results.json`, `results.csv` (one row per question per contender) and `summary.md` to `results/<timestamp>/`.

## Metrics

- **Context token reduction:** `1 - tokens_hg / tokens_baseline`, counted with tiktoken `o200k_base`.
  - Reported at a fixed baseline k (`--k`, default 5).
  - Also reported at a per-question *recall-matched* k: the smallest k whose chunks cover as many gold facts as the hypergraph context does.
- **Gold-fact coverage:** the share of a question's gold hyperedges whose participants are all named in the context. The same text check is applied to both contenders.
- **Stale-fact contamination:** the share of contexts that name the superseded answer, such as the former CEO.
- **Hallucination rate:** the LLM answers from each context, or says `UNKNOWN`.
  - Answers are graded by normalized name match, with an LLM judge as a fallback.
  - Stale and wrong answers both count as hallucinations.
  - Reduction is `(H_base - H_hg) / H_base`.
- **Latency:** p50, p95 and p99 of the retrieval call after warmup (`perf_counter_ns`), measured for in-process PyO3, gRPC over UDS (`--grpc`) and the vector baseline. Query embedding happens outside the timed region for all of them. Criterion means from `cargo bench -p engine-core --bench traversal` are quoted when present.

## Reference run (offline corpus, `--skip-llm --grpc`, Apple Silicon)

200 documents, 400 questions, 3 hops, k=5:

| Metric | Hypergraph | Vector RAG k=5 | Vector RAG recall-matched |
|---|---|---|---|
| Mean context tokens | 252 | 551 | 1051 |
| Gold-fact coverage | 96.0% | 53.1% | 74.7% |
| Stale-fact contamination | 0.0% | 41.8% | 64.8% |

The hypergraph context uses 54% fewer tokens than the fixed-k baseline and 76% fewer than the recall-matched baseline. The ~80% target is not met on template articles; LLM-written articles are longer, so expect a larger gap there.

3-hop latency:

| Path | p50 | p95 | p99 |
|---|---|---|---|
| In-process | 0.017 ms | 0.026 ms | 0.029 ms |
| gRPC over UDS | 0.17 ms | 0.20 ms | 0.23 ms |
| Qdrant in-memory | 0.18 ms | 1.96 ms | 5.0 ms |

Both hypergraph paths are well under the 2 ms target.

At 1000 documents with noise 0.1 (`--offline --noise 0.1`), hypergraph coverage drops to 77.9%, against 33.4% for the baseline. Token reduction is 52% at fixed k and 81% recall-matched. Stale contamination is 0.1% for the hypergraph and 22.6% for the baseline. In-process p95 is 0.055 ms.

Coverage misses come from hub entities shared across documents (drugs, cities): 3 hops through a hub pull in unrelated facts, and the hashing-embedder re-rank sometimes ranks the gold fact below `--top-k`. A semantic embedder narrows this.
