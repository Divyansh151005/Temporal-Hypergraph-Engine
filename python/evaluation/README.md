# hypergraph-evaluation

Synthetic data generation, LLM graph extraction and an F1 harness for the Temporal Hypergraph Engine.

```bash
pip install -e python/evaluation
```

## LLM router

Keys are read only from the environment. Copy `.env.example` to `.env` (git-ignored) and load it with `set -a; source .env; set +a`.

All providers go through the OpenAI SDK and request strict `json_schema` output. A model that rejects `json_schema` is switched to JSON mode with the schema in the system prompt.

| Provider | Base URL | Key |
|---|---|---|
| `xai` (Grok) | `https://api.x.ai/v1` | `XAI_API_KEY` (starts with `xai-`) |
| `groq` | `https://api.groq.com/openai/v1` | `GROQ_API_KEY` (starts with `gsk_`) |
| `openai` | `LLM_BASE_URL` (default OpenAI); also Ollama and vLLM | `LLM_API_KEY` (or `OPENAI_API_KEY`) |

Each pipeline step (role) has a provider chain of `provider:model` entries, tried in order. Override a chain with its environment variable or the matching CLI flag:

| Role | Variable / flag | Default chain |
|---|---|---|
| Article writing | `GENERATOR_LLM`, `synthetic_generator.py --llm` | `xai:grok-4.7,groq:qwen/qwen3.8-27b` |
| Graph extraction | `EXTRACT_LLM`, `extract_graph.py --llm` | `xai:grok-4.7,groq:qwen/qwen3.8-27b` |
| Benchmark answers | `ANSWER_LLM`, `run_benchmark.py --answer-llm` | `groq:qwen/qwen3.8-27b,xai:grok-4.7` |
| Benchmark judge | `JUDGE_LLM`, `run_benchmark.py --judge-llm` | `groq:qwen/qwen3.8-27b,xai:grok-4.7` |

Grok handles the reasoning-heavy steps (writing articles and normalizing temporal boundaries during extraction). Groq's Qwen answers and judges, with Grok as the fallback. Because the judge and answerer are the same model, the judge may be lenient toward the answerer's phrasing; set `JUDGE_LLM=xai:grok-4.7` for an independent judge. Entries whose key isn't set are skipped with a warning. `LLM_PROVIDER` (plus optional `LLM_MODEL`) forces a single provider for every role.

Failover:

- **When a provider fails over:**
  - a 429 whose retry delay is over 10 seconds, or that mentions quota or credits;
  - a 503 (overloaded);
  - rejected credentials or an unknown model;
  - an exhausted retry budget.
- **Circuit breaker:** that provider is skipped for the server's retry delay (60 seconds by default), or for the rest of the run after an auth or model error. The request moves to the next provider in the chain.
- **Unsupported sampling parameters:** when a model rejects `temperature` or `seed` (reasoning models), they are dropped for that model and the request is retried.
- **Output budget:** every request sends `max_tokens` (8192).
- **Rate limits:** a short 429 is waited out on the same provider. Retry delays are parsed from "retry in Ns" and Groq's "try again in 1m2.5s".
- **Invalid output:** invalid JSON or schema violations are retried on the same provider with the validation errors fed back.
- **Single-provider chains** keep waiting and retrying instead of failing over.

Every output is validated against the Draft-07 schemas in `schema/`. Results are cached per provider and model in `LLM_CACHE_DIR` (default `.cache/llm`; empty disables the cache). The model that actually served each call is recorded:

- `generator.writer` in dataset records;
- `model` in extraction predictions;
- `config.llm` in benchmark results.

```bash
# Local models through the openai provider
LLM_PROVIDER=openai LLM_BASE_URL=http://localhost:11434/v1 LLM_MODEL=qwen2.5:7b ...    # Ollama
LLM_PROVIDER=openai LLM_BASE_URL=http://localhost:8000/v1 LLM_MODEL=Qwen/Qwen2.5-7B-Instruct ...   # vLLM
```

## Generate

```bash
python python/evaluation/synthetic_generator.py --n 1000 --workers 8 --noise 0.1 --out datasets/synthetic/corp.jsonl
python python/evaluation/synthetic_generator.py --n 100 --golden --out datasets/core_100_golden_dataset.json
python python/evaluation/synthetic_generator.py --n 200 --offline --out /tmp/offline.jsonl   # templates, no LLM
```

Generation runs in reverse: a seeded ground-truth hypergraph comes first, and the article is written from it.

- **Domains:** `corporate_ma`, `executive_roles`, `customer_support`, `medical_care`.
- **Fact supersession:** each graph contains a 2–3 hop chain that ends in a superseded fact, so "as of" questions have both a correct answer and a stale distractor.
- **Entity mentions:** articles must mention every entity, or they are regenerated.
- **Unique labels:** entity labels are unique across a dataset.
- **Record contents:** `raw_text`, `ground_truth_graph` (ISO-normalized epoch bounds) and `qa` with gold edge ids, seeds, hops and the stale answer.

## Extract and score

```bash
python python/evaluation/extract_graph.py --in datasets/core_100_golden_dataset.json --out results/predictions.jsonl
python python/evaluation/evaluate_f1.py --pred results/predictions.jsonl \
  --truth datasets/core_100_golden_dataset.json --out-csv results/f1.csv --out-json results/f1.json
```

`evaluate_f1.py` works in two steps:

1. **Node alignment:** nodes are matched by label and alias similarity (character TF-IDF by default, or `--embedder openai`) with Hungarian assignment.
2. **Hyperedge matching:** hyperedges are matched with `scipy.optimize.linear_sum_assignment`, scored on `0.3·semantic + 0.5·participant Jaccard + 0.2·temporal IoU`.

It reports:

- node, edge and participant precision/recall/F1;
- participant Jaccard;
- temporal boundary accuracy: half-open interval IoU, with unknown bounds clamped to the corpus horizon and point events matched within `--point-tolerance-days`.

Edge scores depend on the extractor. Smaller open models often leave `valid_from`/`valid_to` null and keep month-precision dates ("from May 2013 until July 2018") only in `raw_temporal_expression`. They also paraphrase event types (`MEDICATION_PRESCRIPTION` for `PRESCRIPTION`), which lowers edge F1 even when participants match exactly. Check `model` in the predictions before comparing runs: documents that failed over from Grok to Groq score lower.

Gates for CI:

- `--baseline prev.json --max-drop 0.02` exits 1 on a regression.
- `--min-recall 0.80` exits 1 when node or edge recall falls below the threshold.

## CI regression gate

`.github/workflows/evaluation.yml` scores committed predictions when these files exist:

- `datasets/core_100_golden_dataset.json`
- `datasets/predictions/core_100_predictions.jsonl`
- `datasets/predictions/core_100_f1.json` (optional baseline)

Generating them needs an LLM:

```bash
python python/evaluation/synthetic_generator.py --n 100 --golden --out datasets/core_100_golden_dataset.json
python python/evaluation/extract_graph.py --in datasets/core_100_golden_dataset.json --out datasets/predictions/core_100_predictions.jsonl
python python/evaluation/evaluate_f1.py --pred datasets/predictions/core_100_predictions.jsonl \
  --truth datasets/core_100_golden_dataset.json --out-json datasets/predictions/core_100_f1.json
```

## Tests

```bash
pytest tests/python/evaluation tests/python/benchmarks
```

The tests need no LLM; the LLM paths are exercised with a fake OpenAI client.
