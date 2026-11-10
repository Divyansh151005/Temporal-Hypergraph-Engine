# llama-index-hypergraph

LlamaIndex retriever and chat memory backed by the Temporal Hypergraph Engine.

```bash
pip install llama-index-hypergraph
docker run -d -p 50051:50051 ghcr.io/divyansh151005/temporal-hypergraph-server
```

For development from a clone: `pip install -e integrations/python-client -e integrations/llamaindex`.

```python
from llama_index_hypergraph import TemporalHypergraph

memory = TemporalHypergraph(api_key="your_key")  # a TemporalHypergraphChatMemory on session "default"
memory = TemporalHypergraph(api_key="your_key", session_id="user-42")
```

The server address comes from `endpoint=` or `$HYPERGRAPH_ENDPOINT` (default
`localhost:50051`).

## Retriever

```python
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index_hypergraph import TemporalHypergraphRetriever

retriever = TemporalHypergraphRetriever(
    known_entities=["org:acme", "org:beta"],  # default linker: label match in the query
    k_hops=3,
    as_of=1700000000,                        # facts valid at this instant (default now)
)
engine = RetrieverQueryEngine.from_args(retriever, llm=llm)
```

- Seeds come from `entity_linker(query)` (pass your own callable for NER or
  fuzzy matching), plus `seed_entities` and `session:<session_id>`.
- `embed_model` (a LlamaIndex `BaseEmbedding` or LangChain-style embedder)
  re-ranks results by similarity; with no seeds the retriever falls back to
  global similarity search.
- Each `TextNode` carries `event_type`, `entities`, `valid_from`, `valid_to`
  and `hyperedge_id` metadata.

## Chat memory

`TemporalHypergraphChatMemory` is a `BaseMemory`. `get(input=...)` returns the
recent history (`history_limit=`), preceded by a system message listing
relevant facts reached within `k_hops` of the session. It uses the same
storage scheme as `langchain-hypergraph`, so either framework can read a
session written by the other.

## Tests

```bash
cargo build --release -p grpc-server   # for the live tests; they skip otherwise
pytest integrations/llamaindex
```
