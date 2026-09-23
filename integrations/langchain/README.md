# langchain-hypergraph

LangChain chat memory and vector store backed by the Temporal Hypergraph Engine.

```bash
pip install -e integrations/python-client -e integrations/langchain
```

```python
from langchain_hypergraph import TemporalHypergraph

memory = TemporalHypergraph(api_key="your_key")  # a TemporalHypergraphMemory on session "default"
memory = TemporalHypergraph(api_key="your_key", session_id="user-42")
```

The server address comes from `endpoint=` or `$HYPERGRAPH_ENDPOINT` (default
`localhost:50051`).

## Chat memory

`TemporalHypergraphMemory` is a `BaseChatMessageHistory`, so it plugs into
`RunnableWithMessageHistory`:

```python
from langchain_core.runnables.history import RunnableWithMessageHistory

chat = RunnableWithMessageHistory(
    prompt | llm,
    lambda session_id: TemporalHypergraphMemory(session_id=session_id),
    input_messages_key="input",
    history_messages_key="history",
)
```

- Each message becomes one hyperedge connecting the `session:<id>` node, a
  per-message record node, and any ids in `additional_kwargs["entities"]`.
- `messages` returns the session oldest first (`history_limit=` keeps the newest N).
- `relevant_context(query, k, k_hops=2)` returns the most relevant facts, also
  reaching those linked through the messages' entities. Pass `embeddings=` for
  similarity ranking.
- `ToolMessage.tool_call_id` and other message fields besides the text and
  entities are not stored.

## Vector store

```python
from langchain_hypergraph import TemporalHypergraphVectorStore

store = TemporalHypergraphVectorStore(embeddings)
store.add_texts(
    ["Acme acquired Beta"],
    [{"entities": ["org:acme", "org:beta"], "valid_from": 1696118400, "event_type": "ACQUISITION"}],
)
store.similarity_search("who owns Beta?", k=4, as_of=1700000000)
store.similarity_search("Beta CEO", seed_entities=["org:acme"], k_hops=3)
```

- Recognized metadata: `entities`, `valid_from`, `valid_to` (epoch seconds),
  `event_type`, `confidence`. **Other metadata is dropped**; the engine stores
  only the text and the graph structure.
- Ids are assigned by the engine and returned by `add_texts`.
- Search keywords: `as_of`, `session_id`, `seed_entities`, `k_hops`, `decay_lambda`.
- The embedding dimension must match the server's `--embedding-dim`.

## Tests

```bash
cargo build --release -p grpc-server   # for the live tests; they skip otherwise
pytest integrations/langchain
```
