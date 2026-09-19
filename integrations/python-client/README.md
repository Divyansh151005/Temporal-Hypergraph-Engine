# hypergraph-client

gRPC client for the Temporal Hypergraph Engine. The LangChain and LlamaIndex
integrations are built on it.

```bash
pip install -e integrations/python-client
```

```python
from hypergraph_client import TemporalHypergraph

memory = TemporalHypergraph(api_key="your_key")  # session "default"; pass session_id= to change
```

```python
from hypergraph_client import HypergraphClient, Hyperedge

with HypergraphClient("localhost:50051", api_key=None) as client:
    client.insert(Hyperedge("ACQUISITION", ["org:microsoft", "org:activision"], 1696118400,
                            summary="Microsoft completed its Activision acquisition"))
    for edge in client.traverse(["org:microsoft"], k_hops=3):
        print(edge.score, edge.summary)
```

- `endpoint` defaults to `$HYPERGRAPH_ENDPOINT`, then `localhost:50051`. It
  accepts `host:port`, `tcp://host:port` and `unix:///path`.
- `api_key` defaults to `$HYPERGRAPH_API_KEY` and is sent as `x-api-key`.
- `AsyncHypergraphClient` offers the same methods for `asyncio`.
- `HypergraphKVOffloadTierManager` is a vLLM secondary KV tier over gRPC
  (`submit_store`, `submit_load`, `close`); point vLLM's
  `secondary_tier_module` at `hypergraph_client.kv_tier.HypergraphKVOffloadTierManager`
  with `{"grpc_endpoint": "localhost:50051"}`.
- `SessionMemory` stores chat messages around a `session:<id>` node; see the
  `hypergraph_client.session` module for the graph conventions.
- `HashingEmbedder` is a deterministic, dependency-free default embedder. Use a
  real embedding model for semantic retrieval quality.

Stubs in `src/hypergraph_client/_proto/` are generated; regenerate them with
`scripts/gen_protos.sh python` after editing `proto/hypergraph_service.proto`.

Tests (they start a local server; build it first with
`cargo build --release -p grpc-server`):

```bash
pytest integrations/python-client
```
