"""LlamaIndex adapters against a real ``hypergraph-server``."""

import uuid

from hypergraph_client import Hyperedge, HypergraphClient, record_node
from hypergraph_client.server import LocalServer
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.llms import MockLLM
from llama_index.core.query_engine import RetrieverQueryEngine

from llama_index_hypergraph import TemporalHypergraphChatMemory, TemporalHypergraphRetriever

try:
    from langchain_hypergraph import TemporalHypergraphMemory
except ImportError:  # the cross-framework check is optional
    TemporalHypergraphMemory = None


def unique(prefix: str) -> str:
    return f"{prefix}:{uuid.uuid4().hex[:8]}"


def test_retriever_query_engine_time_travel(hypergraph_server: LocalServer):
    client = HypergraphClient(hypergraph_server.endpoint)
    beta, acme = unique("org"), unique("org")
    client.insert(Hyperedge("CEO", [beta, unique("person"), record_node()], 100, 500,
                            summary="Alice was CEO of Beta"))
    client.insert(Hyperedge("CEO", [beta, unique("person"), record_node()], 500,
                            summary="Bob is CEO of Beta"))
    client.insert(Hyperedge("ACQUIRED", [acme, beta, record_node()], 50,
                            summary="Acme acquired Beta"))

    retriever = TemporalHypergraphRetriever(client, seed_entities=[acme], k_hops=2, as_of=300)
    engine = RetrieverQueryEngine.from_args(retriever, llm=MockLLM())
    response = engine.query("Who runs the company Acme bought?")
    assert {n.node.get_content() for n in response.source_nodes} == {
        "Acme acquired Beta",
        "Alice was CEO of Beta",
    }

    retriever.as_of = 600
    assert "Bob is CEO of Beta" in {n.node.get_content() for n in retriever.retrieve("who?")}
    client.close()


async def test_async_retrieve(hypergraph_server: LocalServer):
    client = HypergraphClient(hypergraph_server.endpoint)
    node = unique("n")
    client.insert(Hyperedge("E", [node, record_node()], 5, summary="hello"))
    retriever = TemporalHypergraphRetriever(client, seed_entities=[node], as_of=10)
    assert [n.node.get_content() for n in await retriever.aretrieve("x")] == ["hello"]
    client.close()


async def test_memory_shared_with_langchain_scheme(hypergraph_server: LocalServer):
    session = uuid.uuid4().hex[:8]
    memory = TemporalHypergraphChatMemory.from_defaults(session, endpoint=hypergraph_server.endpoint)
    memory.put(ChatMessage(role=MessageRole.USER, content="hi"))
    await memory.aput(ChatMessage(role=MessageRole.ASSISTANT, content="hello"))
    assert [(m.role, m.content) for m in memory.get()] == [
        (MessageRole.USER, "hi"),
        (MessageRole.ASSISTANT, "hello"),
    ]
    assert [m.content for m in await memory.aget_all()] == ["hi", "hello"]

    if TemporalHypergraphMemory is not None:
        lc = TemporalHypergraphMemory(session, endpoint=hypergraph_server.endpoint)
        assert [m.content for m in lc.messages] == ["hi", "hello"]

    memory.reset()
    assert memory.get_all() == []
