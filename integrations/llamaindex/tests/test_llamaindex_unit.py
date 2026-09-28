"""LlamaIndex adapters against the in-memory fake client."""

import pytest
from hypergraph_client import HashingEmbedder, Hyperedge, Role, record_node
from hypergraph_client.testing import FakeHypergraphClient
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.schema import QueryBundle

from llama_index_hypergraph import (
    CONTEXT_HEADER,
    TemporalHypergraph,
    TemporalHypergraphChatMemory,
    TemporalHypergraphRetriever,
    exact_label_linker,
)
from llama_index_hypergraph.memory import from_role, to_role


@pytest.fixture
def client() -> FakeHypergraphClient:
    return FakeHypergraphClient()


def test_roles_cover_every_message_role():
    for role in MessageRole:
        assert from_role(to_role(role)) in MessageRole
    assert {from_role(r) for r in Role} == {
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.SYSTEM,
        MessageRole.TOOL,
    }


def test_exact_label_linker():
    link = exact_label_linker(["org:acme_corp", "org:beta", "person:al"])
    assert link("Who bought Acme Corp?") == ["org:acme_corp"]
    assert link("beta and alpha") == ["org:beta"]


def test_retriever_links_entities_and_hops(client):
    embedder = HashingEmbedder(32)
    for pair, text in [
        (("org:acme", "org:beta"), "Acme acquired Beta"),
        (("org:beta", "person:bob"), "Bob runs Beta"),
        (("person:bob", "city:oslo"), "Bob lives in Oslo"),
    ]:
        client.insert(
            Hyperedge("E", [*pair, record_node()], 10, summary=text, embedding=embedder.embed(text))
        )
    retriever = TemporalHypergraphRetriever(
        client, known_entities=["org:acme", "org:beta"], k_hops=2, as_of=20
    )
    texts = [n.node.get_content() for n in retriever.retrieve("What did Acme do?")]
    assert set(texts) == {"Acme acquired Beta", "Bob runs Beta"}
    node = retriever.retrieve("Acme")[0]
    assert node.node.metadata["entities"] and node.node.metadata["valid_from"] == 10
    assert "hyperedge_id" not in node.node.get_content(metadata_mode="llm")

    assert retriever.retrieve("nothing known") == []
    ranked = TemporalHypergraphRetriever(client, embed_model=embedder, as_of=20, top_k=1)
    assert [n.node.get_content() for n in ranked.retrieve(QueryBundle("Bob lives in Oslo"))] == [
        "Bob lives in Oslo"
    ]


def test_memory_put_get_set_reset(client):
    memory = TemporalHypergraphChatMemory.from_defaults("s1", client=client, history_limit=2)
    memory.put(ChatMessage(role=MessageRole.USER, content="my order 77 is late",
                           additional_kwargs={"entities": ["order:77"]}))
    memory.put_messages([
        ChatMessage(role=MessageRole.ASSISTANT, content="it ships tomorrow"),
        ChatMessage(role=MessageRole.USER, content="thanks"),
    ])
    assert [m.content for m in memory.get_all()] == ["my order 77 is late", "it ships tomorrow", "thanks"]
    assert memory.get_all()[0].additional_kwargs["entities"] == ["order:77"]
    assert [m.content for m in memory.get()] == ["it ships tomorrow", "thanks"]

    with_context = memory.get(input="order 77")
    assert with_context[0].role == MessageRole.SYSTEM
    assert with_context[0].content.startswith(CONTEXT_HEADER)
    assert "my order 77 is late" in with_context[0].content
    assert "thanks" not in with_context[0].content

    memory.set([ChatMessage(role=MessageRole.USER, content="fresh start")])
    assert [m.content for m in memory.get_all()] == ["fresh start"]
    memory.reset()
    assert memory.get_all() == []


def test_temporal_hypergraph_one_liner(client):
    memory = TemporalHypergraph(api_key="k", client=client)
    assert isinstance(memory, TemporalHypergraphChatMemory) and memory.session_id == "default"
    memory.put(ChatMessage(role=MessageRole.USER, content="hello"))
    assert [m.content for m in memory.get_all()] == ["hello"]
