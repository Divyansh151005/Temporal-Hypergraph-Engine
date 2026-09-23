"""LangChain adapters against a real ``hypergraph-server``."""

import uuid

from hypergraph_client import HypergraphClient
from hypergraph_client.server import LocalServer
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_hypergraph import TemporalHypergraphMemory, TemporalHypergraphVectorStore


def unique(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def test_one_line_memory(hypergraph_server: LocalServer, monkeypatch):
    monkeypatch.setenv("HYPERGRAPH_ENDPOINT", hypergraph_server.endpoint)
    memory = TemporalHypergraphMemory(api_key=None, session_id=unique("s"))
    memory.add_messages(
        [
            HumanMessage("my card ending 4242 was charged twice", additional_kwargs={"entities": ["card:4242"]}),
            AIMessage("I refunded the duplicate charge"),
        ]
    )
    assert [m.content for m in memory.messages] == [
        "my card ending 4242 was charged twice",
        "I refunded the duplicate charge",
    ]
    assert memory.messages[0].additional_kwargs == {"entities": ["card:4242"]}
    assert [e.summary for e in memory.relevant_context("anything", k=5)]
    memory.clear()
    assert memory.messages == []


async def test_async_memory(hypergraph_server: LocalServer):
    memory = TemporalHypergraphMemory(unique("s"), endpoint=hypergraph_server.endpoint)
    await memory.aadd_messages([HumanMessage("a"), AIMessage("b")])
    assert [m.content for m in await memory.aget_messages()] == ["a", "b"]
    await memory.aclear()
    assert await memory.aget_messages() == []


def test_runnable_with_message_history(hypergraph_server: LocalServer):
    seen: list[int] = []

    def record_prompt(prompt):
        seen.append(len(prompt.to_messages()))
        return prompt

    prompt = ChatPromptTemplate.from_messages(
        [MessagesPlaceholder("history"), ("human", "{input}")]
    )
    chain = prompt | RunnableLambda(record_prompt) | FakeListChatModel(responses=["first", "second"])
    chat = RunnableWithMessageHistory(
        chain,
        lambda session_id: TemporalHypergraphMemory(session_id, endpoint=hypergraph_server.endpoint),
        input_messages_key="input",
        history_messages_key="history",
    )
    session = unique("s")
    config = {"configurable": {"session_id": session}}
    assert chat.invoke({"input": "hello"}, config).content == "first"
    assert chat.invoke({"input": "again"}, config).content == "second"
    assert seen == [1, 3]

    stored = TemporalHypergraphMemory(session, endpoint=hypergraph_server.endpoint).messages
    assert [(m.type, m.content) for m in stored] == [
        ("human", "hello"),
        ("ai", "first"),
        ("human", "again"),
        ("ai", "second"),
    ]


def test_vector_store_multi_hop_and_time_travel(hypergraph_server: LocalServer, embeddings):
    client = HypergraphClient(hypergraph_server.endpoint)
    store = TemporalHypergraphVectorStore(embeddings, client)
    acme, beta, ceo = unique("org"), unique("org"), unique("person")
    old_id, acq_id, new_id = store.add_texts(
        [
            "Alice was CEO of Beta",
            "Acme acquired Beta",
            "Bob became CEO of Beta",
        ],
        [
            {"entities": [beta, ceo], "valid_from": 100, "valid_to": 500, "event_type": "ROLE"},
            {"entities": [acme, beta], "valid_from": 300, "event_type": "ACQUISITION"},
            {"entities": [beta, unique("person")], "valid_from": 500, "event_type": "ROLE2"},
        ],
    )

    one_hop = store.similarity_search("CEO", k=10, seed_entities=[acme], as_of=400)
    assert [d.id for d in one_hop] == [acq_id]
    two_hop = store.similarity_search("CEO", k=10, seed_entities=[acme], k_hops=2, as_of=400)
    assert {d.id for d in two_hop} == {acq_id, old_id}
    later = store.similarity_search("CEO", k=10, seed_entities=[acme], k_hops=2, as_of=600)
    assert {d.id for d in later} == {acq_id, new_id}

    scored = store.similarity_search_with_relevance_scores("Bob CEO", k=1, as_of=600)
    assert scored[0][0].id == new_id and 0 < scored[0][1] <= 1
    assert store.delete([new_id])
    client.close()
