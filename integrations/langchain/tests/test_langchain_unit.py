"""LangChain adapters against the in-memory fake client."""

import pytest
from hypergraph_client import ChatRecord, Role
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from langchain_hypergraph import (
    TemporalHypergraph,
    TemporalHypergraphMemory,
    TemporalHypergraphVectorStore,
)
from langchain_hypergraph.messages import message_text, to_message, to_record


def test_message_conversion_round_trips():
    human = HumanMessage("hi", additional_kwargs={"entities": ["person:ada"]})
    record = to_record(human)
    assert record == ChatRecord(Role.HUMAN, "hi", entities=("person:ada",))
    back = to_message(record)
    assert isinstance(back, HumanMessage) and back.additional_kwargs == {"entities": ["person:ada"]}

    for message, cls in [
        (AIMessage("a"), AIMessage),
        (SystemMessage("s"), SystemMessage),
        (ToolMessage("t", tool_call_id="call-1"), ToolMessage),
    ]:
        assert isinstance(to_message(to_record(message)), cls)

    blocks = HumanMessage(content=[{"type": "text", "text": "a"}, {"type": "image_url"}, "b"])
    assert message_text(blocks) == "ab"


def test_memory_add_read_clear(fake_client):
    memory = TemporalHypergraphMemory("s1", client=fake_client)
    other = TemporalHypergraphMemory("s2", client=fake_client)
    memory.add_messages([HumanMessage("q1"), AIMessage("a1")])
    memory.add_user_message("q2")
    other.add_ai_message("elsewhere")

    assert [(m.type, m.content) for m in memory.messages] == [
        ("human", "q1"),
        ("ai", "a1"),
        ("human", "q2"),
    ]
    assert all(m.id for m in memory.messages)
    limited = TemporalHypergraphMemory("s1", client=fake_client, history_limit=2)
    assert [m.content for m in limited.messages] == ["a1", "q2"]

    memory.clear()
    assert memory.messages == []
    assert [m.content for m in other.messages] == ["elsewhere"]


def test_vector_store_add_search_delete(fake_client, embeddings):
    store = TemporalHypergraphVectorStore(embeddings, fake_client)
    ids = store.add_texts(
        ["the cat sat on the mat", "quarterly revenue grew"],
        [{"entities": ["animal:cat"], "valid_from": 100}, {"valid_from": 100, "valid_to": 200}],
    )
    assert len(ids) == 2

    [(doc, score)] = store.similarity_search_with_score("cat on a mat", k=1, as_of=150)
    assert doc.page_content == "the cat sat on the mat" and doc.id == ids[0]
    assert doc.metadata["entities"] == ["animal:cat"] and doc.metadata["valid_from"] == 100
    assert score > 0

    assert [d.id for d in store.similarity_search("revenue", k=5, as_of=250)] == [ids[0]]
    seeded = store.similarity_search("anything", k=5, as_of=150, seed_entities=["animal:cat"])
    assert [d.id for d in seeded] == [ids[0]]

    assert store.delete([ids[0]])
    assert store.delete([]) is None
    assert [d.id for d in store.similarity_search("cat", k=5, as_of=150)] == [ids[1]]


def test_vector_store_session_scope_and_retriever(fake_client, embeddings):
    store = TemporalHypergraphVectorStore.from_texts(
        ["alpha notes", "beta notes"], embeddings, client=fake_client, session_id="a"
    )
    store.add_texts(["gamma notes"], session_id="b")
    assert {d.page_content for d in store.similarity_search("notes", k=5)} == {
        "alpha notes",
        "beta notes",
    }
    assert [d.page_content for d in store.similarity_search("notes", k=5, session_id="b")] == [
        "gamma notes"
    ]
    retriever = store.as_retriever(search_kwargs={"k": 1})
    assert retriever.invoke("alpha")[0].page_content == "alpha notes"


def test_metadata_length_mismatch_rejected(fake_client, embeddings):
    store = TemporalHypergraphVectorStore(embeddings, fake_client)
    with pytest.raises(ValueError):
        store.add_texts(["a", "b"], [{}])


def test_temporal_hypergraph_one_liner(fake_client):
    memory = TemporalHypergraph(api_key="k", client=fake_client)
    memory.add_user_message("hello")
    assert memory.session_id == "default"
    assert [m.content for m in TemporalHypergraphMemory("default", client=fake_client).messages] == ["hello"]
