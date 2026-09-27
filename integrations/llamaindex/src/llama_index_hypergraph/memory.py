"""``BaseMemory`` backed by a hypergraph session node.

Uses the same storage scheme as ``langchain-hypergraph``, so one session can
be read and written from either framework.
"""

from __future__ import annotations

from typing import Any

from hypergraph_client import (
    DEFAULT_SESSION,
    ChatRecord,
    Embedder,
    HypergraphClient,
    Role,
    SessionMemory,
)
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.memory.types import BaseMemory
from pydantic import PrivateAttr
from typing_extensions import assert_never

from .embedding import LlamaEmbedding, as_embedder

ENTITIES_KWARG = "entities"
CONTEXT_HEADER = "Relevant facts from long-term memory:"


def to_role(role: MessageRole) -> Role:
    match role:
        case MessageRole.USER:
            return Role.HUMAN
        case MessageRole.ASSISTANT | MessageRole.CHATBOT | MessageRole.MODEL:
            return Role.AI
        case MessageRole.SYSTEM | MessageRole.DEVELOPER:
            return Role.SYSTEM
        case MessageRole.TOOL | MessageRole.FUNCTION:
            return Role.TOOL
        case _:
            assert_never(role)


def from_role(role: Role) -> MessageRole:
    match role:
        case Role.HUMAN:
            return MessageRole.USER
        case Role.AI:
            return MessageRole.ASSISTANT
        case Role.SYSTEM:
            return MessageRole.SYSTEM
        case Role.TOOL:
            return MessageRole.TOOL
        case _:
            assert_never(role)


def to_record(message: ChatMessage) -> ChatRecord:
    entities = message.additional_kwargs.get(ENTITIES_KWARG) or ()
    return ChatRecord(
        to_role(message.role), message.content or "", entities=tuple(str(e) for e in entities)
    )


def to_message(record: ChatRecord) -> ChatMessage:
    kwargs: dict[str, Any] = {"hyperedge_id": record.id}
    if record.entities:
        kwargs[ENTITIES_KWARG] = list(record.entities)
    return ChatMessage(role=from_role(record.role), content=record.content, additional_kwargs=kwargs)


class TemporalHypergraphChatMemory(BaseMemory):
    """Chat memory of one session, stored in the Temporal Hypergraph Engine.

    ``get(input=...)`` returns the recent history, preceded by a system message
    listing older or entity-linked facts relevant to ``input``.
    """

    session_id: str
    history_limit: int = 0
    context_k: int = 4
    k_hops: int = 2

    _store: SessionMemory = PrivateAttr()

    @classmethod
    def class_name(cls) -> str:
        return "TemporalHypergraphChatMemory"

    @classmethod
    def from_defaults(
        cls,
        session_id: str,
        *,
        api_key: str | None = None,
        endpoint: str | None = None,
        client: HypergraphClient | None = None,
        embed_model: LlamaEmbedding | Embedder | None = None,
        decay_lambda: float = 0.0,
        history_limit: int = 0,
        context_k: int = 4,
        k_hops: int = 2,
        **kwargs: Any,
    ) -> TemporalHypergraphChatMemory:
        memory = cls(
            session_id=session_id, history_limit=history_limit, context_k=context_k, k_hops=k_hops
        )
        memory._store = SessionMemory(
            session_id,
            client,
            endpoint=endpoint,
            api_key=api_key,
            embedder=as_embedder(embed_model),
            decay_lambda=decay_lambda,
        )
        return memory

    @property
    def store(self) -> SessionMemory:
        return self._store

    def _with_context(self, history: list[ChatMessage], input: str | None) -> list[ChatMessage]:
        if not input or self.context_k <= 0:
            return history
        shown = {m.additional_kwargs.get("hyperedge_id") for m in history}
        facts = [
            e for e in self._store.relevant(input, self.context_k, k_hops=self.k_hops)
            if e.id not in shown and e.summary
        ]
        if not facts:
            return history
        lines = "\n".join(f"- {e.summary}" for e in facts)
        return [ChatMessage(role=MessageRole.SYSTEM, content=f"{CONTEXT_HEADER}\n{lines}"), *history]

    def get(self, input: str | None = None, **kwargs: Any) -> list[ChatMessage]:
        history = [to_message(r) for r in self._store.history(self.history_limit)]
        return self._with_context(history, input)

    def get_all(self) -> list[ChatMessage]:
        return [to_message(r) for r in self._store.history()]

    def put(self, message: ChatMessage) -> None:
        self._store.add([to_record(message)])

    def put_messages(self, messages: list[ChatMessage]) -> None:
        self._store.add([to_record(m) for m in messages])

    async def aput(self, message: ChatMessage) -> None:
        await self._store.aadd([to_record(message)])

    async def aput_messages(self, messages: list[ChatMessage]) -> None:
        await self._store.aadd([to_record(m) for m in messages])

    async def aget_all(self) -> list[ChatMessage]:
        return [to_message(r) for r in await self._store.ahistory()]

    def set(self, messages: list[ChatMessage]) -> None:
        self.reset()
        self.put_messages(messages)

    def reset(self) -> None:
        self._store.clear()

    async def areset(self) -> None:
        await self._store.aclear()


def TemporalHypergraph(  # noqa: N802 - mirrors the class name used by the other SDKs
    api_key: str | None = None, *, session_id: str = DEFAULT_SESSION, **kwargs: Any
) -> TemporalHypergraphChatMemory:
    """``memory = TemporalHypergraph(api_key="...")``: chat memory on session
    ``"default"`` unless ``session_id`` is given. Other keyword arguments are
    those of :meth:`TemporalHypergraphChatMemory.from_defaults`."""
    return TemporalHypergraphChatMemory.from_defaults(session_id, api_key=api_key, **kwargs)
