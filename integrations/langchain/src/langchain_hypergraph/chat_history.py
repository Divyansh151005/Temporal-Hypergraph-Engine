"""``BaseChatMessageHistory`` backed by a hypergraph session node."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from hypergraph_client import DEFAULT_SESSION, Embedder, HypergraphClient, ScoredEdge, SessionMemory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage

from .messages import to_messages, to_records


class TemporalHypergraphMemory(BaseChatMessageHistory):
    """Chat history of one session, stored in the Temporal Hypergraph Engine.

    ``memory = TemporalHypergraphMemory(api_key="...", session_id="...")``

    Every message becomes a hyperedge linking ``session:<session_id>``, a record
    node, and any entity ids in ``message.additional_kwargs["entities"]``, so
    the same facts are reachable by multi-hop traversal from those entities.

    Args:
        session_id: Conversation id.
        api_key: Server API key (default ``$HYPERGRAPH_API_KEY``).
        endpoint: Server address (default ``$HYPERGRAPH_ENDPOINT`` or ``localhost:50051``).
        embeddings: Optional LangChain ``Embeddings``; enables similarity in
            :meth:`relevant_context`. Its dimension must match the server's.
        decay_lambda: Temporal decay rate used by :meth:`relevant_context`.
        history_limit: Return only the newest ``history_limit`` messages (0 = all).
        client: A preconfigured client (overrides ``endpoint`` and ``api_key``).
    """

    def __init__(
        self,
        session_id: str,
        api_key: str | None = None,
        *,
        endpoint: str | None = None,
        embeddings: Embedder | None = None,
        decay_lambda: float = 0.0,
        history_limit: int = 0,
        client: HypergraphClient | None = None,
    ) -> None:
        self.session_id = session_id
        self.history_limit = history_limit
        self.store = SessionMemory(
            session_id,
            client,
            endpoint=endpoint,
            api_key=api_key,
            embedder=embeddings,
            decay_lambda=decay_lambda,
        )

    @property
    def messages(self) -> list[BaseMessage]:  # type: ignore[override]
        """Messages oldest first."""
        return to_messages(self.store.history(self.history_limit))

    async def aget_messages(self) -> list[BaseMessage]:
        return to_messages(await self.store.ahistory(self.history_limit))

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        self.store.add(to_records(messages))

    async def aadd_messages(self, messages: Sequence[BaseMessage]) -> None:
        await self.store.aadd(to_records(messages))

    def clear(self) -> None:
        self.store.clear()

    async def aclear(self) -> None:
        await self.store.aclear()

    def relevant_context(self, query: str, k: int = 4, *, k_hops: int = 1) -> list[ScoredEdge]:
        """The ``k`` session hyperedges most relevant to ``query``.

        With ``k_hops > 1`` the traversal also reaches facts linked through the
        messages' entities.
        """
        return self.store.relevant(query, k, k_hops=k_hops)


class TemporalHypergraph(TemporalHypergraphMemory):
    """``memory = TemporalHypergraph(api_key="...")``: chat history on session
    ``"default"`` unless ``session_id`` is given. Other keyword arguments are
    those of :class:`TemporalHypergraphMemory`."""

    def __init__(
        self, api_key: str | None = None, *, session_id: str = DEFAULT_SESSION, **kwargs: Any
    ) -> None:
        super().__init__(session_id, api_key, **kwargs)
