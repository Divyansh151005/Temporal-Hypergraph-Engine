"""Chat-session storage shared by the framework integrations.

Each message becomes one hyperedge:

* ``event_type``: ``MESSAGE_HUMAN`` / ``MESSAGE_AI`` / ``MESSAGE_SYSTEM`` / ``MESSAGE_TOOL``
* ``summary``: the message text
* participants: the session node, a fresh record node, and any entity ids
* ``valid_from``: the write time; an embedding of the text when an embedder is set
"""

from __future__ import annotations

import enum
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .aio import AsyncHypergraphClient
from .client import HypergraphClient
from .embeddings import Embedder
from .session import entities_of, record_node, record_of, scoped_participants, session_node
from .types import Hyperedge, ScoredEdge, now


class Role(str, enum.Enum):
    HUMAN = "human"
    AI = "ai"
    SYSTEM = "system"
    TOOL = "tool"

    @property
    def event_type(self) -> str:
        return f"MESSAGE_{self.name}"

    @classmethod
    def from_event_type(cls, event_type: str) -> Role | None:
        name = event_type.removeprefix("MESSAGE_")
        return cls.__members__.get(name) if name != event_type else None


MESSAGE_EVENT_TYPES: tuple[str, ...] = tuple(role.event_type for role in Role)


@dataclass(frozen=True)
class ChatRecord:
    """One stored chat message."""

    role: Role
    content: str
    entities: tuple[str, ...] = ()
    timestamp: int | None = None
    id: str | None = None


def _sort_key(edge: ScoredEdge) -> tuple[int, str]:
    return (edge.valid_from or 0, record_of(edge.participants) or "")


def _to_record(edge: ScoredEdge) -> ChatRecord | None:
    role = Role.from_event_type(edge.event_type)
    if role is None:
        return None
    return ChatRecord(
        role=role,
        content=edge.summary,
        entities=tuple(entities_of(edge.participants)),
        timestamp=edge.valid_from,
        id=edge.id,
    )


class SessionMemory:
    """Messages of one session, stored as hyperedges around ``session:<id>``."""

    def __init__(
        self,
        session_id: str,
        client: HypergraphClient | None = None,
        *,
        endpoint: str | None = None,
        api_key: str | None = None,
        embedder: Embedder | None = None,
        decay_lambda: float = 0.0,
    ) -> None:
        self.session_id = session_id
        self.node = session_node(session_id)
        self.client = client or HypergraphClient(endpoint, api_key)
        self._endpoint = endpoint or self.client.endpoint
        self._api_key = api_key
        self._aclient: AsyncHypergraphClient | None = None
        self.embedder = embedder
        self.decay_lambda = decay_lambda

    @property
    def aclient(self) -> AsyncHypergraphClient:
        if self._aclient is None:
            self._aclient = AsyncHypergraphClient(self._endpoint, self._api_key)
        return self._aclient

    def _edges(self, records: Sequence[ChatRecord]) -> list[Hyperedge]:
        vectors: list[list[float] | None] = [None] * len(records)
        if self.embedder is not None and records:
            vectors = list(self.embedder.embed_documents([r.content for r in records]))
        timestamp = now()
        return [
            Hyperedge(
                event_type=record.role.event_type,
                participants=scoped_participants(
                    self.session_id, [record_node(), *record.entities]
                ),
                valid_from=record.timestamp or timestamp,
                summary=record.content,
                embedding=vector,
            )
            for record, vector in zip(records, vectors)
        ]

    @staticmethod
    def _history(edges: Iterable[ScoredEdge]) -> list[ChatRecord]:
        ordered = sorted(edges, key=_sort_key)
        return [r for r in map(_to_record, ordered) if r is not None]

    def _history_args(self, limit: int) -> dict:
        return {
            "k_hops": 1,
            "top_k": limit,
            "chronological": True,
            "event_types": MESSAGE_EVENT_TYPES,
            "as_of": now() + 1,
        }

    def add(self, records: Sequence[ChatRecord]) -> None:
        if not records:
            return
        result = self.client.insert_batch(self._edges(records), self.session_id)
        if not result.ok:
            raise RuntimeError(f"failed to store messages: {list(result.errors)}")

    def history(self, limit: int = 0) -> list[ChatRecord]:
        """Messages oldest first; ``limit`` keeps only the newest ``limit`` (0 = all)."""
        return self._history(self.client.traverse([self.node], **self._history_args(limit)))

    def relevant(self, query: str, k: int = 4, *, k_hops: int = 1) -> list[ScoredEdge]:
        """Session hyperedges most similar to ``query`` (by decay alone without an embedder)."""
        embedding = self.embedder.embed_query(query) if self.embedder else None
        return self.client.traverse(
            [self.node],
            k_hops=k_hops,
            embedding=embedding,
            top_k=k,
            decay_lambda=self.decay_lambda,
            as_of=now() + 1,
        )

    def clear(self) -> int:
        """Deletes every message of the session; returns how many were removed."""
        ids = [r.id for r in self.history() if r.id]
        return self.client.delete(ids)

    async def aadd(self, records: Sequence[ChatRecord]) -> None:
        if not records:
            return
        result = await self.aclient.insert_batch(self._edges(records), self.session_id)
        if not result.ok:
            raise RuntimeError(f"failed to store messages: {list(result.errors)}")

    async def ahistory(self, limit: int = 0) -> list[ChatRecord]:
        edges = await self.aclient.traverse([self.node], **self._history_args(limit))
        return self._history(edges)

    async def aclear(self) -> int:
        ids = [r.id for r in await self.ahistory() if r.id]
        return await self.aclient.delete(ids)


DEFAULT_SESSION = "default"


class TemporalHypergraph(SessionMemory):
    """One-line entry point: ``memory = TemporalHypergraph(api_key="...")``.

    A :class:`SessionMemory` on session ``"default"`` unless ``session_id`` is
    given; the underlying :class:`HypergraphClient` is ``memory.client``.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        session_id: str = DEFAULT_SESSION,
        endpoint: str | None = None,
        embedder: Embedder | None = None,
        decay_lambda: float = 0.0,
        client: HypergraphClient | None = None,
    ) -> None:
        super().__init__(
            session_id,
            client,
            endpoint=endpoint,
            api_key=api_key,
            embedder=embedder,
            decay_lambda=decay_lambda,
        )
