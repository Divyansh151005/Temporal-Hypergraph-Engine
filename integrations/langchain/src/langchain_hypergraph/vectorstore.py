"""``VectorStore`` over hyperedges, with temporal and multi-hop search options."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

from hypergraph_client import (
    Hyperedge,
    HypergraphClient,
    ScoredEdge,
    entities_of,
    record_node,
    scoped_participants,
    session_node,
)
from hypergraph_client.types import now
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

DOCUMENT_EVENT_TYPE = "DOCUMENT"


def edge_to_document(edge: ScoredEdge) -> Document:
    return Document(
        page_content=edge.summary,
        id=edge.id,
        metadata={
            "id": edge.id,
            "event_type": edge.event_type,
            "entities": entities_of(edge.participants),
            "participants": list(edge.participants),
            "valid_from": edge.valid_from,
            "valid_to": edge.valid_to,
            "confidence": edge.confidence,
        },
    )


class TemporalHypergraphVectorStore(VectorStore):
    """Texts stored as hyperedges in the Temporal Hypergraph Engine.

    Recognized metadata keys:

    * ``entities``: entity ids the text is linked to (graph participants)
    * ``valid_from`` / ``valid_to``: validity window in epoch seconds
      (default: from now, open-ended)
    * ``event_type``: hyperedge type (default ``DOCUMENT``)
    * ``confidence``: ``[0, 1]`` score that scales retrieval scores

    Other metadata is dropped: the engine stores only the text and the graph
    structure. Ids are assigned by the engine; ``add_texts`` returns them.

    Search keyword arguments:

    * ``as_of``: epoch seconds; only hyperedges valid at that instant match
    * ``session_id`` / ``seed_entities``: restrict the search to hyperedges
      within ``k_hops`` (default 1) of those nodes
    * ``decay_lambda``: temporal decay rate
    """

    def __init__(
        self,
        embedding: Embeddings,
        client: HypergraphClient | None = None,
        *,
        endpoint: str | None = None,
        api_key: str | None = None,
        session_id: str | None = None,
        event_type: str = DOCUMENT_EVENT_TYPE,
    ) -> None:
        self._embedding = embedding
        self.client = client or HypergraphClient(endpoint, api_key)
        self.session_id = session_id
        self.event_type = event_type

    @property
    def embeddings(self) -> Embeddings:
        return self._embedding

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict] | None = None,
        *,
        ids: list[str] | None = None,
        session_id: str | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Stores ``texts``; returns their engine-assigned ids (``ids`` is ignored)."""
        texts = list(texts)
        metadatas = metadatas or [{} for _ in texts]
        if len(metadatas) != len(texts):
            raise ValueError("metadatas must have one entry per text")
        vectors = self._embedding.embed_documents(texts)
        session = session_id or self.session_id
        timestamp = now()
        result = []
        for text, metadata, vector in zip(texts, metadatas, vectors):
            participants = [record_node(), *map(str, metadata.get("entities") or ())]
            edge = Hyperedge(
                event_type=str(metadata.get("event_type") or self.event_type),
                participants=scoped_participants(session, participants),
                valid_from=int(metadata.get("valid_from") or timestamp),
                valid_to=metadata.get("valid_to"),
                summary=text,
                confidence=float(metadata.get("confidence", 1.0)),
                embedding=vector,
            )
            result.append(self.client.insert(edge, session or ""))
        return result

    def delete(self, ids: list[str] | None = None, **kwargs: Any) -> bool | None:
        if not ids:
            return None
        self.client.delete(ids)
        return True

    def _search(
        self,
        vector: Sequence[float],
        k: int,
        *,
        as_of: int | None = None,
        session_id: str | None = None,
        seed_entities: Sequence[str] = (),
        k_hops: int = 1,
        decay_lambda: float = 0.0,
        **_: Any,
    ) -> list[ScoredEdge]:
        session = session_id or self.session_id
        seeds = [*([session_node(session)] if session else []), *seed_entities]
        if seeds:
            return self.client.traverse(
                seeds,
                k_hops=k_hops,
                as_of=as_of,
                embedding=vector,
                top_k=k,
                decay_lambda=decay_lambda,
            )
        return self.client.query(vector, as_of=as_of, top_k=k, decay_lambda=decay_lambda)

    def similarity_search_with_score_by_vector(
        self, embedding: list[float], k: int = 4, **kwargs: Any
    ) -> list[tuple[Document, float]]:
        return [(edge_to_document(e), e.score) for e in self._search(embedding, k, **kwargs)]

    def similarity_search_by_vector(
        self, embedding: list[float], k: int = 4, **kwargs: Any
    ) -> list[Document]:
        return [doc for doc, _ in self.similarity_search_with_score_by_vector(embedding, k, **kwargs)]

    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[tuple[Document, float]]:
        vector = self._embedding.embed_query(query)
        return self.similarity_search_with_score_by_vector(vector, k, **kwargs)

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> list[Document]:
        return [doc for doc, _ in self.similarity_search_with_score(query, k, **kwargs)]

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        # Scores are cosine * confidence * decay, already "higher is better".
        return lambda score: max(0.0, min(1.0, score))

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict] | None = None,
        *,
        client: HypergraphClient | None = None,
        endpoint: str | None = None,
        api_key: str | None = None,
        session_id: str | None = None,
        **kwargs: Any,
    ) -> TemporalHypergraphVectorStore:
        store = cls(embedding, client, endpoint=endpoint, api_key=api_key, session_id=session_id)
        store.add_texts(texts, metadatas)
        return store
