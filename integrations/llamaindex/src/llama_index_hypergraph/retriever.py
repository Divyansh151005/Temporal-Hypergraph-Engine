"""``BaseRetriever`` doing temporal multi-hop traversal from linked entities."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence

from hypergraph_client import (
    AsyncHypergraphClient,
    Embedder,
    HypergraphClient,
    ScoredEdge,
    entities_of,
    session_node,
)
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from .embedding import LlamaEmbedding, as_embedder

EntityLinker = Callable[[str], Sequence[str]]

_HIDDEN_FROM_LLM = ["hyperedge_id", "participants"]


def exact_label_linker(entities: Iterable[str]) -> EntityLinker:
    """Links a query to every known entity whose label appears in it as a whole phrase.

    The label of ``org:acme_corp`` is ``acme corp``; matching is case-insensitive.
    """
    patterns = []
    for entity in entities:
        label = entity.split(":", 1)[-1].replace("_", " ").strip()
        if label:
            patterns.append((entity, re.compile(rf"\b{re.escape(label)}\b", re.IGNORECASE)))

    def link(query: str) -> list[str]:
        return [entity for entity, pattern in patterns if pattern.search(query)]

    return link


def edge_to_node(edge: ScoredEdge) -> NodeWithScore:
    node = TextNode(
        id_=edge.id,
        text=edge.summary,
        metadata={
            "hyperedge_id": edge.id,
            "event_type": edge.event_type,
            "participants": list(edge.participants),
            "entities": entities_of(edge.participants),
            "valid_from": edge.valid_from,
            "valid_to": edge.valid_to,
        },
        excluded_llm_metadata_keys=_HIDDEN_FROM_LLM,
        excluded_embed_metadata_keys=_HIDDEN_FROM_LLM,
    )
    return NodeWithScore(node=node, score=edge.score)


class TemporalHypergraphRetriever(BaseRetriever):
    """Retrieves hyperedges within ``k_hops`` of the entities a query mentions.

    Args:
        client: Server client (or pass ``endpoint`` / ``api_key``).
        entity_linker: Maps a query to seed entity ids; by default
            :func:`exact_label_linker` over ``known_entities``.
        known_entities: Entity ids for the default linker.
        seed_entities: Seeds always added to the linked ones.
        session_id: Also seed from ``session:<id>``.
        embed_model: Optional LlamaIndex ``BaseEmbedding`` or LangChain-style
            embedder; re-ranks results by similarity. Without seeds, the
            retriever falls back to global similarity search (and returns
            nothing when there is no embedder either).
        as_of: Epoch seconds the facts must be valid at (default now).
        k_hops: Traversal depth (1 to 4).
        top_k: Results to return.
        decay_lambda: Temporal decay rate.
    """

    def __init__(
        self,
        client: HypergraphClient | None = None,
        *,
        endpoint: str | None = None,
        api_key: str | None = None,
        entity_linker: EntityLinker | None = None,
        known_entities: Iterable[str] = (),
        seed_entities: Sequence[str] = (),
        session_id: str | None = None,
        embed_model: LlamaEmbedding | Embedder | None = None,
        as_of: int | None = None,
        k_hops: int = 2,
        top_k: int = 8,
        decay_lambda: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.client = client or HypergraphClient(endpoint, api_key)
        self._endpoint = endpoint or self.client.endpoint
        self._api_key = api_key
        self._aclient: AsyncHypergraphClient | None = None
        self.entity_linker = entity_linker or exact_label_linker(known_entities)
        self.seed_entities = list(seed_entities)
        self.session_id = session_id
        self.embedder = as_embedder(embed_model)
        self.as_of = as_of
        self.k_hops = k_hops
        self.top_k = top_k
        self.decay_lambda = decay_lambda

    @property
    def aclient(self) -> AsyncHypergraphClient:
        if self._aclient is None:
            self._aclient = AsyncHypergraphClient(self._endpoint, self._api_key)
        return self._aclient

    def seeds_for(self, query: str) -> list[str]:
        seeds = [session_node(self.session_id)] if self.session_id else []
        seeds += self.seed_entities
        seeds += self.entity_linker(query)
        return list(dict.fromkeys(seeds))

    def _plan(self, bundle: QueryBundle) -> tuple[list[str], list[float] | None]:
        vector = bundle.embedding
        if vector is None and self.embedder is not None:
            vector = self.embedder.embed_query(bundle.query_str)
        return self.seeds_for(bundle.query_str), vector

    def _traverse_args(self, vector: list[float] | None) -> dict:
        return {
            "k_hops": self.k_hops,
            "as_of": self.as_of,
            "embedding": vector,
            "top_k": self.top_k,
            "decay_lambda": self.decay_lambda,
        }

    def _query_args(self) -> dict:
        return {"as_of": self.as_of, "top_k": self.top_k, "decay_lambda": self.decay_lambda}

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        seeds, vector = self._plan(query_bundle)
        if seeds:
            edges = self.client.traverse(seeds, **self._traverse_args(vector))
        elif vector is not None:
            edges = self.client.query(vector, **self._query_args())
        else:
            edges = []
        return [edge_to_node(e) for e in edges]

    async def _aretrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        seeds, vector = self._plan(query_bundle)
        if seeds:
            edges = await self.aclient.traverse(seeds, **self._traverse_args(vector))
        elif vector is not None:
            edges = await self.aclient.query(vector, **self._query_args())
        else:
            edges = []
        return [edge_to_node(e) for e in edges]
