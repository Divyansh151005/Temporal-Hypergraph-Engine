"""Synchronous gRPC client."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence

import grpc
import numpy as np

from ._proto import pb, pb_grpc
from .types import (
    API_KEY_HEADER,
    BatchResult,
    Hyperedge,
    ScoredEdge,
    channel_options,
    insert_request,
    now,
    query_request,
    resolve_api_key,
    resolve_endpoint,
    scored_edges,
    traverse_request,
    wrap_rpc_error,
)


class HypergraphClient:
    """Blocking client for ``hypergraph.v1.HypergraphEngineService``.

    ``endpoint`` defaults to ``$HYPERGRAPH_ENDPOINT`` or ``localhost:50051``
    and ``api_key`` to ``$HYPERGRAPH_API_KEY``.
    """

    def __init__(
        self,
        endpoint: str | None = None,
        api_key: str | None = None,
        timeout: float = 5.0,
    ) -> None:
        self.endpoint = resolve_endpoint(endpoint)
        self.timeout = timeout
        key = resolve_api_key(api_key)
        self._metadata = ((API_KEY_HEADER, key),) if key else ()
        self._channel = grpc.insecure_channel(self.endpoint, options=channel_options(self.endpoint))
        self._stub = pb_grpc.HypergraphEngineServiceStub(self._channel)

    def __enter__(self) -> HypergraphClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._channel.close()

    def _call(self, method, request):
        try:
            return method(request, timeout=self.timeout, metadata=self._metadata)
        except grpc.RpcError as error:
            raise wrap_rpc_error(error) from error

    def insert(self, edge: Hyperedge, session_id: str = "") -> str:
        """Inserts one hyperedge and returns its id."""
        return self._call(self._stub.InsertHyperedge, insert_request(edge, session_id)).hyperedge_id

    def insert_batch(self, edges: Iterable[Hyperedge], session_id: str = "") -> BatchResult:
        """Streams hyperedges in one call; the server group-commits them."""
        requests = (insert_request(edge, session_id) for edge in edges)
        response = self._call(self._stub.InsertHyperedgeBatch, requests)
        return BatchResult(response.processed_count, tuple(response.errors))

    def query(
        self,
        embedding: Sequence[float] | np.ndarray,
        *,
        as_of: int | None = None,
        top_k: int = 10,
        decay_lambda: float = 0.0,
        session_id: str = "",
    ) -> list[ScoredEdge]:
        """Global similarity search ranked by ``cosine * confidence * decay``."""
        request = query_request(
            embedding, as_of=as_of, top_k=top_k, decay_lambda=decay_lambda, session_id=session_id
        )
        return scored_edges(self._call(self._stub.QueryTemporalContext, request))

    def stream(
        self,
        embedding: Sequence[float] | np.ndarray,
        *,
        as_of: int | None = None,
        top_k: int = 10,
        decay_lambda: float = 0.0,
    ) -> Iterator[ScoredEdge]:
        """Like :meth:`query`, streamed edge by edge (scores are not streamed)."""
        request = query_request(
            embedding, as_of=as_of, top_k=top_k, decay_lambda=decay_lambda, session_id=""
        )
        try:
            for payload in self._stub.StreamTemporalContext(
                request, timeout=self.timeout, metadata=self._metadata
            ):
                yield ScoredEdge.from_pb(payload)
        except grpc.RpcError as error:
            raise wrap_rpc_error(error) from error

    def traverse(
        self,
        seeds: Sequence[str],
        *,
        k_hops: int = 1,
        as_of: int | None = None,
        embedding: Sequence[float] | np.ndarray | None = None,
        top_k: int = 10,
        chronological: bool = False,
        event_types: Sequence[str] = (),
        decay_lambda: float = 0.0,
    ) -> list[ScoredEdge]:
        """Hyperedges within ``k_hops`` of ``seeds`` valid at ``as_of`` (default now).

        ``top_k=0`` returns every match. With ``embedding`` results are
        re-ranked by similarity; with ``chronological`` newest come first.
        """
        request = traverse_request(
            seeds,
            k_hops=k_hops,
            as_of=as_of,
            embedding=embedding,
            top_k=top_k,
            chronological=chronological,
            event_types=event_types,
            decay_lambda=decay_lambda,
        )
        return scored_edges(self._call(self._stub.TraverseTemporalContext, request))

    def delete(self, hyperedge_ids: Iterable[str]) -> int:
        """Soft-deletes hyperedges; returns how many were live."""
        request = pb.DeleteHyperedgesRequest(hyperedge_ids=list(hyperedge_ids))
        if not request.hyperedge_ids:
            return 0
        return self._call(self._stub.DeleteHyperedges, request).deleted_count

    def store_kv_block(
        self, block_hash: str, data: bytes, hyperedge_ids: Sequence[str] = ()
    ) -> bool:
        """Stores an opaque KV-cache block under a 64-hex-digit ``block_hash``."""
        request = pb.StoreKVBlocksRequest(
            block_hash=block_hash, kv_chunk_bytes=data, associated_hyperedge_ids=list(hyperedge_ids)
        )
        return self._call(self._stub.StoreKVBlocks, request).success

    def load_kv_block(self, block_hash: str) -> bytes | None:
        """The stored block bytes, or ``None`` on a miss."""
        response = self._call(self._stub.LoadKVBlocks, pb.LoadKVBlocksRequest(block_hash=block_hash))
        return bytes(response.kv_chunk_bytes) if response.hit else None

    def plan_kv_evictions(
        self, decay_threshold: float, *, now_epoch: int | None = None, max_candidates: int = 0
    ) -> list[tuple[str, float]]:
        """Blocks whose hyperedges decayed below ``decay_threshold``, lowest score first."""
        request = pb.PlanKVEvictionsRequest(
            decay_threshold=decay_threshold,
            now_epoch=now() if now_epoch is None else now_epoch,
            max_candidates=max_candidates,
        )
        response = self._call(self._stub.PlanKVEvictions, request)
        return list(zip(response.block_hashes, response.scores))

    def health(self) -> bool:
        return self._call(self._stub.HealthCheck, pb.HealthCheckRequest()).healthy

    def checkpoint(self) -> int:
        return self._call(self._stub.TriggerCheckpoint, pb.CheckpointRequest()).wal_sequence_id
