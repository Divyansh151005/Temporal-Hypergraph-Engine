"""``asyncio`` gRPC client mirroring :class:`hypergraph_client.HypergraphClient`."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable, Sequence

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
    query_request,
    resolve_api_key,
    resolve_endpoint,
    scored_edges,
    traverse_request,
    wrap_rpc_error,
)


class AsyncHypergraphClient:
    """Async client; create and use it inside a running event loop."""

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
        self._channel = grpc.aio.insecure_channel(
            self.endpoint, options=channel_options(self.endpoint)
        )
        self._stub = pb_grpc.HypergraphEngineServiceStub(self._channel)

    async def __aenter__(self) -> AsyncHypergraphClient:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()

    async def close(self) -> None:
        await self._channel.close()

    async def _call(self, method, request):
        try:
            return await method(request, timeout=self.timeout, metadata=self._metadata)
        except grpc.RpcError as error:
            raise wrap_rpc_error(error) from error

    async def insert(self, edge: Hyperedge, session_id: str = "") -> str:
        response = await self._call(self._stub.InsertHyperedge, insert_request(edge, session_id))
        return response.hyperedge_id

    async def insert_batch(self, edges: Iterable[Hyperedge], session_id: str = "") -> BatchResult:
        requests = [insert_request(edge, session_id) for edge in edges]
        response = await self._call(self._stub.InsertHyperedgeBatch, iter(requests))
        return BatchResult(response.processed_count, tuple(response.errors))

    async def query(
        self,
        embedding: Sequence[float] | np.ndarray,
        *,
        as_of: int | None = None,
        top_k: int = 10,
        decay_lambda: float = 0.0,
        session_id: str = "",
    ) -> list[ScoredEdge]:
        request = query_request(
            embedding, as_of=as_of, top_k=top_k, decay_lambda=decay_lambda, session_id=session_id
        )
        return scored_edges(await self._call(self._stub.QueryTemporalContext, request))

    async def stream(
        self,
        embedding: Sequence[float] | np.ndarray,
        *,
        as_of: int | None = None,
        top_k: int = 10,
        decay_lambda: float = 0.0,
    ) -> AsyncIterator[ScoredEdge]:
        request = query_request(
            embedding, as_of=as_of, top_k=top_k, decay_lambda=decay_lambda, session_id=""
        )
        try:
            async for payload in self._stub.StreamTemporalContext(
                request, timeout=self.timeout, metadata=self._metadata
            ):
                yield ScoredEdge.from_pb(payload)
        except grpc.RpcError as error:
            raise wrap_rpc_error(error) from error

    async def traverse(
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
        return scored_edges(await self._call(self._stub.TraverseTemporalContext, request))

    async def delete(self, hyperedge_ids: Iterable[str]) -> int:
        request = pb.DeleteHyperedgesRequest(hyperedge_ids=list(hyperedge_ids))
        if not request.hyperedge_ids:
            return 0
        return (await self._call(self._stub.DeleteHyperedges, request)).deleted_count

    async def health(self) -> bool:
        return (await self._call(self._stub.HealthCheck, pb.HealthCheckRequest())).healthy
