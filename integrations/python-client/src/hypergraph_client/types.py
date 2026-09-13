"""Client-side data types and their protobuf conversions."""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import grpc
import numpy as np

from ._proto import pb
from .embeddings import from_f32_bytes, to_f32_bytes

DEFAULT_ENDPOINT = "localhost:50051"
API_KEY_HEADER = "x-api-key"
MAX_MESSAGE_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class Hyperedge:
    """A hyperedge to insert."""

    event_type: str
    participants: Sequence[str]
    valid_from: int
    valid_to: int | None = None
    summary: str = ""
    confidence: float = 1.0
    embedding: Sequence[float] | np.ndarray | None = None

    def to_pb(self) -> pb.HyperedgePayload:
        boundary = pb.TemporalBoundary(
            valid_from_epoch=int(self.valid_from), is_ongoing=self.valid_to is None
        )
        if self.valid_to is not None:
            boundary.valid_to_epoch = int(self.valid_to)
        return pb.HyperedgePayload(
            event_type=self.event_type,
            summary=self.summary,
            participants=[pb.Participant(entity_id=p) for p in self.participants],
            temporal_boundary=boundary,
            confidence_score=float(self.confidence),
            embedding_raw=to_f32_bytes(self.embedding),
        )


@dataclass(frozen=True)
class ScoredEdge:
    """A hyperedge returned by a query or traversal."""

    id: str
    event_type: str
    participants: tuple[str, ...]
    valid_from: int | None
    valid_to: int | None
    is_ongoing: bool
    summary: str
    confidence: float
    score: float = 0.0
    embedding: np.ndarray | None = field(default=None, compare=False, repr=False)

    @classmethod
    def from_pb(cls, payload: pb.HyperedgePayload, score: float = 0.0) -> ScoredEdge:
        boundary = payload.temporal_boundary
        return cls(
            id=payload.id,
            event_type=payload.event_type,
            participants=tuple(p.entity_id for p in payload.participants),
            valid_from=(
                boundary.valid_from_epoch if boundary.HasField("valid_from_epoch") else None
            ),
            valid_to=boundary.valid_to_epoch if boundary.HasField("valid_to_epoch") else None,
            is_ongoing=boundary.is_ongoing,
            summary=payload.summary,
            confidence=payload.confidence_score,
            score=score,
            embedding=from_f32_bytes(payload.embedding_raw),
        )


@dataclass(frozen=True)
class BatchResult:
    """Outcome of a streamed batch insert."""

    processed: int
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


class HypergraphError(RuntimeError):
    """A failed engine call."""

    def __init__(self, code: grpc.StatusCode, message: str) -> None:
        super().__init__(f"{code.name}: {message}")
        self.code = code
        self.message = message


class AuthenticationError(HypergraphError):
    """The server rejected the API key."""


def wrap_rpc_error(error: grpc.RpcError) -> HypergraphError:
    code = error.code() if hasattr(error, "code") else grpc.StatusCode.UNKNOWN
    details = error.details() if hasattr(error, "details") else str(error)
    cls = AuthenticationError if code == grpc.StatusCode.UNAUTHENTICATED else HypergraphError
    return cls(code, details or "")


def resolve_endpoint(endpoint: str | None) -> str:
    """``endpoint``, else ``$HYPERGRAPH_ENDPOINT``, else ``localhost:50051``.

    Accepts ``host:port``, ``tcp://host:port`` and ``unix:///path``.
    """
    target = endpoint or os.environ.get("HYPERGRAPH_ENDPOINT") or DEFAULT_ENDPOINT
    return target.removeprefix("tcp://")


def resolve_api_key(api_key: str | None) -> str | None:
    return api_key or os.environ.get("HYPERGRAPH_API_KEY") or None


def channel_options(target: str) -> list[tuple[str, object]]:
    options: list[tuple[str, object]] = [
        ("grpc.max_send_message_length", MAX_MESSAGE_BYTES),
        ("grpc.max_receive_message_length", MAX_MESSAGE_BYTES),
    ]
    if target.startswith("unix:"):
        # grpcio otherwise sends the socket path as :authority, which h2 rejects.
        options.append(("grpc.default_authority", "localhost"))
    return options


def now() -> int:
    return int(time.time())


def insert_request(edge: Hyperedge, session_id: str) -> pb.InsertHyperedgeRequest:
    return pb.InsertHyperedgeRequest(session_id=session_id, hyperedge=edge.to_pb())


def query_request(
    embedding: Sequence[float] | np.ndarray,
    *,
    as_of: int | None,
    top_k: int,
    decay_lambda: float,
    session_id: str,
) -> pb.QueryTemporalRequest:
    return pb.QueryTemporalRequest(
        session_id=session_id,
        query_embedding_raw=to_f32_bytes(embedding),
        target_timestamp_epoch=now() if as_of is None else int(as_of),
        decay_lambda=decay_lambda,
        top_k=top_k,
    )


def traverse_request(
    seeds: Sequence[str],
    *,
    k_hops: int,
    as_of: int | None,
    embedding: Sequence[float] | np.ndarray | None,
    top_k: int,
    chronological: bool,
    event_types: Sequence[str],
    decay_lambda: float,
) -> pb.TraverseTemporalRequest:
    return pb.TraverseTemporalRequest(
        seed_entity_ids=list(seeds),
        k_hops=k_hops,
        as_of_epoch=0 if as_of is None else int(as_of),
        query_embedding_raw=to_f32_bytes(embedding),
        decay_lambda=decay_lambda,
        top_k=top_k,
        order=(
            pb.TRAVERSAL_ORDER_CHRONOLOGICAL if chronological else pb.TRAVERSAL_ORDER_RELEVANCE
        ),
        event_types=list(event_types),
    )


def scored_edges(response: pb.QueryTemporalResponse) -> list[ScoredEdge]:
    return [
        ScoredEdge.from_pb(edge, score)
        for edge, score in zip(response.matched_edges, response.decay_scores)
    ]
