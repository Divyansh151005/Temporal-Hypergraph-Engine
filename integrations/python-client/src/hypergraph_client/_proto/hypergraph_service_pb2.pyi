from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class TraversalOrder(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    TRAVERSAL_ORDER_RELEVANCE: _ClassVar[TraversalOrder]
    TRAVERSAL_ORDER_CHRONOLOGICAL: _ClassVar[TraversalOrder]

class KvPrecision(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    KV_PRECISION_UNSPECIFIED: _ClassVar[KvPrecision]
    KV_PRECISION_FP16: _ClassVar[KvPrecision]
    KV_PRECISION_BF16: _ClassVar[KvPrecision]
    KV_PRECISION_INT8: _ClassVar[KvPrecision]
TRAVERSAL_ORDER_RELEVANCE: TraversalOrder
TRAVERSAL_ORDER_CHRONOLOGICAL: TraversalOrder
KV_PRECISION_UNSPECIFIED: KvPrecision
KV_PRECISION_FP16: KvPrecision
KV_PRECISION_BF16: KvPrecision
KV_PRECISION_INT8: KvPrecision

class Participant(_message.Message):
    __slots__ = ("entity_id", "role")
    ENTITY_ID_FIELD_NUMBER: _ClassVar[int]
    ROLE_FIELD_NUMBER: _ClassVar[int]
    entity_id: str
    role: str
    def __init__(self, entity_id: _Optional[str] = ..., role: _Optional[str] = ...) -> None: ...

class TemporalBoundary(_message.Message):
    __slots__ = ("valid_from_epoch", "valid_to_epoch", "is_ongoing", "raw_expression")
    VALID_FROM_EPOCH_FIELD_NUMBER: _ClassVar[int]
    VALID_TO_EPOCH_FIELD_NUMBER: _ClassVar[int]
    IS_ONGOING_FIELD_NUMBER: _ClassVar[int]
    RAW_EXPRESSION_FIELD_NUMBER: _ClassVar[int]
    valid_from_epoch: int
    valid_to_epoch: int
    is_ongoing: bool
    raw_expression: str
    def __init__(self, valid_from_epoch: _Optional[int] = ..., valid_to_epoch: _Optional[int] = ..., is_ongoing: bool = ..., raw_expression: _Optional[str] = ...) -> None: ...

class HyperedgePayload(_message.Message):
    __slots__ = ("id", "event_type", "summary", "participants", "temporal_boundary", "confidence_score", "embedding_raw")
    ID_FIELD_NUMBER: _ClassVar[int]
    EVENT_TYPE_FIELD_NUMBER: _ClassVar[int]
    SUMMARY_FIELD_NUMBER: _ClassVar[int]
    PARTICIPANTS_FIELD_NUMBER: _ClassVar[int]
    TEMPORAL_BOUNDARY_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_SCORE_FIELD_NUMBER: _ClassVar[int]
    EMBEDDING_RAW_FIELD_NUMBER: _ClassVar[int]
    id: str
    event_type: str
    summary: str
    participants: _containers.RepeatedCompositeFieldContainer[Participant]
    temporal_boundary: TemporalBoundary
    confidence_score: float
    embedding_raw: bytes
    def __init__(self, id: _Optional[str] = ..., event_type: _Optional[str] = ..., summary: _Optional[str] = ..., participants: _Optional[_Iterable[_Union[Participant, _Mapping]]] = ..., temporal_boundary: _Optional[_Union[TemporalBoundary, _Mapping]] = ..., confidence_score: _Optional[float] = ..., embedding_raw: _Optional[bytes] = ...) -> None: ...

class InsertHyperedgeRequest(_message.Message):
    __slots__ = ("session_id", "hyperedge")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    HYPEREDGE_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    hyperedge: HyperedgePayload
    def __init__(self, session_id: _Optional[str] = ..., hyperedge: _Optional[_Union[HyperedgePayload, _Mapping]] = ...) -> None: ...

class InsertHyperedgeResponse(_message.Message):
    __slots__ = ("success", "hyperedge_id", "sequence_id")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    HYPEREDGE_ID_FIELD_NUMBER: _ClassVar[int]
    SEQUENCE_ID_FIELD_NUMBER: _ClassVar[int]
    success: bool
    hyperedge_id: str
    sequence_id: int
    def __init__(self, success: bool = ..., hyperedge_id: _Optional[str] = ..., sequence_id: _Optional[int] = ...) -> None: ...

class BatchResponse(_message.Message):
    __slots__ = ("processed_count", "all_succeeded", "errors")
    PROCESSED_COUNT_FIELD_NUMBER: _ClassVar[int]
    ALL_SUCCEEDED_FIELD_NUMBER: _ClassVar[int]
    ERRORS_FIELD_NUMBER: _ClassVar[int]
    processed_count: int
    all_succeeded: bool
    errors: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, processed_count: _Optional[int] = ..., all_succeeded: bool = ..., errors: _Optional[_Iterable[str]] = ...) -> None: ...

class QueryTemporalRequest(_message.Message):
    __slots__ = ("session_id", "query_embedding_raw", "target_timestamp_epoch", "decay_lambda", "top_k")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    QUERY_EMBEDDING_RAW_FIELD_NUMBER: _ClassVar[int]
    TARGET_TIMESTAMP_EPOCH_FIELD_NUMBER: _ClassVar[int]
    DECAY_LAMBDA_FIELD_NUMBER: _ClassVar[int]
    TOP_K_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    query_embedding_raw: bytes
    target_timestamp_epoch: int
    decay_lambda: float
    top_k: int
    def __init__(self, session_id: _Optional[str] = ..., query_embedding_raw: _Optional[bytes] = ..., target_timestamp_epoch: _Optional[int] = ..., decay_lambda: _Optional[float] = ..., top_k: _Optional[int] = ...) -> None: ...

class QueryTemporalResponse(_message.Message):
    __slots__ = ("matched_edges", "decay_scores", "query_latency_us")
    MATCHED_EDGES_FIELD_NUMBER: _ClassVar[int]
    DECAY_SCORES_FIELD_NUMBER: _ClassVar[int]
    QUERY_LATENCY_US_FIELD_NUMBER: _ClassVar[int]
    matched_edges: _containers.RepeatedCompositeFieldContainer[HyperedgePayload]
    decay_scores: _containers.RepeatedScalarFieldContainer[float]
    query_latency_us: int
    def __init__(self, matched_edges: _Optional[_Iterable[_Union[HyperedgePayload, _Mapping]]] = ..., decay_scores: _Optional[_Iterable[float]] = ..., query_latency_us: _Optional[int] = ...) -> None: ...

class TraverseTemporalRequest(_message.Message):
    __slots__ = ("seed_entity_ids", "k_hops", "as_of_epoch", "query_embedding_raw", "decay_lambda", "top_k", "order", "event_types")
    SEED_ENTITY_IDS_FIELD_NUMBER: _ClassVar[int]
    K_HOPS_FIELD_NUMBER: _ClassVar[int]
    AS_OF_EPOCH_FIELD_NUMBER: _ClassVar[int]
    QUERY_EMBEDDING_RAW_FIELD_NUMBER: _ClassVar[int]
    DECAY_LAMBDA_FIELD_NUMBER: _ClassVar[int]
    TOP_K_FIELD_NUMBER: _ClassVar[int]
    ORDER_FIELD_NUMBER: _ClassVar[int]
    EVENT_TYPES_FIELD_NUMBER: _ClassVar[int]
    seed_entity_ids: _containers.RepeatedScalarFieldContainer[str]
    k_hops: int
    as_of_epoch: int
    query_embedding_raw: bytes
    decay_lambda: float
    top_k: int
    order: TraversalOrder
    event_types: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, seed_entity_ids: _Optional[_Iterable[str]] = ..., k_hops: _Optional[int] = ..., as_of_epoch: _Optional[int] = ..., query_embedding_raw: _Optional[bytes] = ..., decay_lambda: _Optional[float] = ..., top_k: _Optional[int] = ..., order: _Optional[_Union[TraversalOrder, str]] = ..., event_types: _Optional[_Iterable[str]] = ...) -> None: ...

class DeleteHyperedgesRequest(_message.Message):
    __slots__ = ("hyperedge_ids",)
    HYPEREDGE_IDS_FIELD_NUMBER: _ClassVar[int]
    hyperedge_ids: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, hyperedge_ids: _Optional[_Iterable[str]] = ...) -> None: ...

class DeleteHyperedgesResponse(_message.Message):
    __slots__ = ("deleted_count",)
    DELETED_COUNT_FIELD_NUMBER: _ClassVar[int]
    deleted_count: int
    def __init__(self, deleted_count: _Optional[int] = ...) -> None: ...

class KvBlockLayout(_message.Message):
    __slots__ = ("num_layers", "num_heads", "head_dim", "block_size", "precision")
    NUM_LAYERS_FIELD_NUMBER: _ClassVar[int]
    NUM_HEADS_FIELD_NUMBER: _ClassVar[int]
    HEAD_DIM_FIELD_NUMBER: _ClassVar[int]
    BLOCK_SIZE_FIELD_NUMBER: _ClassVar[int]
    PRECISION_FIELD_NUMBER: _ClassVar[int]
    num_layers: int
    num_heads: int
    head_dim: int
    block_size: int
    precision: KvPrecision
    def __init__(self, num_layers: _Optional[int] = ..., num_heads: _Optional[int] = ..., head_dim: _Optional[int] = ..., block_size: _Optional[int] = ..., precision: _Optional[_Union[KvPrecision, str]] = ...) -> None: ...

class StoreKVBlocksRequest(_message.Message):
    __slots__ = ("block_hash", "kv_chunk_bytes", "associated_hyperedge_ids", "layout")
    BLOCK_HASH_FIELD_NUMBER: _ClassVar[int]
    KV_CHUNK_BYTES_FIELD_NUMBER: _ClassVar[int]
    ASSOCIATED_HYPEREDGE_IDS_FIELD_NUMBER: _ClassVar[int]
    LAYOUT_FIELD_NUMBER: _ClassVar[int]
    block_hash: str
    kv_chunk_bytes: bytes
    associated_hyperedge_ids: _containers.RepeatedScalarFieldContainer[str]
    layout: KvBlockLayout
    def __init__(self, block_hash: _Optional[str] = ..., kv_chunk_bytes: _Optional[bytes] = ..., associated_hyperedge_ids: _Optional[_Iterable[str]] = ..., layout: _Optional[_Union[KvBlockLayout, _Mapping]] = ...) -> None: ...

class StoreKVBlocksResponse(_message.Message):
    __slots__ = ("success", "bytes_written", "deduplicated")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    BYTES_WRITTEN_FIELD_NUMBER: _ClassVar[int]
    DEDUPLICATED_FIELD_NUMBER: _ClassVar[int]
    success: bool
    bytes_written: int
    deduplicated: bool
    def __init__(self, success: bool = ..., bytes_written: _Optional[int] = ..., deduplicated: bool = ...) -> None: ...

class LoadKVBlocksRequest(_message.Message):
    __slots__ = ("block_hash",)
    BLOCK_HASH_FIELD_NUMBER: _ClassVar[int]
    block_hash: str
    def __init__(self, block_hash: _Optional[str] = ...) -> None: ...

class LoadKVBlocksResponse(_message.Message):
    __slots__ = ("hit", "kv_chunk_bytes", "layout")
    HIT_FIELD_NUMBER: _ClassVar[int]
    KV_CHUNK_BYTES_FIELD_NUMBER: _ClassVar[int]
    LAYOUT_FIELD_NUMBER: _ClassVar[int]
    hit: bool
    kv_chunk_bytes: bytes
    layout: KvBlockLayout
    def __init__(self, hit: bool = ..., kv_chunk_bytes: _Optional[bytes] = ..., layout: _Optional[_Union[KvBlockLayout, _Mapping]] = ...) -> None: ...

class PlanKVEvictionsRequest(_message.Message):
    __slots__ = ("decay_threshold", "now_epoch", "max_candidates")
    DECAY_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    NOW_EPOCH_FIELD_NUMBER: _ClassVar[int]
    MAX_CANDIDATES_FIELD_NUMBER: _ClassVar[int]
    decay_threshold: float
    now_epoch: int
    max_candidates: int
    def __init__(self, decay_threshold: _Optional[float] = ..., now_epoch: _Optional[int] = ..., max_candidates: _Optional[int] = ...) -> None: ...

class PlanKVEvictionsResponse(_message.Message):
    __slots__ = ("block_hashes", "scores")
    BLOCK_HASHES_FIELD_NUMBER: _ClassVar[int]
    SCORES_FIELD_NUMBER: _ClassVar[int]
    block_hashes: _containers.RepeatedScalarFieldContainer[str]
    scores: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, block_hashes: _Optional[_Iterable[str]] = ..., scores: _Optional[_Iterable[float]] = ...) -> None: ...

class HealthCheckRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class HealthCheckResponse(_message.Message):
    __slots__ = ("healthy", "uptime_seconds", "active_connections")
    HEALTHY_FIELD_NUMBER: _ClassVar[int]
    UPTIME_SECONDS_FIELD_NUMBER: _ClassVar[int]
    ACTIVE_CONNECTIONS_FIELD_NUMBER: _ClassVar[int]
    healthy: bool
    uptime_seconds: int
    active_connections: int
    def __init__(self, healthy: bool = ..., uptime_seconds: _Optional[int] = ..., active_connections: _Optional[int] = ...) -> None: ...

class CheckpointRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class CheckpointResponse(_message.Message):
    __slots__ = ("success", "wal_sequence_id")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    WAL_SEQUENCE_ID_FIELD_NUMBER: _ClassVar[int]
    success: bool
    wal_sequence_id: int
    def __init__(self, success: bool = ..., wal_sequence_id: _Optional[int] = ...) -> None: ...
