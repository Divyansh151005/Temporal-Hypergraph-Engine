"""Python client for the Temporal Hypergraph Engine."""

from .aio import AsyncHypergraphClient
from .client import HypergraphClient
from .embeddings import Embedder, HashingEmbedder, from_f32_bytes, to_f32_bytes
from .kv_tier import HypergraphKVOffloadTierManager, block_key
from .memory import (
    DEFAULT_SESSION,
    MESSAGE_EVENT_TYPES,
    ChatRecord,
    Role,
    SessionMemory,
    TemporalHypergraph,
)
from .session import (
    RECORD_PREFIX,
    SESSION_PREFIX,
    entities_of,
    record_node,
    record_of,
    scoped_participants,
    session_node,
)
from .types import (
    AuthenticationError,
    BatchResult,
    Hyperedge,
    HypergraphError,
    ScoredEdge,
)

__all__ = [
    "DEFAULT_SESSION",
    "MESSAGE_EVENT_TYPES",
    "RECORD_PREFIX",
    "SESSION_PREFIX",
    "AsyncHypergraphClient",
    "AuthenticationError",
    "BatchResult",
    "ChatRecord",
    "Embedder",
    "HashingEmbedder",
    "Hyperedge",
    "HypergraphClient",
    "HypergraphError",
    "HypergraphKVOffloadTierManager",
    "Role",
    "ScoredEdge",
    "SessionMemory",
    "TemporalHypergraph",
    "block_key",
    "entities_of",
    "from_f32_bytes",
    "record_node",
    "record_of",
    "scoped_participants",
    "session_node",
    "to_f32_bytes",
]
