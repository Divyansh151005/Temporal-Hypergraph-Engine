"""Type stubs for the ``hypergraph_engine_native`` extension module."""

from typing import Optional, TypedDict

__version__: str
KV_HEADER_LEN: int

# (num_layers, num_heads, head_dim, block_size, precision_id); precision 1=FP16, 2=BF16, 3=INT8
LayoutTuple = tuple[int, int, int, int, int]

class ScoredHyperedge(TypedDict):
    id: str
    event_type: Optional[str]
    participants: list[str]
    valid_from: int
    valid_to: Optional[int]
    is_ongoing: bool
    confidence: float
    score: float
    summary: Optional[str]
    embedding: Optional[bytes]

class EngineStats(TypedDict):
    nodes: int
    hyperedges: int
    last_sequence: int
    kv_blocks: int
    kv_records: int
    kv_segments: int
    kv_bytes_written: int
    kv_dedup_hits: int

class NativeHypergraphEngine:
    def __init__(self, storage_path: str, embedding_dim: int = 0, sync: str = "fsync") -> None: ...
    @staticmethod
    def in_memory(embedding_dim: int = 0) -> NativeHypergraphEngine: ...
    def insert_hyperedge_fast(
        self,
        event_type: str,
        node_labels: list[str],
        valid_from: int,
        valid_to: Optional[int],
        embedding_buffer: bytes,
        confidence: float = 1.0,
        summary: Optional[str] = None,
    ) -> str:
        """Raises ValueError when the buffer length is not a multiple of 4."""
    def query_temporal(
        self, query_bytes: bytes, target_ts: int, decay_lambda: float = 0.0, top_k: int = 10
    ) -> list[ScoredHyperedge]: ...
    def traverse(
        self,
        seed_labels: list[str],
        k_hops: int,
        as_of: int,
        query_bytes: Optional[bytes] = None,
        decay_lambda: float = 0.0,
        top_k: int = 10,
        chronological: bool = False,
        event_types: Optional[list[str]] = None,
        *,
        with_embeddings: bool = False,
    ) -> list[ScoredHyperedge]:
        """Hyperedges within ``k_hops`` of the seeds valid at ``as_of``; ``top_k=0`` returns all.

        ``embedding`` is ``None`` in results unless ``with_embeddings`` is set.
        """
    def soft_delete_hyperedge(self, hyperedge_id: str) -> None: ...
    def checkpoint(self) -> int: ...
    def stats(self) -> EngineStats: ...
    def store_kv_block_raw(
        self,
        block_hash: bytes,
        data_ptr: int,
        num_bytes: int,
        hyperedge_id: Optional[str] = None,
        *,
        layout: Optional[LayoutTuple] = None,
    ) -> bool:
        """Unsafe by contract: ``data_ptr`` must address ``num_bytes`` live, contiguous bytes."""
    def load_kv_block_raw(self, block_hash: bytes, target_ptr: int, target_len: int) -> bool:
        """Unsafe by contract: ``target_ptr`` must address ``target_len`` writable bytes."""
    def store_kv_block(
        self,
        block_hash: bytes,
        buffer: bytes | bytearray | memoryview,
        hyperedge_id: Optional[str] = None,
        *,
        layout: Optional[LayoutTuple] = None,
    ) -> bool: ...
    def load_kv_block(self, block_hash: bytes) -> Optional[bytes]: ...
    def has_kv_block(self, block_hash: bytes) -> bool: ...
    def kv_block_layout(self, block_hash: bytes) -> Optional[LayoutTuple]: ...
    def remove_kv_block(self, block_hash: bytes) -> bool: ...
    def plan_kv_evictions(self, decay_threshold: float, now: int) -> list[bytes]: ...

def blake3_token_hash(token_ids: bytes, parent: Optional[bytes] = None) -> bytes: ...
def blake3_content_hash(data: bytes) -> bytes: ...
