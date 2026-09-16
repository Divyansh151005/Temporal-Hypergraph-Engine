"""vLLM secondary KV tier over gRPC.

Out-of-process counterpart of ``hypergraph_offloader.HypergraphKVOffloader``:
vLLM's offloading connector hands evicted KV chunks to :meth:`submit_store`
and asks for them back during prefix matching with :meth:`submit_load`.

    vllm serve <model> --kv-transfer-config '{
      "kv_connector": "OffloadingConnector",
      "secondary_tier_module": "hypergraph_client.kv_tier.HypergraphKVOffloadTierManager",
      "secondary_tier_config": {"grpc_endpoint": "localhost:50051"}
    }'
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from typing import Any

from .client import HypergraphClient

try:
    from vllm.v1.kv_offload.tiering.base import BaseSecondaryTierManager as _TierBase
except ImportError:
    _TierBase = object

_HEX_KEY = re.compile(r"^[0-9a-fA-F]{64}$")

BlockHash = str | bytes | int


def block_key(block_hash: BlockHash) -> str:
    """The server's 64-hex-digit key for a vLLM block hash.

    32-byte digests and 64-digit hex strings map to themselves; any other
    hash (vLLM's integer hashes, shorter strings) is hashed with BLAKE2b-256.
    """
    if isinstance(block_hash, bytes):
        raw = block_hash if len(block_hash) == 32 else hashlib.blake2b(block_hash, digest_size=32).digest()
        return raw.hex()
    if isinstance(block_hash, str) and _HEX_KEY.match(block_hash):
        return block_hash.lower()
    return hashlib.blake2b(str(block_hash).encode(), digest_size=32).hexdigest()


class HypergraphKVOffloadTierManager(_TierBase):
    """Stores and reloads vLLM KV chunks in the engine's KV block store.

    ``config`` keys: ``grpc_endpoint`` (default ``$HYPERGRAPH_ENDPOINT`` or
    ``localhost:50051``), ``api_key`` (default ``$HYPERGRAPH_API_KEY``) and
    ``timeout`` in seconds.
    """

    def __init__(self, config: Mapping[str, Any] | None = None, *, client: HypergraphClient | None = None):
        config = dict(config or {})
        if _TierBase is not object:
            super().__init__(config)
        self.client = client or HypergraphClient(
            config.get("grpc_endpoint"), config.get("api_key"), timeout=float(config.get("timeout", 5.0))
        )

    def submit_store(
        self, block_hash: BlockHash, kv_chunk_bytes: bytes, metadata: Mapping[str, Any] | None = None
    ) -> bool:
        """Called when vLLM evicts an inactive KV chunk from GPU/host memory."""
        hyperedge_ids = (metadata or {}).get("associated_hyperedge_ids", ())
        return self.client.store_kv_block(block_key(block_hash), bytes(kv_chunk_bytes), hyperedge_ids)

    def submit_load(self, block_hash: BlockHash) -> bytes | None:
        """Called during prefix matching; ``None`` is a miss and vLLM recomputes the prefill."""
        return self.client.load_kv_block(block_key(block_hash))

    def close(self) -> None:
        self.client.close()
