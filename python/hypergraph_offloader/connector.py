"""vLLM KV-cache offloader backed by the native hypergraph engine.

On CUDA hosts blocks move GPU -> page-locked staging buffer -> engine (and
back). On Apple Silicon / CPU hosts tensors are made contiguous in host RAM
and their pointers are handed to the engine directly. Either way the native
call receives a raw pointer and copies the bytes exactly once.
"""

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, Optional, Sequence

import numpy as np
import torch

import hypergraph_engine_native as native

from .hardware import HardwareProfile, detect_hardware
from .layout import LLAMA3_8B_FP16, KvLayout


class HypergraphKVOffloader:
    """Secondary storage tier for PagedAttention KV blocks.

    Thread-safe: a lock serializes use of the shared staging buffer.
    """

    def __init__(
        self,
        rust_engine_handle: native.NativeHypergraphEngine,
        layout: KvLayout = LLAMA3_8B_FP16,
        device_id: int = 0,
        on_evict: Optional[Callable[[bytes], None]] = None,
        hardware: Optional[HardwareProfile] = None,
    ):
        self.engine = rust_engine_handle
        self.layout = layout
        self.device_id = device_id
        self.on_evict = on_evict
        self.hardware = hardware or detect_hardware()
        self.has_cuda = self.hardware.has_cuda
        self._lock = threading.Lock()
        self._prefetcher: Optional[ThreadPoolExecutor] = None
        self._staging = torch.empty(
            layout.shape,
            dtype=layout.dtype,
            pin_memory=self.hardware.supports_pinned_memory,
        )

    # ------------------------------------------------------------------ keys

    def hash_block_tokens(self, token_ids: Sequence[int], parent_hash: Optional[bytes] = None) -> bytes:
        """Deterministic BLAKE3 block key over ``parent_hash || int32(token_ids)``."""
        tokens = np.asarray(token_ids, dtype="<i4").tobytes()
        return native.blake3_token_hash(tokens, parent_hash)

    # ------------------------------------------------------------- offload

    def offload_kv_block(
        self,
        block_hash: bytes,
        key_tensor: torch.Tensor,
        value_tensor: torch.Tensor,
        hyperedge_id: Optional[str] = None,
    ) -> bool:
        """Stores a block given its key and value halves (each ``[L, B, H, D]``)."""
        self._check_half(key_tensor, "key_tensor")
        self._check_half(value_tensor, "value_tensor")
        with self._lock:
            self._staging[0].copy_(self._host_source(key_tensor), non_blocking=key_tensor.is_cuda)
            self._staging[1].copy_(self._host_source(value_tensor), non_blocking=value_tensor.is_cuda)
            self._synchronize(key_tensor, value_tensor)
            return self._store(block_hash, self._staging, hyperedge_id)

    def offload_kv_tensor(
        self,
        block_hash: bytes,
        kv_tensor: torch.Tensor,
        hyperedge_id: Optional[str] = None,
    ) -> bool:
        """Stores a combined ``[2, L, B, H, D]`` block. Contiguous CPU tensors
        are passed to the engine without any intermediate copy."""
        self._check_block(kv_tensor, "kv_tensor")
        if kv_tensor.device.type == "cpu" and kv_tensor.is_contiguous():
            return self._store(block_hash, kv_tensor, hyperedge_id)
        if not self.has_cuda:
            # Local development route (Apple Silicon / CPU): materialize a
            # contiguous host copy so the raw pointer covers the whole block.
            staged = kv_tensor.cpu().contiguous()
            return self._store(block_hash, staged, hyperedge_id)
        with self._lock:
            self._staging.copy_(kv_tensor, non_blocking=kv_tensor.is_cuda)
            self._synchronize(kv_tensor)
            return self._store(block_hash, self._staging, hyperedge_id)

    # ------------------------------------------------------------ prefetch

    def prefetch_kv_block(self, block_hash: bytes, target_tensor: torch.Tensor) -> bool:
        """Loads a block into ``target_tensor`` (``[2, L, B, H, D]``). Returns
        ``False`` on a miss, leaving the target untouched.

        Contiguous CPU targets are filled in place by the engine; other
        targets go through the staging buffer and the copy is synchronized
        before returning, so the staging buffer can be reused safely.
        """
        self._check_block(target_tensor, "target_tensor")
        if target_tensor.device.type == "cpu" and target_tensor.is_contiguous():
            return self.engine.load_kv_block_raw(block_hash, target_tensor.data_ptr(), self.layout.expected_bytes)
        with self._lock:
            if not self.engine.load_kv_block_raw(block_hash, self._staging.data_ptr(), self.layout.expected_bytes):
                return False
            target_tensor.copy_(self._staging, non_blocking=target_tensor.is_cuda)
            self._synchronize(target_tensor)
            return True

    def prefetch_kv_block_async(self, block_hash: bytes, target_tensor: torch.Tensor) -> Future[bool]:
        """Starts :meth:`prefetch_kv_block` on a background thread.

        Issue it as soon as a traversal names the block and wait on the future
        right before the attention kernel needs ``target_tensor``; the native
        load releases the GIL, so decoding continues meanwhile. Prefetches run
        in submission order.
        """
        self._check_block(target_tensor, "target_tensor")
        with self._lock:
            if self._prefetcher is None:
                self._prefetcher = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kv-prefetch")
            prefetcher = self._prefetcher
        return prefetcher.submit(self.prefetch_kv_block, block_hash, target_tensor)

    def close(self) -> None:
        """Waits for pending prefetches and stops the prefetch thread."""
        with self._lock:
            prefetcher, self._prefetcher = self._prefetcher, None
        if prefetcher is not None:
            prefetcher.shutdown(wait=True)

    # ------------------------------------------------------------ eviction

    def plan_evictions(self, decay_threshold: float, now: Optional[int] = None) -> list[bytes]:
        """Blocks whose hyperedges' decayed relevance fell below
        ``decay_threshold``, least relevant first. ``on_evict`` (the stand-in
        for vLLM's ``BlockSpaceManager.evict_to_secondary``) is invoked for
        each candidate."""
        timestamp = int(time.time()) if now is None else now
        candidates = self.engine.plan_kv_evictions(decay_threshold, timestamp)
        if self.on_evict is not None:
            for block_hash in candidates:
                self.on_evict(block_hash)
        return candidates

    def has_block(self, block_hash: bytes) -> bool:
        return self.engine.has_kv_block(block_hash)

    # ------------------------------------------------------------- helpers

    def _store(self, block_hash: bytes, staged: torch.Tensor, hyperedge_id: Optional[str]) -> bool:
        if staged.device.type != "cpu" or not staged.is_contiguous():
            raise RuntimeError("staged KV tensor must be contiguous host memory")
        num_bytes = staged.numel() * staged.element_size()
        if num_bytes != self.layout.expected_bytes:
            raise ValueError(f"KV block is {num_bytes} bytes but the layout requires {self.layout.expected_bytes}")
        return self.engine.store_kv_block_raw(
            block_hash,
            staged.data_ptr(),
            num_bytes,
            hyperedge_id,
            layout=self.layout.as_native(),
        )

    @staticmethod
    def _host_source(tensor: torch.Tensor) -> torch.Tensor:
        # torch 2.1 MPS: copy_ from a strided/offset MPS view into host memory
        # ignores the view's storage offset; .cpu() materializes it correctly.
        if tensor.device.type in ("cpu", "cuda"):
            return tensor
        return tensor.cpu()

    def _synchronize(self, *tensors: torch.Tensor) -> None:
        if any(t.device.type == "mps" for t in tensors):
            torch.mps.synchronize()
        if not self.has_cuda:
            return
        for device in {t.device for t in tensors if t.is_cuda}:
            torch.cuda.current_stream(device).synchronize()

    def _check_half(self, tensor: torch.Tensor, name: str) -> None:
        if tuple(tensor.shape) != self.layout.kv_shape or tensor.dtype != self.layout.dtype:
            raise ValueError(
                f"{name} must be {self.layout.dtype} with shape {self.layout.kv_shape}, "
                f"got {tensor.dtype} {tuple(tensor.shape)}"
            )

    def _check_block(self, tensor: torch.Tensor, name: str) -> None:
        if tuple(tensor.shape) != self.layout.shape or tensor.dtype != self.layout.dtype:
            raise ValueError(
                f"{name} must be {self.layout.dtype} with shape {self.layout.shape}, "
                f"got {tensor.dtype} {tuple(tensor.shape)}"
            )
