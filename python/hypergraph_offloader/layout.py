"""PagedAttention KV block geometry."""

from dataclasses import dataclass

import torch

PRECISION_IDS: dict[torch.dtype, int] = {
    torch.float16: 1,
    torch.bfloat16: 2,
    torch.int8: 3,
}


@dataclass(frozen=True)
class KvLayout:
    """A block tensor of shape ``[2, num_layers, block_size, num_heads, head_dim]``
    where index 0 holds keys and index 1 holds values."""

    num_layers: int = 32
    block_size: int = 16
    num_heads: int = 32
    head_dim: int = 128
    dtype: torch.dtype = torch.float16

    def __post_init__(self) -> None:
        if self.dtype not in PRECISION_IDS:
            raise ValueError(f"unsupported KV dtype {self.dtype}; expected one of {list(PRECISION_IDS)}")

    @property
    def shape(self) -> tuple[int, int, int, int, int]:
        """Combined K/V block shape."""
        return (2, *self.kv_shape)

    @property
    def kv_shape(self) -> tuple[int, int, int, int]:
        """Shape of the key (or value) half."""
        return (self.num_layers, self.block_size, self.num_heads, self.head_dim)

    @property
    def precision_id(self) -> int:
        return PRECISION_IDS[self.dtype]

    @property
    def element_size(self) -> int:
        return torch.empty((), dtype=self.dtype).element_size()

    @property
    def expected_bytes(self) -> int:
        """``2 * L * B * H * D * sizeof(dtype)``."""
        count = 1
        for dim in self.shape:
            count *= dim
        return count * self.element_size

    def as_native(self) -> tuple[int, int, int, int, int]:
        """Layout tuple in the native module's order:
        ``(num_layers, num_heads, head_dim, block_size, precision_id)``."""
        return (self.num_layers, self.num_heads, self.head_dim, self.block_size, self.precision_id)

    @classmethod
    def from_tensor(cls, kv: torch.Tensor) -> "KvLayout":
        """Infers the layout of a combined ``[2, L, B, H, D]`` block tensor."""
        if kv.dim() != 5 or kv.shape[0] != 2:
            raise ValueError(f"expected a [2, L, B, H, D] tensor, got shape {tuple(kv.shape)}")
        _, layers, block_size, heads, head_dim = kv.shape
        return cls(layers, block_size, heads, head_dim, kv.dtype)


LLAMA3_8B_FP16 = KvLayout()
"""Llama-3-8B, 16-token FP16 blocks: exactly 8,388,608 bytes."""
