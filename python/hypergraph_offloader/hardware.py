"""Hardware detection: CUDA (production) vs. Apple Silicon / CPU (local development)."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class HardwareProfile:
    """Accelerators visible to this process."""

    device: str
    has_cuda: bool
    has_mps: bool

    @property
    def supports_pinned_memory(self) -> bool:
        """Page-locked host buffers only exist alongside a CUDA driver."""
        return self.has_cuda


def _mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def detect_hardware() -> HardwareProfile:
    """Probes CUDA first, then Apple Metal (MPS), falling back to CPU."""
    has_cuda = torch.cuda.is_available()
    has_mps = not has_cuda and _mps_available()
    device = "cuda" if has_cuda else "mps" if has_mps else "cpu"
    return HardwareProfile(device=device, has_cuda=has_cuda, has_mps=has_mps)


def detect_device() -> str:
    """``"cuda"``, ``"mps"`` or ``"cpu"``."""
    return detect_hardware().device
