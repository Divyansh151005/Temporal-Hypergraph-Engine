"""vLLM KV-cache secondary storage tier backed by the Temporal Hypergraph Engine."""

from .connector import HypergraphKVOffloader
from .hardware import HardwareProfile, detect_device, detect_hardware
from .layout import LLAMA3_8B_FP16, PRECISION_IDS, KvLayout

__all__ = [
    "HardwareProfile",
    "HypergraphKVOffloader",
    "KvLayout",
    "LLAMA3_8B_FP16",
    "PRECISION_IDS",
    "detect_device",
    "detect_hardware",
]
