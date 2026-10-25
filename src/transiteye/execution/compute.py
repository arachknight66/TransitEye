"""Compute backend abstraction with CUDA GPU support and transparent CPU fallback."""

from __future__ import annotations

import logging
from enum import StrEnum

LOGGER = logging.getLogger("transiteye.execution.compute")


class ComputeDevice(StrEnum):
    """Execution device for compute-heavy routines."""

    AUTO = "auto"
    CUDA = "cuda"
    CPU = "cpu"


_CUDA_AVAILABLE: bool | None = None


def has_cuda() -> bool:
    """Check if CUDA GPU is available via CuPy without throwing exceptions."""
    global _CUDA_AVAILABLE
    if _CUDA_AVAILABLE is not None:
        return _CUDA_AVAILABLE
    try:
        import cupy as cp

        count = cp.cuda.runtime.getDeviceCount()
        _CUDA_AVAILABLE = count > 0
    except Exception:
        _CUDA_AVAILABLE = False
    return _CUDA_AVAILABLE


def resolve_device(preference: ComputeDevice | str = ComputeDevice.AUTO) -> str:
    """Resolve compute preference to either 'cuda' or 'cpu'."""
    pref = ComputeDevice(preference)
    if pref is ComputeDevice.CUDA:
        if not has_cuda():
            raise RuntimeError(
                "CUDA compute device requested, but no CUDA-capable device was detected."
            )
        return "cuda"
    if pref is ComputeDevice.CPU:
        return "cpu"
    # AUTO
    return "cuda" if has_cuda() else "cpu"


def configure_gpu_memory_limit(limit_mb: int = 3072) -> None:
    """Enforce a configurable memory pool ceiling so desktop sessions avoid VRAM exhaustion."""
    if not has_cuda():
        return
    try:
        import cupy as cp

        pool = cp.get_default_memory_pool()
        pool.set_limit(size=limit_mb * 1024 * 1024)
    except Exception as exc:
        LOGGER.warning("Could not set CuPy memory pool limit: %s", exc)
