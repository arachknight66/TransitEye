"""Execution framework: compute device management, worker pools, and persistence."""

from transiteye.execution.compute import (
    ComputeDevice,
    configure_gpu_memory_limit,
    has_cuda,
    resolve_device,
)

__all__ = [
    "ComputeDevice",
    "configure_gpu_memory_limit",
    "has_cuda",
    "resolve_device",
]

