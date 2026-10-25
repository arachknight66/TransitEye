"""Sector benchmark runner recording throughput, memory, disk footprint, and cache hits."""

from __future__ import annotations

import json
import logging
import resource
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from transiteye.execution.runner import BatchRunner
from transiteye.execution.store import JobStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    """Benchmark results tracking sector-scale throughput and resource footprint."""

    benchmark_id: str
    target_count: int
    completed_targets: int
    failed_targets: int
    cached_targets: int
    elapsed_seconds: float
    throughput_targets_per_sec: float
    peak_rss_mb: float
    disk_footprint_mb: float
    device: str
    worker_count: int
    details: dict[str, Any]


def get_peak_rss_mb() -> float:
    """Return peak resident memory usage in megabytes (Linux getrusage)."""
    # On Linux, ru_maxrss is in kilobytes
    usage_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(usage_kb / 1024.0)


def compute_dir_size_mb(directory: Path) -> float:
    """Compute total disk footprint of a directory in megabytes."""
    if not directory.is_dir():
        return 0.0
    total_bytes = sum(f.stat().st_size for f in directory.rglob("*") if f.is_file())
    return float(total_bytes / (1024.0 * 1024.0))


def run_sector_benchmark(
    target_items: list[tuple[str, str]],  # (target_id, source_uri)
    output_dir: Path,
    device: str = "cpu",
    workers: int = 4,
    benchmark_id: str | None = None,
) -> BenchmarkReport:
    """Execute a reproducible sector benchmark run and persist metrics."""
    out_dir = output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    b_id = benchmark_id or f"benchmark_{int(time.time())}"
    db_path = out_dir / "benchmark_state.db"
    store = JobStore(db_path)

    try:
        store.get_job(b_id)
    except KeyError:
        store.create_job(
            b_id,
            target_items,
            {
                "device": device,
                "workers": workers,
                "max_candidates": 1,
                "snr_threshold": 6.0,
                "screen_snr_threshold": 50.0,
            },
        )

    cached_count = 0

    def _prog(p):
        nonlocal cached_count
        if p.status == "cached":
            cached_count += 1

    runner = BatchRunner(
        store=store,
        max_workers=workers,
        device=device,
    )

    t_start = time.perf_counter()
    job = runner.run_job(b_id, progress_callback=_prog)
    t_end = time.perf_counter()

    elapsed = max(1e-4, t_end - t_start)
    throughput = job.total_targets / elapsed
    peak_rss = get_peak_rss_mb()
    disk_mb = compute_dir_size_mb(out_dir)

    report = BenchmarkReport(
        benchmark_id=b_id,
        target_count=job.total_targets,
        completed_targets=job.completed_targets,
        failed_targets=job.failed_targets,
        cached_targets=cached_count,
        elapsed_seconds=elapsed,
        throughput_targets_per_sec=throughput,
        peak_rss_mb=peak_rss,
        disk_footprint_mb=disk_mb,
        device=device,
        worker_count=workers,
        details={"status": job.status.value},
    )

    # Persist benchmark report
    rep_file = out_dir / f"{b_id}_report.json"
    with open(rep_file, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2, sort_keys=True)

    return report
