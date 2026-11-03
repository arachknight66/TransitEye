"""Sector benchmark runner recording throughput, memory, disk footprint, and cache hits."""

from __future__ import annotations

import json
import logging
import math
import os
import resource
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from transiteye.config import AppConfig
from transiteye.execution.runner import BatchRunner
from transiteye.execution.store import JobStatus, JobStore
from transiteye.provenance import code_fingerprint

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


def _process_tree_rss_mb(root_pid: int) -> tuple[float, int]:
    """Sample summed Linux RSS of the coordinator and its live descendants."""
    pending, seen = [root_pid], set()
    total_kib = 0
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        try:
            status = Path(f"/proc/{pid}/status").read_text()
            for line in status.splitlines():
                if line.startswith("VmRSS:"):
                    total_kib += int(line.split()[1])
                    break
            children = Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
            pending.extend(int(child) for child in children)
        except (OSError, ValueError, IndexError):
            continue  # Processes may exit between procfs reads.
    return total_kib / 1024.0, len(seen)


def _all_tasks(store: JobStore, job_id: str) -> list[Any]:
    tasks = []
    while True:
        page = store.list_tasks(job_id, limit=1000, offset=len(tasks))
        tasks.extend(page)
        if len(page) < 1000:
            return tasks


def run_sector_benchmark(
    target_items: list[tuple[str, str]],  # (target_id, source_uri)
    output_dir: Path,
    device: str = "cpu",
    workers: int = 4,
    benchmark_id: str | None = None,
    job_config: dict[str, Any] | None = None,
    app_config: AppConfig | None = None,
    pause_after_finished: int | None = None,
    rss_sample_seconds: float = 0.25,
    report_context: dict[str, Any] | None = None,
) -> BenchmarkReport:
    """Run a bounded benchmark; sample process-tree RSS and count fresh successful work."""
    if not Path("/proc/self/status").is_file():
        raise RuntimeError("Process-tree memory sampling requires Linux procfs")
    if not math.isfinite(rss_sample_seconds) or rss_sample_seconds <= 0 or not target_items:
        raise ValueError("A positive sample interval and at least one target are required")
    if len({item[0] for item in target_items}) != len(target_items):
        raise ValueError("Benchmark target IDs must be unique")
    if pause_after_finished is not None and not 0 < pause_after_finished < len(target_items):
        raise ValueError("Pause threshold must be positive and smaller than the target count")
    out_dir = output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    b_id = benchmark_id or f"benchmark_{int(time.time())}"
    db_path = out_dir / "benchmark_state.db"
    store = JobStore(db_path)

    settings = job_config or {
        "device": device, "workers": workers, "max_candidates": 1,
        "snr_threshold": 6.0, "screen_snr_threshold": 50.0,
    }
    if settings.get("device", device) != device:
        raise ValueError("Benchmark job and runner devices must agree")
    try:
        initial = store.get_job(b_id)
        previous = _all_tasks(store, b_id)
        if json.loads(initial.config_json) != settings or {(t.target_id, t.source_uri) for t in previous} != set(target_items):
            raise ValueError("Existing benchmark job does not match the requested inputs and settings")
    except KeyError:
        initial = store.create_job(b_id, target_items, settings)

    cached_count = 0
    pause_requested = False

    def _prog(p):
        nonlocal cached_count, pause_requested
        if p.status == "cached":
            cached_count += 1
        if (pause_after_finished is not None and not pause_requested
                and p.completed_targets + p.failed_targets >= pause_after_finished):
            pause_requested = store.transition_job(b_id, (JobStatus.RUNNING,), JobStatus.PAUSED)

    runner = BatchRunner(
        store=store,
        config=app_config,
        max_workers=workers,
        device=device,
    )

    stop_sampling = threading.Event()
    samples: list[tuple[float, int]] = []

    def sample() -> None:
        while not stop_sampling.is_set():
            samples.append(_process_tree_rss_mb(os.getpid()))
            stop_sampling.wait(rss_sample_seconds)

    monitor = threading.Thread(target=sample, name="benchmark-rss", daemon=True)
    monitor.start()
    t_start = time.perf_counter()
    paused = None
    try:
        job = runner.run_job(b_id, progress_callback=_prog)
        if pause_requested and job.status is JobStatus.PAUSED:
            paused = {"status": job.status.value, "completed": job.completed_targets,
                      "failed": job.failed_targets, "total": job.total_targets}
            job = runner.run_job(b_id, progress_callback=_prog)
        t_end = time.perf_counter()
    finally:
        stop_sampling.set()
        monitor.join()

    elapsed = max(1e-4, t_end - t_start)
    fresh_completed = max(0, job.completed_targets - initial.completed_targets - cached_count)
    throughput = fresh_completed / elapsed
    peak_rss = max((rss for rss, _ in samples), default=0.0)
    if peak_rss <= 0:
        raise RuntimeError("Could not sample process-tree RSS; benchmark memory is unavailable")
    disk_mb = compute_dir_size_mb(out_dir)
    tasks = _all_tasks(store, b_id)
    failures = [{"target_id": task.target_id, "error": task.error_message,
                 "attempt_count": task.attempt_count} for task in tasks if task.status.value == "failed"]
    partial_failures = [{"target_id": task.target_id, "error": task.error_message}
                        for task in tasks if task.status.value == "completed" and task.error_message]
    analysis_statuses: dict[str, int] = {}
    bundle_read_errors = []
    candidate_flags: Counter[str] = Counter()
    physical_fit_failure_tasks = []
    for task in tasks:
        if task.status.value != "completed":
            continue
        try:
            if not task.bundle_path:
                raise ValueError("completed task has no result bundle")
            result = json.loads((Path(task.bundle_path) / "result.json").read_text())
            status = result["status"]
            analysis_statuses[status] = analysis_statuses.get(status, 0) + 1
            for candidate in result.get("candidates", []):
                flags = candidate.get("flags", [])
                candidate_flags.update(flags)
                if "physical_fit_failed" in flags:
                    physical_fit_failure_tasks.append(task.target_id)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            bundle_read_errors.append({"target_id": task.target_id, "error": f"{type(exc).__name__}: {exc}"})
    rep_file = out_dir / f"{b_id}_report.json"
    if rep_file.exists():
        rep_file = out_dir / f"{b_id}_{time.time_ns()}_report.json"

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
        details={"report_schema_version": "2", "status": job.status.value,
                 "code_sha256": code_fingerprint(), "job_config": settings,
                 "memory_metric": "sampled sum of coordinator and descendant RSS; shared pages may be counted multiple times",
                 "rss_sample_seconds": rss_sample_seconds, "rss_sample_count": len(samples),
                 "peak_process_count": max((count for _, count in samples), default=0),
                 "rss_limitation": "polling can miss short-lived peaks",
                 "elapsed_scope": "job execution including controlled pause and restart; excludes input preflight",
                 "preexisting_completed": initial.completed_targets,
                 "fresh_completed_excluding_cache": fresh_completed,
                 "paused_snapshot": paused, "failed_tasks": failures,
                 "candidate_failure_tasks": partial_failures,
                 "candidate_flag_counts": dict(candidate_flags),
                 "physical_fit_failure_tasks": physical_fit_failure_tasks,
                 "analysis_status_counts": analysis_statuses, "bundle_read_errors": bundle_read_errors,
                 "task_records": len(tasks), "unique_task_ids": len({task.task_id for task in tasks}),
                 "report_path": str(rep_file), "context": report_context or {}},
    )

    # Persist benchmark report
    temporary = rep_file.with_suffix(rep_file.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2, sort_keys=True)
    temporary.replace(rep_file)

    return report
