"""Persistent batch runner with bounded multiprocessing, GPU scheduling, and caching."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import traceback
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transiteye.config import AppConfig, default_config
from transiteye.execution.compute import (
    ComputeDevice,
    configure_gpu_memory_limit,
    has_cuda,
    resolve_device,
)
from transiteye.execution.service import analyze_light_curve
from transiteye.execution.store import JobRecord, JobStatus, JobStore, TargetTaskRecord
from transiteye.io.fits import read_tess_light_curve

# Prevent OpenMP and BLAS deadlocks across forked worker pools
for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_v] = "1"


def _worker_init() -> None:
    for v in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[v] = "1"


logger = logging.getLogger(__name__)


def compute_task_cache_key(
    input_sha256: str,
    config_sha256: str,
    model_sha256: str = "v1",
    schema_version: str = "2",
) -> str:
    """Compute SHA-256 cache key for identical input, config, and model versions."""
    token = f"{input_sha256}:{config_sha256}:{model_sha256}:{schema_version}".encode()
    return hashlib.sha256(token).hexdigest()


def _analyze_target_worker(
    target_id: str,
    source_path_str: str,
    config_dict: dict[str, Any],
    device: str,
    output_dir_str: str | None,
) -> dict[str, Any]:
    """Top-level worker function executing inside ProcessPoolExecutor child."""
    source_path = Path(source_path_str)
    output_dir = Path(output_dir_str) if output_dir_str else None

    if device == "cuda":
        gpu_limit = int(config_dict.get("gpu_memory_limit_mb", 2048))
        configure_gpu_memory_limit(gpu_limit)

    # Load delivered light curve
    lc = read_tess_light_curve(source_path)

    max_candidates = int(config_dict.get("max_candidates", 3))
    snr_threshold = float(config_dict.get("snr_threshold", 6.0))
    screen_snr_threshold = float(config_dict.get("screen_snr_threshold", 5.0))

    # Execute analysis service
    res, bundle_path = analyze_light_curve(
        light_curve=lc,
        max_candidates=max_candidates,
        snr_threshold=snr_threshold,
        screen_snr_threshold=screen_snr_threshold,
        device=device,
        output_dir=output_dir,
    )

    return {
        "target_id": target_id,
        "status": res.status.value,
        "candidate_count": len(res.candidates),
        "bundle_path": str(bundle_path),
        "input_sha256": lc.provenance.checksum_sha256 or "unknown",
        "analysis_id": res.analysis_id,
    }


@dataclass(frozen=True, slots=True)
class BatchProgress:
    """Progress event yielded during batch execution."""

    job_id: str
    target_id: str
    status: str
    completed_targets: int
    failed_targets: int
    total_targets: int
    percent: float
    candidate_count: int | None = None
    bundle_path: str | None = None
    error_message: str | None = None


class BatchRunner:
    """Coordinates persistent, resumable sector-scale analysis across CPU/GPU workers."""

    def __init__(
        self,
        store: JobStore,
        config: AppConfig | None = None,
        max_workers: int = 4,
        device: str = "auto",
        max_gpu_workers: int = 2,
    ) -> None:
        self.store = store
        self.config = config or default_config()
        self.max_workers = max_workers
        self.device = device
        self.max_gpu_workers = max_gpu_workers

    def run_job(
        self,
        job_id: str,
        progress_callback: Callable[[BatchProgress], None] | None = None,
    ) -> JobRecord:
        """Execute all pending tasks in a job until completion or cancellation."""
        job = self.store.get_job(job_id)
        if job.status in (JobStatus.COMPLETED, JobStatus.CANCELLED):
            logger.info("Job %s is already in terminal state %s", job_id, job.status)
            return job

        self.store.reset_interrupted_tasks(job_id)
        self.store.update_job_status(job_id, JobStatus.RUNNING)
        pending_tasks = self.store.get_pending_tasks(job_id)

        if not pending_tasks:
            self.store.update_job_status(job_id, JobStatus.COMPLETED)
            return self.store.get_job(job_id)

        cfg_data = json.loads(job.config_json)
        cfg_sha256 = hashlib.sha256(job.config_json.encode()).hexdigest()[:16]

        # Determine worker device policy
        resolved_device = resolve_device(self.device)
        is_cuda = resolved_device == ComputeDevice.CUDA and has_cuda()

        # Bounded ProcessPoolExecutor with max_tasks_per_child=25 to prevent memory leaks
        with ProcessPoolExecutor(
            max_workers=self.max_workers,
            max_tasks_per_child=25,
            initializer=_worker_init,
        ) as executor:
            futures: dict[Any, TargetTaskRecord] = {}

            for task in pending_tasks:
                # Check cancellation
                current_job = self.store.get_job(job_id)
                if current_job.status == JobStatus.CANCELLED:
                    logger.info("Job %s cancelled by user", job_id)
                    break

                self.store.mark_task_running(task.task_id)

                # Compute or reuse input_sha256 for cache lookup before spawning worker
                src_file = Path(task.source_uri.replace("file://", ""))
                inp_sha_lookup = task.input_sha256
                if not inp_sha_lookup and src_file.is_file():
                    try:
                        digest = hashlib.sha256()
                        with src_file.open("rb") as fh:
                            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                                digest.update(chunk)
                        inp_sha_lookup = digest.hexdigest()
                    except OSError:
                        inp_sha_lookup = None

                if inp_sha_lookup:
                    cache_key = compute_task_cache_key(inp_sha_lookup, cfg_sha256)
                    cached_bundle = self.store.get_cache(cache_key)
                    if cached_bundle and Path(cached_bundle).is_dir():
                        logger.info("Cache hit for target %s at %s", task.target_id, cached_bundle)
                        self.store.mark_task_completed(
                            task.task_id,
                            bundle_path=cached_bundle,
                            candidate_count=task.candidate_count or 1,
                            input_sha256=inp_sha_lookup,
                        )
                        if progress_callback:
                            j_rec = self.store.get_job(job_id)
                            pct = (j_rec.completed_targets / max(1, j_rec.total_targets)) * 100.0
                            progress_callback(
                                BatchProgress(
                                    job_id=job_id,
                                    target_id=task.target_id,
                                    status="cached",
                                    completed_targets=j_rec.completed_targets,
                                    failed_targets=j_rec.failed_targets,
                                    total_targets=j_rec.total_targets,
                                    percent=pct,
                                    candidate_count=task.candidate_count or 1,
                                    bundle_path=cached_bundle,
                                )
                            )
                        continue

                # Submit to worker pool with GPU worker cap & CPU spillover
                if is_cuda and self.max_gpu_workers > 0:
                    gpu_slot = len(futures) % max(1, self.max_workers)
                    worker_device = "cuda" if gpu_slot < self.max_gpu_workers else "cpu"
                else:
                    worker_device = "cpu"
                fut = executor.submit(
                    _analyze_target_worker,
                    task.target_id,
                    task.source_uri.replace("file://", ""),
                    cfg_data,
                    worker_device,
                    None,
                )
                futures[fut] = task

            target_timeout = float(cfg_data.get("target_timeout_seconds", 180.0))

            # Collect completed futures as they finish with per-target timeout
            for fut in as_completed(futures):
                task = futures[fut]
                try:
                    res_data = fut.result(timeout=target_timeout)
                    bundle_path = res_data["bundle_path"]
                    cand_count = res_data["candidate_count"]
                    inp_sha = res_data["input_sha256"]

                    self.store.mark_task_completed(
                        task.task_id,
                        bundle_path=bundle_path,
                        candidate_count=cand_count,
                        input_sha256=inp_sha,
                    )

                    # Store in cache
                    if inp_sha and inp_sha != "unknown":
                        cache_key = compute_task_cache_key(inp_sha, cfg_sha256)
                        self.store.put_cache(cache_key, bundle_path, task.target_id)

                    j_rec = self.store.get_job(job_id)
                    pct = (j_rec.completed_targets / max(1, j_rec.total_targets)) * 100.0

                    if progress_callback:
                        progress_callback(
                            BatchProgress(
                                job_id=job_id,
                                target_id=task.target_id,
                                status="completed",
                                completed_targets=j_rec.completed_targets,
                                failed_targets=j_rec.failed_targets,
                                total_targets=j_rec.total_targets,
                                percent=pct,
                                candidate_count=cand_count,
                                bundle_path=bundle_path,
                            )
                        )
                except Exception as exc:
                    err_msg = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
                    logger.error("Target %s failed: %s", task.target_id, exc)
                    self.store.mark_task_failed(task.task_id, err_msg)

                    j_rec = self.store.get_job(job_id)
                    pct = (
                        (j_rec.completed_targets + j_rec.failed_targets)
                        / max(1, j_rec.total_targets)
                    ) * 100.0

                    if progress_callback:
                        progress_callback(
                            BatchProgress(
                                job_id=job_id,
                                target_id=task.target_id,
                                status="failed",
                                completed_targets=j_rec.completed_targets,
                                failed_targets=j_rec.failed_targets,
                                total_targets=j_rec.total_targets,
                                percent=pct,
                                error_message=str(exc),
                            )
                        )

        final_job = self.store.get_job(job_id)
        if final_job.status != JobStatus.CANCELLED:
            if final_job.completed_targets + final_job.failed_targets >= final_job.total_targets:
                final_status = JobStatus.COMPLETED
            else:
                final_status = JobStatus.PAUSED
            self.store.update_job_status(job_id, final_status)

        return self.store.get_job(job_id)
