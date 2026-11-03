"""Persistent batch runner with bounded multiprocessing, GPU scheduling, and caching."""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import multiprocessing
import os
import threading
import time
import traceback
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing.connection import Connection, wait
from pathlib import Path
from typing import Any

from transiteye.classification.models import ModelBundle
from transiteye.config import AppConfig, default_config
from transiteye.domain.models import AnalysisSettings
from transiteye.execution.compute import (
    ComputeDevice,
    configure_gpu_memory_limit,
    has_cuda,
    resolve_device,
)
from transiteye.execution.service import analyze_light_curve
from transiteye.execution.store import JobRecord, JobStatus, JobStore, TargetTaskRecord
from transiteye.io.fits import read_tess_light_curve
from transiteye.provenance import code_fingerprint, model_fingerprint
from transiteye.reporting.bundles import read_result_bundle, verify_bundle_integrity

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


def _model_fingerprint(config: dict[str, Any]) -> str:
    """Fingerprint actual selected model files; no classifier has a distinct identity."""
    model_path = config.get("model_dir")
    if not model_path:
        return "no-classifier"
    return model_fingerprint(Path(model_path).expanduser().resolve())


def _analyze_target_worker(
    target_id: str,
    source_path_str: str,
    config_dict: dict[str, Any],
    device: str,
    output_dir_str: str | None,
    app_config: AppConfig,
) -> dict[str, Any]:
    """Analyze one target in an independently supervised child process."""
    source_path = Path(source_path_str)
    output_dir = Path(output_dir_str) if output_dir_str else None
    expected = config_dict.get("expected_checksums", {}).get(target_id)
    if expected:
        digest = hashlib.sha256()
        with source_path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError(f"Input checksum mismatch for {target_id}.")

    if device == "cuda":
        gpu_limit = int(config_dict.get("gpu_memory_limit_mb", 2048))
        configure_gpu_memory_limit(gpu_limit)

    # Load delivered light curve
    lc = read_tess_light_curve(
        source_path, flux_preference=str(config_dict.get("flux", "PDCSAP_FLUX"))
    )

    max_candidates = int(config_dict.get("max_candidates", 3))
    snr_threshold = float(config_dict.get("snr_threshold", 6.0))
    screen_snr_threshold = float(config_dict.get("screen_snr_threshold", 5.0))
    model_dir = config_dict.get("model_dir")
    exposure_seconds = config_dict.get("exposure_seconds")
    configurable_keys = (
        "baseline_window_days", "bootstrap_replicates", "quality_mask", "gap_threshold_hours",
        "reject_positive_outliers_sigma", "bls_min_period_days", "bls_max_period_days",
        "bls_min_duration_hours", "bls_max_duration_hours", "bls_num_durations",
        "bls_frequency_factor",
    )
    options = {key: config_dict[key] for key in configurable_keys if key in config_dict}
    model_bundle = (
        ModelBundle.load(
            Path(model_dir),
            expected_calibrator_sha256=config_dict.get("model_calibrator_sha256"),
            require_verified_calibrator=True,
        )
        if model_dir else None
    )

    # Execute analysis service
    res, bundle_path = analyze_light_curve(
        light_curve=lc,
        config=app_config,
        analysis_settings=AnalysisSettings(
            max_candidates=max_candidates, snr_threshold=snr_threshold,
            screen_snr_threshold=screen_snr_threshold,
            device=str(config_dict.get("device", device)),
            trapezoid_exposure_days=(float(exposure_seconds) / 86400.0 if exposure_seconds is not None else None),
            **options,
        ),
        execution_device=device,
        model_bundle=model_bundle,
        output_dir=output_dir,
    )

    return {
        "target_id": target_id,
        "status": res.status.value,
        "candidate_count": len(res.candidates),
        "bundle_path": str(bundle_path),
        "input_sha256": lc.provenance.checksum_sha256 or "unknown",
        "analysis_id": res.analysis_id,
        "candidate_errors": dict(res.candidate_errors),
    }


def _worker_process(
    connection: Connection,
    target_id: str,
    source_path: str,
    config: dict[str, Any],
    device: str,
    app_config: AppConfig,
    lease_path: str,
) -> None:
    """Send one terminal result to the coordinator and never write job state."""
    parent = multiprocessing.parent_process()
    with Path(lease_path).open("a", encoding="utf-8") as lease:
        fcntl.flock(lease, fcntl.LOCK_SH)
        if parent is None or not parent.is_alive():
            connection.close()
            return

        def watch_parent() -> None:
            wait([parent.sentinel])
            os._exit(70)  # The isolated worker must not outlive its coordinator.

        threading.Thread(target=watch_parent, daemon=True, name="coordinator-watchdog").start()
        if not parent.is_alive():
            return
        try:
            _worker_init()
            result = _analyze_target_worker(target_id, source_path, config, device, None, app_config)
            connection.send({"result": result})
        except Exception as exc:
            connection.send({"error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"})
        finally:
            connection.close()


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


@dataclass(slots=True)
class _ActiveTask:
    task: TargetTaskRecord
    process: Any
    connection: Connection
    started_at: float
    device: str
    cache_key: str | None


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
        if max_workers < 1 or max_gpu_workers < 0:
            raise ValueError("max_workers must be positive and max_gpu_workers nonnegative.")
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
        """Execute a job with one coordinator and bounded, individually supervised workers."""
        lock_name = hashlib.sha256(job_id.encode()).hexdigest()
        lock_path = self.store.db_path.parent / f".{self.store.db_path.name}.{lock_name}.lock"
        lease_path = lock_path.with_suffix(".workers.lock")
        with lock_path.open("a", encoding="utf-8") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"Job {job_id} already has an active coordinator.") from exc
            try:
                with lease_path.open("a", encoding="utf-8") as lease:
                    try:
                        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError as exc:
                        raise RuntimeError(
                            f"Workers from an earlier coordinator for job {job_id} are still stopping; retry shortly."
                        ) from exc
                    fcntl.flock(lease, fcntl.LOCK_UN)
                return self._run_locked(job_id, progress_callback, lease_path)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _run_locked(
        self,
        job_id: str,
        progress_callback: Callable[[BatchProgress], None] | None,
        lease_path: Path,
    ) -> JobRecord:
        job = self.store.get_job(job_id)
        if job.status in (JobStatus.COMPLETED, JobStatus.CANCELLED):
            return job

        config = json.loads(job.config_json)
        timeout = float(config.get("target_timeout_seconds", 180.0))
        if not 0 < timeout < float("inf"):
            raise ValueError("target_timeout_seconds must be finite and positive.")
        cfg_hash = hashlib.sha256(job.config_json.encode()).hexdigest()
        model_hash = _model_fingerprint(config)
        code_hash = code_fingerprint()
        resolved_device = resolve_device(self.device)
        use_gpu = resolved_device == ComputeDevice.CUDA and has_cuda()
        code_hash = f"{code_hash}:{resolved_device}:gpu-workers={self.max_gpu_workers}"
        context = multiprocessing.get_context("spawn")
        active: dict[str, _ActiveTask] = {}

        def progress(
            task: TargetTaskRecord, status: str, *, count: int | None = None,
            bundle: str | None = None, error: str | None = None,
        ) -> None:
            if progress_callback is None:
                return
            current = self.store.get_job(job_id)
            finished = current.completed_targets + current.failed_targets
            progress_callback(BatchProgress(
                job_id=job_id, target_id=task.target_id, status=status,
                completed_targets=current.completed_targets, failed_targets=current.failed_targets,
                total_targets=current.total_targets,
                percent=100.0 * finished / max(1, current.total_targets),
                candidate_count=count, bundle_path=bundle, error_message=error,
            ))

        def stop(child: _ActiveTask) -> None:
            child.process.join(timeout=0.1)
            if child.process.is_alive():
                child.process.terminate()
                child.process.join(timeout=0.5)
            if child.process.is_alive():
                child.process.kill()
            child.process.join(timeout=1.0)
            child.connection.close()

        def cached_result(task: TargetTaskRecord, worker_device: str) -> tuple[str | None, str | None, int | None]:
            source = Path(task.source_uri.replace("file://", ""))
            if not source.is_file():
                return None, None, None
            try:
                digest = hashlib.sha256()
                with source.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                input_hash = digest.hexdigest()
            except OSError:
                return None, None, None
            key = compute_task_cache_key(
                input_hash, cfg_hash, model_hash, f"3:{code_hash}:worker={worker_device}"
            )
            expected = config.get("expected_checksums", {}).get(task.target_id)
            if expected and expected != input_hash:
                return key, None, None
            bundle_path = self.store.get_cache(key)
            if not bundle_path or not verify_bundle_integrity(Path(bundle_path)):
                return key, None, None
            try:
                loaded = read_result_bundle(Path(bundle_path))
                doc = loaded["result"]
                count = loaded["candidates"].num_rows
                valid = (
                    doc["provenance"]["checksum_sha256"] == input_hash
                    and doc["schema_version"] == "3"
                    and len(doc["candidates"]) == count
                    and not doc.get("candidate_errors")
                    and doc["status"] in {"complete", "no_significant_signal", "insufficient_data"}
                )
                if valid:
                    return key, bundle_path, count
            except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
                logger.warning("Discarding incomplete cached bundle %s", bundle_path)
            return key, None, None

        if not self.store.transition_job(
            job_id, (JobStatus.PENDING, JobStatus.PAUSED, JobStatus.RUNNING, JobStatus.FAILED),
            JobStatus.RUNNING,
        ):
            return self.store.get_job(job_id)
        self.store.reset_interrupted_tasks(job_id)
        try:
            while True:
                state = self.store.get_job(job_id).status
                if state is JobStatus.CANCELLED:
                    for child in active.values():
                        stop(child)
                    active.clear()
                    self.store.reset_interrupted_tasks(job_id)
                    return self.store.get_job(job_id)

                if state is JobStatus.RUNNING:
                    while len(active) < self.max_workers:
                        if self.store.get_job(job_id).status is not JobStatus.RUNNING:
                            break
                        pending = self.store.get_pending_tasks(job_id, limit=1)
                        if not pending:
                            break
                        task = pending[0]
                        self.store.mark_task_running(task.task_id)
                        gpu_active = sum(item.device == "cuda" for item in active.values())
                        worker_device = (
                            "cuda" if use_gpu and gpu_active < self.max_gpu_workers else "cpu"
                        )
                        cache_key, cached_path, cached_count = cached_result(task, worker_device)
                        if cached_path is not None and cached_count is not None:
                            source = Path(task.source_uri.replace("file://", ""))
                            try:
                                digest = hashlib.sha256()
                                with source.open("rb") as stream:
                                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                                        digest.update(block)
                                current_hash = digest.hexdigest()
                                current_key = compute_task_cache_key(
                                    current_hash, cfg_hash, model_hash,
                                    f"3:{code_hash}:worker={worker_device}",
                                )
                                if current_key == cache_key:
                                    self.store.mark_task_completed(
                                        task.task_id, cached_path, cached_count, current_hash
                                    )
                                    progress(task, "cached", count=cached_count, bundle=cached_path)
                                    continue
                            except OSError:
                                pass

                        receiving, sending = context.Pipe(duplex=False)
                        process = context.Process(
                            target=_worker_process,
                            args=(sending, task.target_id, task.source_uri.replace("file://", ""),
                                  config, worker_device, self.config, str(lease_path)),
                        )
                        try:
                            process.start()
                        except Exception as exc:
                            receiving.close()
                            sending.close()
                            self.store.mark_task_failed(task.task_id, f"Worker start failed: {exc}")
                            progress(task, "failed", error=f"Worker start failed: {exc}")
                            continue
                        sending.close()
                        active[task.task_id] = _ActiveTask(
                            task, process, receiving, time.monotonic(), worker_device, cache_key
                        )

                for task_id, child in list(active.items()):
                    payload: dict[str, Any] | None = None
                    if child.connection.poll():
                        try:
                            payload = child.connection.recv()
                        except EOFError:
                            payload = {"error": f"Worker exited without a result (exit code {child.process.exitcode})."}
                    elif not child.process.is_alive():
                        payload = {"error": f"Worker exited without a result (exit code {child.process.exitcode})."}
                    elif time.monotonic() - child.started_at >= timeout:
                        payload = {"error": f"Target exceeded {timeout:g} second timeout."}
                    if payload is None:
                        continue
                    stop(child)
                    del active[task_id]
                    if "error" in payload:
                        error = str(payload["error"])
                        self.store.mark_task_failed(task_id, error)
                        progress(child.task, "failed", error=error)
                        continue
                    result = payload["result"]
                    self.store.mark_task_completed(
                        task_id, result["bundle_path"], result["candidate_count"], result["input_sha256"],
                        json.dumps(result["candidate_errors"]) if result["candidate_errors"] else None,
                    )
                    if (
                        child.cache_key is not None
                        and result["input_sha256"] != "unknown"
                        and not result["candidate_errors"]
                    ):
                        actual_key = compute_task_cache_key(
                            result["input_sha256"], cfg_hash, model_hash,
                            f"3:{code_hash}:worker={child.device}",
                        )
                        if actual_key == child.cache_key:
                            self.store.put_cache(actual_key, result["bundle_path"], child.task.target_id)
                    progress(
                        child.task, "completed", count=result["candidate_count"],
                        bundle=result["bundle_path"],
                        error=json.dumps(result["candidate_errors"]) if result["candidate_errors"] else None,
                    )

                state = self.store.get_job(job_id).status
                if not active and state is JobStatus.PAUSED:
                    return self.store.get_job(job_id)
                if not active and state is JobStatus.RUNNING and not self.store.get_pending_tasks(job_id, limit=1):
                    self.store.transition_job(job_id, (JobStatus.RUNNING,), JobStatus.COMPLETED)
                    return self.store.get_job(job_id)
                if active:
                    time.sleep(0.05)
        except BaseException:
            if self.store.get_job(job_id).status is JobStatus.RUNNING:
                self.store.transition_job(job_id, (JobStatus.RUNNING,), JobStatus.PAUSED)
            raise
        finally:
            for child in active.values():
                stop(child)
            self.store.reset_interrupted_tasks(job_id)
