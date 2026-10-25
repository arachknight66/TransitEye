"""Tests for SQLite WAL state store, ProcessPool runner, and persistent batch execution."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits

from transiteye.execution.benchmark import BenchmarkReport, run_sector_benchmark
from transiteye.execution.runner import BatchRunner
from transiteye.execution.store import JobRecord, JobStatus, JobStore


def _create_mock_fits(path: Path, target_id: str = "12345678", valid: bool = True) -> Path:
    """Helper to write a valid or invalid test FITS file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not valid:
        path.write_bytes(b"corrupt")
        return path

    t = np.linspace(1.0, 10.0, 500)
    f = np.ones_like(t)
    f[200:220] = 0.99
    cols = [
        fits.Column(name="TIME", format="D", array=t),
        fits.Column(name="PDCSAP_FLUX", format="D", array=f),
        fits.Column(name="PDCSAP_FLUX_ERR", format="D", array=np.full_like(t, 0.001)),
        fits.Column(name="QUALITY", format="K", array=np.zeros(len(t), dtype=np.int64)),
        fits.Column(name="CADENCENO", format="K", array=np.arange(len(t), dtype=np.int64)),
        fits.Column(name="MOM_CENTR1", format="D", array=np.zeros(len(t))),
        fits.Column(name="MOM_CENTR2", format="D", array=np.zeros(len(t))),
    ]
    table = fits.BinTableHDU.from_columns(cols)
    primary = fits.PrimaryHDU()
    primary.header["TICID"] = int(target_id)
    fits.HDUList([primary, table]).writeto(path, overwrite=True)
    return path


def test_job_store_lifecycle_and_wal(tmp_path: Path) -> None:
    """Verify JobStore creates WAL SQLite DB, registers jobs, and updates task states."""
    db_path = tmp_path / "state.db"
    store = JobStore(db_path)

    targets = [("t1", "file:///t1.fits"), ("t2", "file:///t2.fits")]
    job = store.create_job("job_01", targets, {"max_candidates": 1})

    assert isinstance(job, JobRecord)
    assert job.status == JobStatus.PENDING
    assert job.total_targets == 2
    assert job.completed_targets == 0

    pending = store.get_pending_tasks("job_01")
    assert len(pending) == 2

    # Mark t1 running then completed
    store.mark_task_running("job_01:t1")
    store.mark_task_completed("job_01:t1", bundle_path="/results/t1", candidate_count=1)

    updated_job = store.get_job("job_01")
    assert updated_job.completed_targets == 1

    # Mark t2 failed
    store.mark_task_running("job_01:t2")
    store.mark_task_failed("job_01:t2", "Simulated error")

    updated_job = store.get_job("job_01")
    assert updated_job.failed_targets == 1

    # Cache put and get
    store.put_cache("cache_abc", "/results/t1", "t1")
    assert store.get_cache("cache_abc") == "/results/t1"
    assert store.get_cache("cache_nonexistent") is None


def test_batch_runner_execution_and_caching(tmp_path: Path) -> None:
    """Verify BatchRunner processes targets, populates cache, and supports rerun."""
    db_path = tmp_path / "batch_state.db"
    store = JobStore(db_path)

    fits1 = _create_mock_fits(tmp_path / "fits" / "target1_lc.fits", "111111")
    fits2 = _create_mock_fits(tmp_path / "fits" / "target2_lc.fits", "222222")

    targets = [
        ("111111", f"file://{fits1}"),
        ("222222", f"file://{fits2}"),
    ]

    job_id = "test_batch_01"
    store.create_job(job_id, targets, {"max_candidates": 1})

    progress_events = []

    def _on_prog(p):
        progress_events.append(p)

    runner = BatchRunner(store=store, max_workers=2, device="cpu")
    final_job = runner.run_job(job_id, progress_callback=_on_prog)

    assert final_job.status == JobStatus.COMPLETED
    assert final_job.completed_targets == 2
    assert final_job.failed_targets == 0
    assert len(progress_events) == 2

    # Re-run should find tasks already completed
    rerun_job = runner.run_job(job_id)
    assert rerun_job.status == JobStatus.COMPLETED


def test_batch_runner_resilience_to_failed_targets(tmp_path: Path) -> None:
    """Verify corrupted target fails gracefully without terminating valid targets."""
    db_path = tmp_path / "resilience_state.db"
    store = JobStore(db_path)

    valid_fits = _create_mock_fits(tmp_path / "good" / "good_lc.fits", "1001", valid=True)
    bad_fits = _create_mock_fits(tmp_path / "bad" / "bad_lc.fits", "1002", valid=False)

    targets = [
        ("1001", f"file://{valid_fits}"),
        ("1002", f"file://{bad_fits}"),
    ]

    job_id = "resilience_job"
    store.create_job(job_id, targets, {"max_candidates": 1})

    runner = BatchRunner(store=store, max_workers=2, device="cpu")
    final_job = runner.run_job(job_id)

    assert final_job.status == JobStatus.COMPLETED
    assert final_job.completed_targets == 1
    assert final_job.failed_targets == 1


def test_sector_benchmark_execution(tmp_path: Path) -> None:
    """Verify sector benchmark runner records throughput, peak memory, and persists JSON."""
    fits1 = _create_mock_fits(tmp_path / "bench_fits" / "target1.fits", "7001")
    fits2 = _create_mock_fits(tmp_path / "bench_fits" / "target2.fits", "7002")

    targets = [
        ("7001", f"file://{fits1}"),
        ("7002", f"file://{fits2}"),
    ]

    out_dir = tmp_path / "benchmark_run"
    report = run_sector_benchmark(
        target_items=targets,
        output_dir=out_dir,
        device="cpu",
        workers=2,
        benchmark_id="test_bench_run",
    )

    assert isinstance(report, BenchmarkReport)
    assert report.benchmark_id == "test_bench_run"
    assert report.target_count == 2
    assert report.completed_targets == 2
    assert report.failed_targets == 0
    assert report.throughput_targets_per_sec > 0
    assert report.peak_rss_mb > 0

    json_report = out_dir / "test_bench_run_report.json"
    assert json_report.exists()

