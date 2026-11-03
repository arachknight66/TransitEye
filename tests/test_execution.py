"""Tests for SQLite WAL state store, ProcessPool runner, and persistent batch execution."""

from __future__ import annotations

import fcntl
import hashlib
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from transiteye.config import load_config
from transiteye.execution.benchmark import BenchmarkReport, run_sector_benchmark
from transiteye.execution.runner import BatchRunner
from transiteye.execution.store import JobRecord, JobStatus, JobStore, TaskStatus


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


def test_task_accounting_is_idempotent_and_retry_preserves_success(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "state.db")
    store.create_job("job:with:colons", [("good", "a"), ("bad", "b")], {})
    store.mark_task_completed("job:with:colons:good", "bundle", 1)
    store.mark_task_completed("job:with:colons:good", "bundle", 1)
    store.mark_task_failed("job:with:colons:bad", "failure")
    store.mark_task_failed("job:with:colons:bad", "failure")
    store.update_job_status("job:with:colons", JobStatus.COMPLETED)
    assert (store.get_job("job:with:colons").completed_targets, store.get_job("job:with:colons").failed_targets) == (1, 1)
    assert store.retry_failed_tasks("job:with:colons") == 1
    assert store.get_job("job:with:colons").status is JobStatus.PAUSED
    assert store.get_job("job:with:colons").failed_targets == 0
    assert [t.target_id for t in store.get_pending_tasks("job:with:colons")] == ["bad"]
    assert len(store.list_tasks("job:with:colons", limit=1, offset=1)) == 1
    with pytest.raises(ValueError):
        store.list_tasks("job:with:colons", limit=1001)


def test_worker_timeout_terminates_and_counts_failure(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "state.db")
    source = _create_mock_fits(tmp_path / "target.fits")
    store.create_job("timeout", [("target", str(source))], {"target_timeout_seconds": 0.01})
    events = []
    result = BatchRunner(store, max_workers=1, device="cpu").run_job("timeout", events.append)
    assert (result.completed_targets, result.failed_targets) == (0, 1)
    assert "timeout" in events[0].error_message


def test_pause_drains_bounded_work_and_resume_finishes(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "state.db")
    source = _create_mock_fits(tmp_path / "target.fits")
    store.create_job("pause", [(str(i), str(source)) for i in range(3)], {"max_candidates": 1})
    runner = BatchRunner(store, max_workers=1, device="cpu")
    outcome = []
    worker = threading.Thread(target=lambda: outcome.append(runner.run_job("pause")))
    worker.start()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        with store._get_connection() as conn:
            running = conn.execute("SELECT count(*) FROM target_tasks WHERE status = ?", (TaskStatus.RUNNING.value,)).fetchone()[0]
        if running:
            break
        time.sleep(0.02)
    assert running == 1
    assert store.transition_job("pause", (JobStatus.RUNNING,), JobStatus.PAUSED)
    worker.join(timeout=30)
    assert not worker.is_alive()
    assert outcome[0].status is JobStatus.PAUSED
    assert outcome[0].completed_targets == 1
    assert len(store.get_pending_tasks("pause")) == 2
    final = runner.run_job("pause")
    assert (final.status, final.completed_targets, final.failed_targets) == (JobStatus.COMPLETED, 3, 0)


def test_manifest_checksum_mismatch_fails_target(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "state.db")
    source = _create_mock_fits(tmp_path / "target.fits")
    store.create_job("checksum", [("target", str(source))], {"expected_checksums": {"target": "0" * 64}})
    events = []
    result = BatchRunner(store, max_workers=1, device="cpu").run_job("checksum", events.append)
    assert result.failed_targets == 1
    assert "checksum mismatch" in events[0].error_message


def test_cancel_stops_worker_and_rejects_second_coordinator(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "state.db")
    source = _create_mock_fits(tmp_path / "target.fits")
    store.create_job("cancel", [("target", str(source))], {})
    runner = BatchRunner(store, max_workers=1, device="cpu")
    outcome = []
    worker = threading.Thread(target=lambda: outcome.append(runner.run_job("cancel")))
    worker.start()
    deadline = time.monotonic() + 10
    running = False
    while time.monotonic() < deadline:
        with store._get_connection() as conn:
            running = bool(conn.execute("SELECT count(*) FROM target_tasks WHERE status = ?", (TaskStatus.RUNNING.value,)).fetchone()[0])
        if running:
            break
        time.sleep(0.02)
    assert running
    with pytest.raises(RuntimeError, match="active coordinator"):
        runner.run_job("cancel")
    assert store.transition_job("cancel", (JobStatus.RUNNING,), JobStatus.CANCELLED)
    worker.join(timeout=10)
    assert not worker.is_alive()
    assert outcome[0].status is JobStatus.CANCELLED
    assert len(store.get_pending_tasks("cancel")) == 1
    assert runner.run_job("cancel").status is JobStatus.CANCELLED


def test_sigkill_coordinator_stops_worker_and_resumes_once(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"
    workspace = tmp_path / "workspace"
    config_file.write_text(f'[workspace]\nroot = "{workspace}"\n', encoding="utf-8")
    store = JobStore(workspace / "state.db")
    source = _create_mock_fits(tmp_path / "target.fits")
    store.create_job("crash", [("target", str(source))], {"max_candidates": 1})
    script = (
        "from pathlib import Path; from transiteye.config import load_config; "
        "from transiteye.execution.runner import BatchRunner; from transiteye.execution.store import JobStore; "
        "import sys; cfg=load_config(Path(sys.argv[1])); "
        "BatchRunner(JobStore(cfg.workspace.root / 'state.db'), cfg, max_workers=1, device='cpu').run_job('crash')"
    )
    coordinator = subprocess.Popen([sys.executable, "-c", script, str(config_file)])
    lease_path = workspace / f".state.db.{hashlib.sha256(b'crash').hexdigest()}.workers.lock"
    try:
        deadline = time.monotonic() + 20
        worker_has_lease = False
        while time.monotonic() < deadline and coordinator.poll() is None:
            if lease_path.exists():
                with lease_path.open("a") as lease:
                    try:
                        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        worker_has_lease = True
                        break
                    else:
                        fcntl.flock(lease, fcntl.LOCK_UN)
            time.sleep(0.01)
        assert worker_has_lease
        coordinator.kill()
        coordinator.wait(timeout=10)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            with lease_path.open("a") as lease:
                try:
                    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    time.sleep(0.05)
                else:
                    fcntl.flock(lease, fcntl.LOCK_UN)
                    break
        else:
            pytest.fail("Orphaned worker did not release its lease after coordinator death")
        resumed = BatchRunner(store, load_config(config_file), max_workers=1, device="cpu").run_job("crash")
        assert (resumed.completed_targets, resumed.failed_targets) == (1, 0)
        task = store.list_tasks("crash")[0]
        assert task.attempt_count == 2
    finally:
        if coordinator.poll() is None:
            coordinator.kill()
            coordinator.wait(timeout=10)


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


def test_batch_respects_workspace_and_validates_empty_cached_result(tmp_path: Path) -> None:
    config_file = tmp_path / "settings.toml"
    workspace = tmp_path / "external_workspace"
    config_file.write_text(f'[workspace]\nroot = "{workspace}"\n', encoding="utf-8")
    config = load_config(config_file)
    store = JobStore(workspace / "state.db")
    fits_path = _create_mock_fits(tmp_path / "target_lc.fits", "9191")
    targets = [("9191", f"file://{fits_path}")]
    settings = {
        "max_candidates": 1, "snr_threshold": 1e20,
        "quality_mask": 0, "gap_threshold_hours": 4.0,
        "bls_min_period_days": 1.0, "bls_max_period_days": 3.0,
        "bootstrap_replicates": 10, "device": "auto",
    }
    runner = BatchRunner(store=store, config=config, max_workers=1, device="cpu")

    store.create_job("first", targets, settings)
    first_events = []
    assert runner.run_job("first", first_events.append).completed_targets == 1
    first = first_events[0]
    assert first.candidate_count == 0
    assert Path(first.bundle_path).is_relative_to(workspace)
    document = json.loads((Path(first.bundle_path) / "result.json").read_text())
    assert document["analysis_settings"]["quality_mask"] == 0
    assert document["analysis_settings"]["gap_threshold_hours"] == 4.0
    assert document["analysis_settings"]["bls_min_period_days"] == 1.0
    assert document["analysis_settings"]["bls_max_period_days"] == 3.0
    assert document["analysis_settings"]["bootstrap_replicates"] == 10
    assert document["analysis_settings"]["device"] == "auto"
    assert document["resolved_device"] == "cpu"

    store.create_job("second", targets, settings)
    second_events = []
    assert runner.run_job("second", second_events.append).completed_targets == 1
    assert second_events[0].status == "cached"
    assert second_events[0].candidate_count == 0

    # A partial or corrupt cached bundle must be recomputed, not marked completed.
    np.savez_compressed(Path(first.bundle_path) / "arrays.npz", time_days=np.array([123.0]))
    store.create_job("third", targets, settings)
    third_events = []
    assert runner.run_job("third", third_events.append).completed_targets == 1
    assert third_events[0].status == "completed"
    assert third_events[0].candidate_count == 0

    changed_settings = {**settings, "gap_threshold_hours": 5.0}
    store.create_job("fourth", targets, changed_settings)
    fourth_events = []
    assert runner.run_job("fourth", fourth_events.append).completed_targets == 1
    assert fourth_events[0].status == "completed"


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
        workers=1,
        benchmark_id="test_bench_run",
        pause_after_finished=1,
    )

    assert isinstance(report, BenchmarkReport)
    assert report.benchmark_id == "test_bench_run"
    assert report.target_count == 2
    assert report.completed_targets == 2
    assert report.failed_targets == 0
    assert report.throughput_targets_per_sec > 0
    assert report.peak_rss_mb > 0
    assert report.details["paused_snapshot"]["status"] == "paused"
    assert report.details["paused_snapshot"]["completed"] == 1
    assert report.details["fresh_completed_excluding_cache"] == 2
    assert report.details["unique_task_ids"] == report.details["task_records"] == 2
    assert report.details["peak_process_count"] >= 2

    json_report = out_dir / "test_bench_run_report.json"
    assert json_report.exists()
    original_report = json_report.read_bytes()
    assert not report.details["bundle_read_errors"]
    assert sum(report.details["analysis_status_counts"].values()) == 2
    assert isinstance(report.details["candidate_flag_counts"], dict)
    assert isinstance(report.details["physical_fit_failure_tasks"], list)
    replay = run_sector_benchmark(
        target_items=targets, output_dir=out_dir, device="cpu", workers=1,
        benchmark_id="test_bench_run", pause_after_finished=1,
    )
    assert replay.throughput_targets_per_sec == 0
    assert replay.details["preexisting_completed"] == 2
    assert json_report.read_bytes() == original_report
    assert Path(replay.details["report_path"]).is_file()


def test_benchmark_reports_failed_tasks_without_counting_them_as_throughput(tmp_path: Path) -> None:
    valid = _create_mock_fits(tmp_path / "valid.fits", "7001")
    missing = tmp_path / "missing.fits"
    report = run_sector_benchmark(
        [("valid", valid.as_uri()), ("missing", missing.as_uri())], tmp_path / "failure-benchmark",
        device="cpu", workers=1, benchmark_id="failed-task",
    )
    assert report.completed_targets == report.failed_targets == 1
    assert report.details["fresh_completed_excluding_cache"] == 1
    assert report.details["failed_tasks"][0]["target_id"] == "missing"
    assert report.details["task_records"] == 2

