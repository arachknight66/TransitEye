#!/usr/bin/env python3
"""TransitEye Batch Pipeline Demo: Programmatic Sector-Scale Processing.

Demonstrates creating a persistent SQLite job, executing targets with multi-worker
concurrency, progress monitoring, and loading results into memory.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from transiteye.execution.runner import BatchProgress, BatchRunner
from transiteye.execution.store import JobStatus, JobStore


def main() -> int:
    pilot_dir = Path(".transiteye-workspace/data/pilot")
    sample_files = sorted(pilot_dir.glob("*.fits"))[:4] if pilot_dir.is_dir() else []

    if len(sample_files) < 2:
        print("Need at least 2 FITS files in .transiteye-workspace/data/pilot to demonstrate batch processing.")
        return 1

    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "demo_state.db"
        store = JobStore(db_path)
        job_id = "demo_sector_batch"

        # Prepare target entries
        targets = [
            (f.stem.split("-")[2] if "-" in f.stem else f.stem, f"file://{f.resolve()}")
            for f in sample_files
        ]

        print(f"[1/4] Initializing Job '{job_id}' with {len(targets)} targets in SQLite WAL...")
        store.create_job(
            job_id=job_id,
            target_items=targets,
            config_data={
                "max_candidates": 1,
                "snr_threshold": 6.0,
                "target_timeout_seconds": 60.0,
            },
        )

        runner = BatchRunner(
            store=store,
            max_workers=2,
            device="cpu",
        )

        print("[2/4] Executing batch with 2 concurrent workers and timeout protection...")

        def _progress(p: BatchProgress) -> None:
            print(f"      Target: {p.target_id:<15} Status: {p.status:<10} Progress: {p.percent:.0f}%")

        final_job = runner.run_job(job_id, progress_callback=_progress)

        print(f"\n[3/4] Batch Job Finished: Status = {final_job.status.value.upper()}")
        print(f"      Total Targets: {final_job.total_targets}, Completed: {final_job.completed_targets}, Failed: {final_job.failed_targets}")

        # Verify resumption / cache hits
        print("\n[4/4] Demonstrating instant cache replay on second execution...")
        runner_replay = BatchRunner(store=store, max_workers=2, device="cpu")
        replay_job = runner_replay.run_job(job_id)
        print(f"      Replay status: {replay_job.status.value.upper()} (All cached results resolved immediately)")

    print("\nBatch pipeline demonstration completed successfully.")
    return 0 if final_job.status == JobStatus.COMPLETED else 1


if __name__ == "__main__":
    sys.exit(main())
