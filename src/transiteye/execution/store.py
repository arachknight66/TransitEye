"""SQLite WAL state store for reproducible, resumable batch job execution."""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


class JobStatus(StrEnum):
    """Lifecycle state of a batch analysis job."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class TaskStatus(StrEnum):
    """Lifecycle state of an individual target analysis task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class JobRecord:
    """Persistent metadata for a batch execution job."""

    job_id: str
    status: JobStatus
    created_at: float
    updated_at: float
    total_targets: int
    completed_targets: int
    failed_targets: int
    config_json: str
    error_message: str | None = None


@dataclass(frozen=True, slots=True)
class TargetTaskRecord:
    """Persistent state of a single target within a job."""

    task_id: str
    job_id: str
    target_id: str
    source_uri: str
    status: TaskStatus
    attempt_count: int
    created_at: float
    completed_at: float | None
    input_sha256: str | None
    bundle_path: str | None
    candidate_count: int
    error_message: str | None = None


def _task_from_row(row: sqlite3.Row) -> TargetTaskRecord:
    return TargetTaskRecord(
        task_id=row["task_id"], job_id=row["job_id"], target_id=row["target_id"],
        source_uri=row["source_uri"], status=TaskStatus(row["status"]),
        attempt_count=row["attempt_count"], created_at=row["created_at"],
        completed_at=row["completed_at"], input_sha256=row["input_sha256"],
        bundle_path=row["bundle_path"], candidate_count=row["candidate_count"],
        error_message=row["error_message"],
    )


class JobStore:
    """Thread-safe SQLite state store operating with Write-Ahead Logging (WAL)."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path.expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA busy_timeout = 30000;")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    total_targets INTEGER NOT NULL,
                    completed_targets INTEGER NOT NULL DEFAULT 0,
                    failed_targets INTEGER NOT NULL DEFAULT 0,
                    config_json TEXT NOT NULL,
                    error_message TEXT
                );

                CREATE TABLE IF NOT EXISTS target_tasks (
                    task_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL REFERENCES jobs(job_id),
                    target_id TEXT NOT NULL,
                    source_uri TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    completed_at REAL,
                    input_sha256 TEXT,
                    bundle_path TEXT,
                    candidate_count INTEGER NOT NULL DEFAULT 0,
                    error_message TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_tasks_job_status
                ON target_tasks(job_id, status);

                CREATE TABLE IF NOT EXISTS cache_entries (
                    cache_key TEXT PRIMARY KEY,
                    bundle_path TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    target_id TEXT NOT NULL
                );
                """
            )

    def create_job(
        self,
        job_id: str,
        target_items: list[tuple[str, str]],  # (target_id, source_uri)
        config_data: dict[str, Any],
    ) -> JobRecord:
        """Register a new batch job and queue all targets as pending tasks."""
        now = time.time()
        cfg_str = json.dumps(config_data, sort_keys=True)
        total = len(target_items)

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO jobs (
                    job_id, status, created_at, updated_at,
                    total_targets, completed_targets, failed_targets,
                    config_json
                ) VALUES (?, ?, ?, ?, ?, 0, 0, ?);
                """,
                (job_id, JobStatus.PENDING.value, now, now, total, cfg_str),
            )

            tasks_data = [
                (
                    f"{job_id}:{t_id}",
                    job_id,
                    t_id,
                    uri,
                    TaskStatus.PENDING.value,
                    0,
                    now,
                )
                for t_id, uri in target_items
            ]
            conn.executemany(
                """
                INSERT INTO target_tasks (
                    task_id, job_id, target_id, source_uri,
                    status, attempt_count, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                tasks_data,
            )

        return self.get_job(job_id)

    def get_job(self, job_id: str) -> JobRecord:
        """Fetch job details by ID."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?;", (job_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Job not found: {job_id}")
            return JobRecord(
                job_id=row["job_id"],
                status=JobStatus(row["status"]),
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                total_targets=row["total_targets"],
                completed_targets=row["completed_targets"],
                failed_targets=row["failed_targets"],
                config_json=row["config_json"],
                error_message=row["error_message"],
            )

    def reset_interrupted_tasks(self, job_id: str) -> int:
        """Reset tasks stuck in RUNNING state back to PENDING after an interruption or restart."""
        with self._get_connection() as conn:
            cur = conn.execute(
                """
                UPDATE target_tasks
                SET status = ?
                WHERE job_id = ? AND status = ?;
                """,
                (TaskStatus.PENDING.value, job_id, TaskStatus.RUNNING.value),
            )
            return cur.rowcount

    def get_pending_tasks(self, job_id: str, limit: int | None = None) -> list[TargetTaskRecord]:
        """Fetch pending tasks; interrupted tasks must first be reset explicitly."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM target_tasks
                WHERE job_id = ? AND status = ?
                ORDER BY created_at ASC, task_id ASC
                LIMIT ?;
                """,
                (job_id, TaskStatus.PENDING.value, limit if limit is not None else -1),
            ).fetchall()
            return [_task_from_row(row) for row in rows]

    def list_tasks(self, job_id: str, *, limit: int = 500, offset: int = 0) -> list[TargetTaskRecord]:
        """Read a bounded page of persisted target records, including failures."""
        if not 1 <= limit <= 1000 or offset < 0:
            raise ValueError("Task page limit must be 1–1000 and offset must be nonnegative.")
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM target_tasks WHERE job_id = ? ORDER BY created_at, task_id LIMIT ? OFFSET ?",
                (job_id, limit, offset),
            ).fetchall()
            return [_task_from_row(row) for row in rows]

    def update_job_status(
        self,
        job_id: str,
        status: JobStatus,
        error_message: str | None = None,
    ) -> None:
        """Update job terminal or intermediate status."""
        now = time.time()
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = ?, updated_at = ?, error_message = ?
                WHERE job_id = ?;
                """,
                (status.value, now, error_message, job_id),
            )

    def transition_job(self, job_id: str, allowed: tuple[JobStatus, ...], status: JobStatus) -> bool:
        """Atomically change a job only if its current state permits the transition."""
        if not allowed:
            return False
        placeholders = ",".join("?" for _ in allowed)
        with self._get_connection() as conn:
            cursor = conn.execute(
                f"UPDATE jobs SET status = ?, updated_at = ?, error_message = NULL "
                f"WHERE job_id = ? AND status IN ({placeholders})",
                (status.value, time.time(), job_id, *(item.value for item in allowed)),
            )
            return cursor.rowcount == 1

    def retry_failed_tasks(self, job_id: str) -> int:
        """Requeue failed tasks after a job has stopped; successful results are retained."""
        with self._get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT status FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(f"Job not found: {job_id}")
            if row["status"] not in (JobStatus.COMPLETED.value, JobStatus.FAILED.value):
                raise ValueError("Retry requires a completed or failed job.")
            result = conn.execute(
                "UPDATE target_tasks SET status = ?, completed_at = NULL, error_message = NULL "
                "WHERE job_id = ? AND status = ?",
                (TaskStatus.PENDING.value, job_id, TaskStatus.FAILED.value),
            )
            if result.rowcount:
                conn.execute(
                    "UPDATE jobs SET status = ?, failed_targets = failed_targets - ?, updated_at = ? WHERE job_id = ?",
                    (JobStatus.PAUSED.value, result.rowcount, time.time(), job_id),
                )
            return result.rowcount

    def mark_task_running(self, task_id: str) -> None:
        """Mark task as actively running and increment attempt count."""
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE target_tasks
                SET status = ?, attempt_count = attempt_count + 1
                WHERE task_id = ?;
                """,
                (TaskStatus.RUNNING.value, task_id),
            )

    def mark_task_completed(
        self,
        task_id: str,
        bundle_path: str,
        candidate_count: int,
        input_sha256: str | None = None,
        error_message: str | None = None,
    ) -> None:
        """Mark task as successfully completed and increment job counter."""
        now = time.time()
        with self._get_connection() as conn:
            updated = conn.execute(
                """
                UPDATE target_tasks
                SET status = ?, completed_at = ?, bundle_path = ?,
                    candidate_count = ?, input_sha256 = ?, error_message = ?
                WHERE task_id = ? AND status IN (?, ?);
                """,
                (
                    TaskStatus.COMPLETED.value,
                    now,
                    bundle_path,
                    candidate_count,
                    input_sha256,
                    error_message,
                    task_id,
                    TaskStatus.PENDING.value,
                    TaskStatus.RUNNING.value,
                ),
            )
            if updated.rowcount != 1:
                return
            # Update parent job completed_targets
            job_id = conn.execute("SELECT job_id FROM target_tasks WHERE task_id = ?", (task_id,)).fetchone()[0]
            conn.execute(
                """
                UPDATE jobs
                SET completed_targets = completed_targets + 1, updated_at = ?
                WHERE job_id = ?;
                """,
                (now, job_id),
            )

    def mark_task_failed(self, task_id: str, error_message: str) -> None:
        """Mark task as failed and increment job failed counter."""
        now = time.time()
        with self._get_connection() as conn:
            updated = conn.execute(
                """
                UPDATE target_tasks
                SET status = ?, completed_at = ?, error_message = ?
                WHERE task_id = ? AND status IN (?, ?);
                """,
                (TaskStatus.FAILED.value, now, error_message, task_id,
                 TaskStatus.PENDING.value, TaskStatus.RUNNING.value),
            )
            if updated.rowcount != 1:
                return
            job_id = conn.execute("SELECT job_id FROM target_tasks WHERE task_id = ?", (task_id,)).fetchone()[0]
            conn.execute(
                """
                UPDATE jobs
                SET failed_targets = failed_targets + 1, updated_at = ?
                WHERE job_id = ?;
                """,
                (now, job_id),
            )

    def get_cache(self, cache_key: str) -> str | None:
        """Retrieve cached bundle path for identical checksums."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT bundle_path FROM cache_entries WHERE cache_key = ?;",
                (cache_key,),
            ).fetchone()
            return row["bundle_path"] if row else None

    def put_cache(self, cache_key: str, bundle_path: str, target_id: str) -> None:
        """Store result bundle in cache."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO cache_entries (
                    cache_key, bundle_path, created_at, target_id
                ) VALUES (?, ?, ?, ?);
                """,
                (cache_key, bundle_path, time.time(), target_id),
            )
