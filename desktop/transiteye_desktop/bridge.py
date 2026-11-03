"""Supervised QProcess bridge between desktop UI and headless scientific backend.

Executes `python -m transiteye analyze-fits` or `run-batch` with JSONL streaming,
parsing events and forwarding them as Qt signals across the process boundary.
Zero scientific logic resides here.
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QObject, QProcess, QTimer, pyqtSignal


class BackendProcessBridge(QObject):
    """Supervises backend CLI invocations via non-blocking QProcess."""

    # Signals
    progress_changed = pyqtSignal(str, float)  # (stage_name, fraction 0.0-1.0)
    batch_progress = pyqtSignal(dict)  # batch task payload
    batch_snapshot = pyqtSignal(dict)
    export_completed = pyqtSignal(str)
    export_failed = pyqtSignal(str)
    analysis_completed = pyqtSignal(str, dict)  # (bundle_path, summary_dict)
    analysis_failed = pyqtSignal(str)  # error message
    process_state_changed = pyqtSignal(bool)  # is_running

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.ProcessChannelMode.SeparateChannels)
        self._process.readyReadStandardOutput.connect(self._on_stdout_ready)
        self._process.readyReadStandardError.connect(self._on_stderr_ready)
        self._process.finished.connect(self._on_process_finished)

        self._stdout_buffer = ""
        self._stderr_buffer = ""
        self._full_stdout = ""
        self._last_result_bundle: str | None = None
        self._last_summary: dict[str, Any] | None = None
        self.batch_job_id: str | None = None
        self._controls: set[QProcess] = set()

    @property
    def is_running(self) -> bool:
        """Return True if background process is active."""
        return self._process.state() != QProcess.ProcessState.NotRunning

    def analyze_fits(
        self,
        fits_path: Path | str,
        device: str = "cpu",
        tpf_path: Path | str | None = None,
        max_candidates: int = 3,
        snr_threshold: float = 6.0,
        output_dir: Path | str | None = None,
        min_period_days: float = 0.3,
        max_period_days: float | None = None,
        model_dir: Path | str | None = None,
        model_calibrator_sha256: str | None = None,
    ) -> None:
        """Launch headless analysis of a single FITS file."""
        if self.is_running:
            raise RuntimeError("Another backend process is already running.")

        self._stdout_buffer = ""
        self._stderr_buffer = ""
        self._full_stdout = ""
        self._last_result_bundle = None
        self._last_summary = None
        self.batch_job_id = None

        args = [
            "-m",
            "transiteye",
            "analyze-fits",
            str(fits_path),
            "--device",
            device,
            "--max-candidates",
            str(max_candidates),
            "--snr-threshold",
            str(snr_threshold),
            "--progress-jsonl",
            "--bls-min-period-days",
            str(min_period_days),
        ]
        if max_period_days is not None:
            args.extend(["--bls-max-period-days", str(max_period_days)])
        if tpf_path:
            args.extend(["--tpf", str(tpf_path)])
        if output_dir:
            args.extend(["--output-dir", str(output_dir)])
        if model_dir:
            args.extend(["--model-dir", str(model_dir)])
        if model_calibrator_sha256:
            args.extend(["--model-calibrator-sha256", model_calibrator_sha256])

        self.process_state_changed.emit(True)
        self.progress_changed.emit("Starting analysis process...", 0.0)
        self._process.start(sys.executable, args)

    def run_batch(
        self,
        source_path: Path | str,
        device: str = "cpu",
        workers: int = 4,
        max_candidates: int = 3,
        job_id: str | None = None,
        min_period_days: float = 0.3,
        max_period_days: float | None = None,
        model_dir: Path | str | None = None,
        model_calibrator_sha256: str | None = None,
    ) -> None:
        """Launch headless batch execution on a manifest or folder."""
        if self.is_running:
            raise RuntimeError("Another backend process is already running.")

        self._stdout_buffer = ""
        self._stderr_buffer = ""
        self._full_stdout = ""
        self._last_result_bundle = None
        self._last_summary = None

        args = [
            "-m",
            "transiteye",
            "run-batch",
            str(source_path),
            "--device",
            device,
            "--workers",
            str(workers),
            "--max-candidates",
            str(max_candidates),
            "--snr-threshold",
            "6.0",
            "--bls-min-period-days",
            str(min_period_days),
            "--progress-jsonl",
        ]
        if max_period_days is not None:
            args.extend(["--bls-max-period-days", str(max_period_days)])
        if model_dir:
            args.extend(["--model-dir", str(model_dir)])
        if model_calibrator_sha256:
            args.extend(["--model-calibrator-sha256", model_calibrator_sha256])
        self.batch_job_id = job_id or f"batch_{uuid.uuid4().hex}"
        args.extend(["--job-id", self.batch_job_id])

        self.process_state_changed.emit(True)
        self.progress_changed.emit("Starting batch job...", 0.0)
        self._process.start(sys.executable, args)

    def control_batch(self, action: str, job_id: str) -> None:
        """Send a non-blocking control request through the backend CLI."""
        if action not in {"pause", "cancel", "retry"} or not job_id:
            raise ValueError("Invalid batch control request.")
        process = QProcess(self)
        self._controls.add(process)

        def finished(code: int, _status: QProcess.ExitStatus) -> None:
            details = bytes(process.readAllStandardOutput()).decode().strip()
            error = bytes(process.readAllStandardError()).decode().strip()
            if code:
                self.analysis_failed.emit(error or details or f"{action} request failed")
            else:
                self.progress_changed.emit(f"{action.capitalize()} requested: {details}", 0.0)
            self._controls.discard(process)
            process.deleteLater()

        process.finished.connect(finished)
        process.start(sys.executable, ["-m", "transiteye", f"{action}-job", job_id])

    def resume_batch(self, job_id: str, workers: int) -> None:
        """Resume a saved job using its persisted analysis settings."""
        if self.is_running:
            raise RuntimeError("Another backend process is already running.")
        self._stdout_buffer = self._stderr_buffer = self._full_stdout = ""
        self._last_result_bundle = None
        self._last_summary = None
        self.batch_job_id = job_id
        self.process_state_changed.emit(True)
        self._process.start(sys.executable, ["-m", "transiteye", "resume-job", job_id,
                                             "--workers", str(workers), "--progress-jsonl"])

    def inspect_batch(self, job_id: str, offset: int) -> None:
        """Fetch a page of saved task records asynchronously."""
        process = QProcess(self)
        self._controls.add(process)

        def finished(code: int, _status: QProcess.ExitStatus) -> None:
            output = bytes(process.readAllStandardOutput()).decode()
            error = bytes(process.readAllStandardError()).decode().strip()
            if code:
                self.analysis_failed.emit(error or "Could not load saved job.")
            else:
                try:
                    self.batch_snapshot.emit(json.loads(output))
                except (ValueError, TypeError):
                    self.analysis_failed.emit("Could not parse saved job status.")
            self._controls.discard(process)
            process.deleteLater()

        process.finished.connect(finished)
        process.start(sys.executable, ["-m", "transiteye", "job-status", job_id, "--tasks",
                                      "--limit", "500", "--offset", str(offset)])

    def export_vetting_sheet(
        self,
        bundle_dir: Path | str,
        output_file: Path | str,
        candidate_index: int = 0,
    ) -> bool:
        """Render a diagnostic vetting sheet (.png/.pdf/.svg) via backend CLI subprocess."""
        args = [
            "-m",
            "transiteye",
            "export-vetting-sheet",
            str(bundle_dir),
            str(output_file),
            "--candidate-index",
            str(candidate_index),
        ]
        code = QProcess.execute(sys.executable, args)
        return code == 0

    def start_vetting_sheet_export(
        self, bundle_dir: Path | str, output_file: Path | str, candidate_index: int = 0,
    ) -> None:
        """Render a vetting sheet without blocking the desktop event loop."""
        process = QProcess(self)
        self._controls.add(process)

        def finished(code: int, _status: QProcess.ExitStatus) -> None:
            error = bytes(process.readAllStandardError()).decode(errors="replace").strip()
            if code:
                self.export_failed.emit(error or f"Export failed with exit code {code}.")
            else:
                self.export_completed.emit(str(output_file))
            self._controls.discard(process)
            process.deleteLater()

        process.finished.connect(finished)
        process.start(sys.executable, ["-m", "transiteye", "export-vetting-sheet", str(bundle_dir),
                                      str(output_file), "--candidate-index", str(candidate_index)])

    def cancel(self) -> None:
        """Request graceful termination of backend process, then kill if unresponsive."""
        if not self.is_running:
            return
        if self.batch_job_id:
            self.control_batch("cancel", self.batch_job_id)
            return
        self.progress_changed.emit("Canceling background process...", 0.0)
        process_id = self._process.processId()
        self._process.terminate()
        def kill_if_unresponsive() -> None:
            if self.is_running and self._process.processId() == process_id:
                self._process.kill()

        QTimer.singleShot(1500, kill_if_unresponsive)

    def _handle_message(self, msg: dict[str, Any]) -> None:
        """Dispatch a parsed JSON message from the backend process."""
        if "stage" in msg:
            if "fraction" in msg:
                frac = float(msg.get("fraction", 0.0))
            elif "percent" in msg:
                frac = float(msg.get("percent", 0.0)) / 100.0
            else:
                frac = 0.0
            stage = str(msg.get("stage", ""))
            self.progress_changed.emit(stage, frac)
        elif "task_id" in msg or ("job_id" in msg and "target_id" in msg):
            self.batch_progress.emit(msg)
        elif "bundle_path" in msg:
            self._last_result_bundle = str(msg["bundle_path"])
            self._last_summary = msg
        elif "job_id" in msg and ("status" in msg or "total_targets" in msg):
            self._last_summary = msg

    def _on_stdout_ready(self) -> None:
        data = self._process.readAllStandardOutput().data().decode("utf-8", errors="replace")
        self._stdout_buffer += data
        self._full_stdout += data

        while "\n" in self._stdout_buffer:
            line, self._stdout_buffer = self._stdout_buffer.split("\n", 1)
            line = line.strip()
            if not line:
                continue

            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue

            if isinstance(msg, dict):
                self._handle_message(msg)

    def _on_stderr_ready(self) -> None:
        data = self._process.readAllStandardError().data().decode("utf-8", errors="replace")
        self._stderr_buffer += data

    def _on_process_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        # Drain any residual stdout or stderr
        self._on_stdout_ready()
        self._on_stderr_ready()

        # If summary was not captured line-by-line, attempt fallback parsing
        if not self._last_result_bundle and not self._last_summary:
            buf = self._stdout_buffer.strip()
            if buf:
                try:
                    msg = json.loads(buf)
                    if isinstance(msg, dict):
                        self._handle_message(msg)
                except json.JSONDecodeError:
                    pass

        # If still missing, scan full accumulated stdout for the trailing JSON dictionary
        if not self._last_result_bundle and not self._last_summary:
            full = self._full_stdout.strip()
            start_brace = full.rfind("{")
            end_brace = full.rfind("}")
            if start_brace != -1 and end_brace > start_brace:
                try:
                    msg = json.loads(full[start_brace : end_brace + 1])
                    if isinstance(msg, dict):
                        self._handle_message(msg)
                except json.JSONDecodeError:
                    pass

        self.process_state_changed.emit(False)
        if exit_code == 0 and self._last_result_bundle:
            self.progress_changed.emit("Analysis completed", 1.0)
            self.analysis_completed.emit(self._last_result_bundle, self._last_summary or {})
        elif exit_code == 0 and self._last_summary and "job_id" in self._last_summary:
            status = str(self._last_summary.get("status", "finished"))
            self.progress_changed.emit(f"Batch {status}", 1.0 if status == "completed" else 0.0)
            self.analysis_completed.emit("", self._last_summary)
        else:
            if exit_code == 0:
                err_msg = (
                    self._stderr_buffer.strip()
                    or "Process finished with exit code 0, but no result bundle or summary was found in output."
                )
            else:
                err_msg = (
                    self._stderr_buffer.strip()
                    or f"Process failed with exit code {exit_code}"
                )
            self.progress_changed.emit("Process failed", 0.0)
            self.analysis_failed.emit(err_msg)
