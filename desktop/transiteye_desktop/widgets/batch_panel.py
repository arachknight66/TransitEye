"""Sector batch processing view and control panel."""

from __future__ import annotations

import time
from typing import Any

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class BatchPanel(QWidget):
    """Sector batch job configuration, progress tracking, and results browser."""

    start_batch_requested = pyqtSignal(str, str, int)  # source_path, device, workers
    cancel_batch_requested = pyqtSignal()
    control_requested = pyqtSignal(str, str)
    resume_requested = pyqtSignal(str, int)
    inspect_requested = pyqtSignal(str, int)
    open_bundle_requested = pyqtSignal(str)  # bundle_path

    HEADERS = ["Target ID", "Status", "Candidates", "Bundle Path", "Error Details"]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)

        # Batch Configuration Group
        grp_cfg = QGroupBox("Batch Sector Configuration")
        cfg_layout = QHBoxLayout(grp_cfg)

        self.txt_source = QLineEdit()
        self.txt_source.setPlaceholderText("Path to JSON manifest or directory containing *_lc.fits...")
        self.btn_browse = QPushButton("Browse...")
        self.btn_browse.clicked.connect(self._browse_source)

        lbl_device = QLabel("Device:")
        self.combo_device = QComboBox()
        self.combo_device.addItems(["auto", "cuda", "cpu"])

        lbl_workers = QLabel("Workers:")
        self.spin_workers = QSpinBox()
        self.spin_workers.setRange(1, 32)
        self.spin_workers.setValue(4)

        self.btn_start = QPushButton("Start Batch Run")
        self.btn_start.setStyleSheet("background-color: #2E7D32; color: white; font-weight: bold; padding: 6px 12px;")
        self.btn_start.clicked.connect(self._on_start_clicked)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setStyleSheet("background-color: #C62828; color: white; font-weight: bold; padding: 6px 12px;")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.cancel_batch_requested)

        cfg_layout.addWidget(self.txt_source)
        cfg_layout.addWidget(self.btn_browse)
        cfg_layout.addWidget(lbl_device)
        cfg_layout.addWidget(self.combo_device)
        cfg_layout.addWidget(lbl_workers)
        cfg_layout.addWidget(self.spin_workers)
        cfg_layout.addWidget(self.btn_start)
        cfg_layout.addWidget(self.btn_cancel)

        layout.addWidget(grp_cfg)
        self.lbl_search_options = QLabel("New batch jobs use the search-period range from Target Inspector.")
        self.lbl_search_options.setWordWrap(True)
        layout.addWidget(self.lbl_search_options)

        job_controls = QHBoxLayout()
        self.txt_job_id = QLineEdit()
        self.txt_job_id.setPlaceholderText("Saved batch job ID")
        job_controls.addWidget(self.txt_job_id)
        for action in ("pause", "cancel", "retry"):
            button = QPushButton(action.capitalize())
            button.clicked.connect(lambda _checked=False, name=action: self.control_requested.emit(name, self.txt_job_id.text().strip()))
            job_controls.addWidget(button)
        self.btn_resume = QPushButton("Resume")
        self.btn_resume.clicked.connect(lambda: self.resume_requested.emit(self.txt_job_id.text().strip(), self.spin_workers.value()))
        job_controls.addWidget(self.btn_resume)
        self.btn_inspect = QPushButton("Load job")
        self.btn_inspect.clicked.connect(self._load_job)
        job_controls.addWidget(self.btn_inspect)
        self.btn_more = QPushButton("Load more")
        self.btn_more.setEnabled(False)
        self.btn_more.clicked.connect(lambda: self.inspect_requested.emit(self.txt_job_id.text().strip(), self._loaded_offset))
        job_controls.addWidget(self.btn_more)
        layout.addLayout(job_controls)

        # Progress bar & Metrics
        prog_layout = QHBoxLayout()
        self.prog_bar = QProgressBar()
        self.prog_bar.setRange(0, 100)
        self.prog_bar.setValue(0)
        self.lbl_stats = QLabel("Ready")
        self.lbl_stats.setStyleSheet("font-weight: bold;")

        prog_layout.addWidget(self.prog_bar)
        prog_layout.addWidget(self.lbl_stats)
        layout.addLayout(prog_layout)

        # Targets table
        self.table = QTableWidget(0, len(self.HEADERS))
        self.table.setHorizontalHeaderLabels(self.HEADERS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemDoubleClicked.connect(self._on_item_double_clicked)

        layout.addWidget(self.table)

        self._start_time: float | None = None
        self._target_rows: dict[str, int] = {}
        self._loaded_offset = 0

    def _load_job(self) -> None:
        job_id = self.txt_job_id.text().strip()
        if job_id:
            self.table.setRowCount(0)
            self._target_rows.clear()
            self._loaded_offset = 0
            self.btn_more.setEnabled(False)
            self.inspect_requested.emit(job_id, 0)

    def on_batch_snapshot(self, data: dict[str, Any]) -> None:
        """Restore a page of persisted targets, including errors and bundle paths."""
        if data.get("job_id") != self.txt_job_id.text().strip():
            return
        tasks = data.get("tasks", [])
        for task in tasks:
            self.on_batch_progress({**task, "total_targets": data["total_targets"],
                                    "completed_targets": data["completed_targets"],
                                    "failed_targets": data["failed_targets"]})
        self._loaded_offset = int(data.get("task_offset", 0)) + len(tasks)
        self.btn_more.setEnabled(self._loaded_offset < int(data["total_targets"]))
        if not tasks:
            self.lbl_stats.setText(f"Job {data.get('status', 'unknown')}: no targets in this page")

    def _browse_source(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Manifest or FITS file",
            "",
            "Manifest or FITS (*.json *.fits);;All Files (*)",
        )
        if path:
            self.txt_source.setText(path)

    def _on_start_clicked(self) -> None:
        src = self.txt_source.text().strip()
        if not src:
            return
        dev = self.combo_device.currentText()
        workers = self.spin_workers.value()
        self.btn_start.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.prog_bar.setValue(0)
        self.table.setRowCount(0)
        self._target_rows.clear()
        self._start_time = time.perf_counter()
        self.start_batch_requested.emit(src, dev, workers)

    def on_batch_progress(self, data: dict[str, Any]) -> None:
        """Handle streaming progress from BatchProgress event."""
        task_id = str(data.get("task_id", ""))
        target_id = str(data.get("target_id", task_id.split(":")[-1]))
        status = str(data.get("status", ""))
        bundle = str(data.get("bundle_path", "") or "")
        err = str(data.get("error_message", "") or "")
        c_count = data.get("candidate_count", None)

        if target_id not in self._target_rows:
            row = self.table.rowCount()
            self.table.insertRow(row)
            self._target_rows[target_id] = row
            self.table.setItem(row, 0, QTableWidgetItem(target_id))
        else:
            row = self._target_rows[target_id]

        self.table.setItem(row, 1, QTableWidgetItem(status))
        self.table.setItem(row, 2, QTableWidgetItem(str(c_count) if c_count is not None else "--"))
        self.table.setItem(row, 3, QTableWidgetItem(bundle))
        self.table.setItem(row, 4, QTableWidgetItem(err))

        # Update stats
        total = int(data.get("total_targets", 0))
        completed = int(data.get("completed_targets", 0))
        failed = int(data.get("failed_targets", 0))
        if total > 0:
            self.prog_bar.setValue(round(100 * (completed + failed) / total))

        elapsed = time.perf_counter() - (self._start_time or time.perf_counter())
        tps = (completed / elapsed) if elapsed > 0 else 0.0
        self.lbl_stats.setText(f"{completed} completed | {failed} failed | {total} total | {tps:.2f} targets/sec")

    def on_batch_finished(self) -> None:
        """Batch process completed or terminated."""
        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)

    def _on_item_double_clicked(self, item: QTableWidgetItem) -> None:
        row = item.row()
        bundle_item = self.table.item(row, 3)
        if bundle_item and bundle_item.text():
            self.open_bundle_requested.emit(bundle_item.text())
