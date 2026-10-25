"""Main workstation window for TransitEye Desktop."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from transiteye_desktop.bridge import BackendProcessBridge
from transiteye_desktop.bundle_loader import LoadedBundle, load_result_bundle
from transiteye_desktop.widgets.batch_panel import BatchPanel
from transiteye_desktop.widgets.candidate_table_widget import CandidateTableWidget
from transiteye_desktop.widgets.classification_card import ClassificationCard
from transiteye_desktop.widgets.folded_transit_widget import FoldedTransitWidget
from transiteye_desktop.widgets.light_curve_widget import LightCurvePlotWidget
from transiteye_desktop.widgets.periodogram_widget import PeriodogramWidget
from transiteye_desktop.widgets.vetting_card import VettingCard


class TransitEyeMainWindow(QMainWindow):
    """Scientific desktop analysis interface for TransitEye."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("TransitEye — TESS Light Curve Analysis & Vetting")
        self.resize(1400, 900)

        self._bridge = BackendProcessBridge(self)
        self._current_bundle: LoadedBundle | None = None
        self._current_fits: Path | None = None
        self._current_tpf: Path | None = None
        self._selected_candidate_index: int = 0

        self._init_ui()
        self._init_shortcuts()
        self._connect_signals()

    def _init_ui(self) -> None:
        # Central Tabs
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Tab 1: Single Target Inspector
        tab_inspector = QWidget()
        insp_layout = QHBoxLayout(tab_inspector)
        insp_layout.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left pane: Controls & Summary Cards
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.candidate_table = CandidateTableWidget()
        self.classification_card = ClassificationCard()
        self.vetting_card = VettingCard()

        left_layout.addWidget(self.candidate_table, stretch=2)
        left_layout.addWidget(self.classification_card, stretch=1)
        left_layout.addWidget(self.vetting_card, stretch=1)

        splitter.addWidget(left_widget)

        # Right pane: Scientific Visualizations
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.lc_widget = LightCurvePlotWidget()
        self.periodogram_widget = PeriodogramWidget()
        self.folded_widget = FoldedTransitWidget()

        # Splitter for plots
        plot_splitter = QSplitter(Qt.Orientation.Vertical)
        plot_splitter.addWidget(self.lc_widget)
        plot_splitter.addWidget(self.periodogram_widget)
        plot_splitter.addWidget(self.folded_widget)

        right_layout.addWidget(plot_splitter)
        splitter.addWidget(right_widget)

        splitter.setSizes([480, 920])
        insp_layout.addWidget(splitter)
        self.tabs.addTab(tab_inspector, "Target Inspector")

        # Tab 2: Sector Batch Runner
        self.batch_panel = BatchPanel()
        self.tabs.addTab(self.batch_panel, "Sector Batch Runner")

        # Toolbar
        toolbar = QToolBar("Main Controls")
        self.addToolBar(toolbar)

        self.btn_open_fits = QPushButton("Open FITS...")
        self.btn_open_fits.clicked.connect(self._on_open_fits)
        toolbar.addWidget(self.btn_open_fits)

        self.btn_open_tpf = QPushButton("Companion TPF...")
        self.btn_open_tpf.clicked.connect(self._on_open_tpf)
        toolbar.addWidget(self.btn_open_tpf)

        toolbar.addSeparator()

        toolbar.addWidget(QLabel(" Device: "))
        self.combo_device = QComboBox()
        self.combo_device.addItems(["auto", "cuda", "cpu"])
        toolbar.addWidget(self.combo_device)

        toolbar.addSeparator()

        self.btn_analyze = QPushButton("Analyze Target")
        self.btn_analyze.setStyleSheet(
            "background-color: #2E7D32; color: white; font-weight: bold; padding: 4px 10px;"
        )
        self.btn_analyze.clicked.connect(self._on_analyze_clicked)
        toolbar.addWidget(self.btn_analyze)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setStyleSheet(
            "background-color: #C62828; color: white; font-weight: bold; padding: 4px 10px;"
        )
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._bridge.cancel)
        toolbar.addWidget(self.btn_cancel)

        toolbar.addSeparator()

        self.btn_load_bundle = QPushButton("Load Bundle...")
        self.btn_load_bundle.clicked.connect(self._on_load_bundle_dialog)
        toolbar.addWidget(self.btn_load_bundle)

        self.btn_save_session = QPushButton("Save Session...")
        self.btn_save_session.clicked.connect(self._on_save_session_dialog)
        toolbar.addWidget(self.btn_save_session)

        self.btn_open_session = QPushButton("Open Session...")
        self.btn_open_session.clicked.connect(self._on_open_session_dialog)
        toolbar.addWidget(self.btn_open_session)

        self.btn_export_fig = QPushButton("Export Vetting Sheet...")
        self.btn_export_fig.clicked.connect(self._on_export_vetting_sheet_dialog)
        toolbar.addWidget(self.btn_export_fig)

        self.btn_export = QPushButton("Export Bundle...")
        self.btn_export.clicked.connect(self._on_export_bundle)
        toolbar.addWidget(self.btn_export)

        # Status Bar
        self.status_bar = self.statusBar()
        self.lbl_status = QLabel("Ready")
        self.prog_bar = QProgressBar()
        self.prog_bar.setRange(0, 100)
        self.prog_bar.setValue(0)
        self.prog_bar.setFixedWidth(200)
        self.prog_bar.setVisible(False)

        self.status_bar.addWidget(self.lbl_status, stretch=1)
        self.status_bar.addPermanentWidget(self.prog_bar)

    def _init_shortcuts(self) -> None:
        self.shortcut_open = QShortcut(QKeySequence("Ctrl+O"), self)
        self.shortcut_open.activated.connect(self._on_open_fits)

        self.shortcut_run = QShortcut(QKeySequence("Ctrl+R"), self)
        self.shortcut_run.activated.connect(self._on_analyze_clicked)

        self.shortcut_save_session = QShortcut(QKeySequence("Ctrl+S"), self)
        self.shortcut_save_session.activated.connect(self._on_save_session_dialog)

        self.shortcut_export = QShortcut(QKeySequence("Ctrl+E"), self)
        self.shortcut_export.activated.connect(self._on_export_vetting_sheet_dialog)

    def _connect_signals(self) -> None:
        # Bridge signals
        self._bridge.progress_changed.connect(self._on_progress_changed)
        self._bridge.batch_progress.connect(self.batch_panel.on_batch_progress)
        self._bridge.analysis_completed.connect(self._on_analysis_completed)
        self._bridge.analysis_failed.connect(self._on_analysis_failed)
        self._bridge.process_state_changed.connect(self._on_process_state_changed)

        # Batch panel signals
        self.batch_panel.start_batch_requested.connect(self._on_start_batch)
        self.batch_panel.cancel_batch_requested.connect(self._bridge.cancel)
        self.batch_panel.open_bundle_requested.connect(self.display_bundle_from_path)

        # Candidate table selection
        self.candidate_table.candidate_selected.connect(self._on_candidate_selected)

        # Interactive periodogram peak picking
        self.periodogram_widget.period_selected.connect(self._on_period_picked)

    def _on_period_picked(self, period: float) -> None:
        """Handle user clicking an alternative period peak on the periodogram."""
        if not self._current_bundle or not self._current_bundle.candidates:
            return
        if self._selected_candidate_index >= len(self._current_bundle.candidates):
            return
        from dataclasses import replace

        cand = self._current_bundle.candidates[self._selected_candidate_index]
        custom_cand = replace(cand, period_days=period)
        self.folded_widget.set_candidate(
            custom_cand,
            self._current_bundle.times,
            self._current_bundle.detrended_flux,
        )
        self.lbl_status.setText(f"Folded on custom period: {period:.5f} days")

    def _on_open_fits(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open TESS Light Curve FITS",
            "",
            "FITS Files (*.fits *.fits.gz);;All Files (*)",
        )
        if path:
            self._current_fits = Path(path)
            self.lbl_status.setText(f"Selected: {self._current_fits.name}")

    def _on_open_tpf(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open TESS Companion Target Pixel File (TPF)",
            "",
            "TPF FITS Files (*_tp.fits *.fits);;All Files (*)",
        )
        if path:
            self._current_tpf = Path(path)
            self.lbl_status.setText(f"Companion TPF: {self._current_tpf.name}")

    def _on_analyze_clicked(self) -> None:
        if not self._current_fits or not self._current_fits.exists():
            QMessageBox.warning(
                self, "No Input File", "Please select a valid TESS FITS file first."
            )
            return

        dev = self.combo_device.currentText()
        self._bridge.analyze_fits(
            self._current_fits,
            device=dev,
            tpf_path=self._current_tpf,
        )

    def _on_start_batch(self, source: str, device: str, workers: int) -> None:
        self._bridge.run_batch(
            source_path=source,
            device=device,
            workers=workers,
        )

    def _on_progress_changed(self, stage: str, fraction: float) -> None:
        self.lbl_status.setText(stage)
        if fraction > 0:
            self.prog_bar.setVisible(True)
            self.prog_bar.setValue(int(round(fraction * 100)))
        else:
            self.prog_bar.setVisible(False)

    def _on_process_state_changed(self, is_running: bool) -> None:
        self.btn_analyze.setEnabled(not is_running)
        self.btn_cancel.setEnabled(is_running)
        if not is_running:
            self.prog_bar.setVisible(False)
            self.batch_panel.on_batch_finished()

    def _on_analysis_completed(self, bundle_path: str, summary: dict) -> None:
        self.lbl_status.setText("Analysis completed successfully.")
        if bundle_path:
            self.display_bundle_from_path(bundle_path)

    def _on_analysis_failed(self, error_msg: str) -> None:
        self.lbl_status.setText(f"Error: {error_msg}")
        QMessageBox.critical(
            self, "Analysis Failed", f"Backend execution error:\n\n{error_msg}"
        )

    def display_bundle_from_path(self, bundle_dir: str | Path) -> None:
        """Load and render bundle in the inspector tab."""
        try:
            bundle = load_result_bundle(bundle_dir)
        except Exception as e:
            QMessageBox.warning(self, "Failed to Load Bundle", str(e))
            return

        self._current_bundle = bundle
        self.tabs.setCurrentIndex(0)  # switch to Inspector

        # Populate plots and tables
        self.lc_widget.set_bundle(bundle)
        self.periodogram_widget.set_bundle(bundle)
        self.candidate_table.set_candidates(bundle.candidates)

        if bundle.candidates:
            self._on_candidate_selected(0)
        else:
            self._selected_candidate_index = 0
            self.folded_widget.clear()
            self.classification_card.clear()
            self.vetting_card.clear()

        self.lbl_status.setText(
            f"Loaded TIC {bundle.target_id} ({len(bundle.candidates)} candidates)"
        )

    def _on_candidate_selected(self, index: int) -> None:
        if not self._current_bundle or index >= len(self._current_bundle.candidates):
            return

        self._selected_candidate_index = index
        cand = self._current_bundle.candidates[index]
        self.folded_widget.set_candidate(
            candidate=cand,
            times=self._current_bundle.times,
            detrended_flux=self._current_bundle.detrended_flux,
        )
        self.periodogram_widget.set_peak_period(cand.period_days)
        self.classification_card.set_candidate(cand)
        self.vetting_card.set_data(self._current_bundle, cand)

        if len(self._current_bundle.times) > 0:
            self.lc_widget.highlight_candidate_transits(
                period=cand.period_days,
                epoch=cand.epoch_days,
                duration_hours=cand.duration_hours,
                t_min=float(self._current_bundle.times[0]),
                t_max=float(self._current_bundle.times[-1]),
            )

    def _on_load_bundle_dialog(self) -> None:
        dir_path = QFileDialog.getExistingDirectory(self, "Select Result Bundle Directory")
        if dir_path:
            self.display_bundle_from_path(dir_path)

    def save_session(self, session_file: str | Path) -> Path:
        """Persist current desktop workspace session to a .teye.json file."""
        dest = Path(session_file).expanduser().resolve()
        if not dest.name.endswith(".teye.json") and dest.suffix != ".json":
            dest = dest.with_suffix(".teye.json")
        dest.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, Any] = {
            "schema_version": "1",
            "fits_path": str(self._current_fits) if self._current_fits else None,
            "tpf_path": str(self._current_tpf) if self._current_tpf else None,
            "device": self.combo_device.currentText(),
            "bundle_path": str(self._current_bundle.bundle_path)
            if self._current_bundle
            else None,
            "selected_candidate_index": self._selected_candidate_index,
        }
        dest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self.lbl_status.setText(f"Saved session: {dest.name}")
        return dest

    def load_session(self, session_file: str | Path) -> None:
        """Restore desktop workspace session from a .teye.json file."""
        src = Path(session_file).expanduser().resolve()
        doc = json.loads(src.read_text(encoding="utf-8"))

        dev = str(doc.get("device", "auto"))
        idx = self.combo_device.findText(dev)
        if idx >= 0:
            self.combo_device.setCurrentIndex(idx)

        fits_p = doc.get("fits_path")
        self._current_fits = Path(fits_p) if fits_p else None
        tpf_p = doc.get("tpf_path")
        self._current_tpf = Path(tpf_p) if tpf_p else None

        bundle_p = doc.get("bundle_path")
        if bundle_p and Path(bundle_p).is_dir():
            self.display_bundle_from_path(bundle_p)
            cand_idx = int(doc.get("selected_candidate_index", 0))
            if (
                self._current_bundle
                and 0 <= cand_idx < len(self._current_bundle.candidates)
            ):
                self.candidate_table.table.selectRow(cand_idx)
                self._on_candidate_selected(cand_idx)

        self.lbl_status.setText(f"Loaded session: {src.name}")

    def _on_save_session_dialog(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save TransitEye Session",
            "workspace.teye.json",
            "TransitEye Session (*.teye.json *.json)",
        )
        if path:
            self.save_session(path)

    def _on_open_session_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open TransitEye Session",
            "",
            "TransitEye Session (*.teye.json *.json);;All Files (*)",
        )
        if path:
            try:
                self.load_session(path)
            except Exception as e:
                QMessageBox.warning(self, "Failed to Load Session", str(e))

    def export_vetting_sheet(self, output_file: str | Path) -> bool:
        """Export publication-grade PNG/PDF/SVG vetting summary sheet via backend CLI."""
        if not self._current_bundle:
            return False
        ok = self._bridge.export_vetting_sheet(
            self._current_bundle.bundle_path,
            output_file,
            candidate_index=self._selected_candidate_index,
        )
        if ok:
            self.lbl_status.setText(f"Exported vetting sheet: {Path(output_file).name}")
        return ok

    def _on_export_vetting_sheet_dialog(self) -> None:
        if not self._current_bundle:
            QMessageBox.information(
                self, "Export Vetting Sheet", "No active result bundle loaded to export."
            )
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Diagnostic Vetting Sheet",
            f"vetting_{self._current_bundle.target_id}.pdf",
            "Figures (*.pdf *.png *.svg)",
        )
        if path:
            ok = self.export_vetting_sheet(path)
            if ok:
                QMessageBox.information(
                    self, "Export Complete", f"Vetting sheet saved to:\n{path}"
                )
            else:
                QMessageBox.warning(
                    self, "Export Failed", "Failed to render diagnostic vetting sheet."
                )

    def _on_export_bundle(self) -> None:
        if not self._current_bundle:
            QMessageBox.information(
                self, "Export", "No active result bundle loaded to export."
            )
            return

        out_dir = QFileDialog.getExistingDirectory(
            self, "Export Bundle Destination Directory"
        )
        if out_dir:
            dest = Path(out_dir) / self._current_bundle.bundle_path.name
            shutil.copytree(self._current_bundle.bundle_path, dest, dirs_exist_ok=True)
            QMessageBox.information(
                self, "Export Complete", f"Bundle exported to:\n{dest}"
            )
