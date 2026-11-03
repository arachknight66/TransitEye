"""Main workstation window for TransitEye Desktop."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTextBrowser,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from transiteye_desktop.bridge import BackendProcessBridge
from transiteye_desktop.bundle_loader import LoadedBundle, load_result_bundle
from transiteye_desktop.sample import resolve_sample
from transiteye_desktop.widgets.batch_panel import BatchPanel
from transiteye_desktop.widgets.candidate_table_widget import CandidateTableWidget
from transiteye_desktop.widgets.classification_card import ClassificationCard
from transiteye_desktop.widgets.folded_transit_widget import FoldedTransitWidget
from transiteye_desktop.widgets.light_curve_widget import LightCurvePlotWidget
from transiteye_desktop.widgets.periodogram_widget import PeriodogramWidget
from transiteye_desktop.widgets.vetting_card import VettingCard

HISTORICAL_MODEL = Path(".transiteye-workspace/models/morphology_real_catalog_v1")
HISTORICAL_CALIBRATOR_SHA256 = "59338074204c8545bad72711a82e45713a29f811ecd78cb3a59194c0d4a5c9a7"
HISTORICAL_MODEL_SHA256 = {
    "model_metadata.json": "aa930ff02dd56a0da22be50ef5386989f43615cd1c6f26524f4a3fd04bf093a3",
    "estimator.joblib": "de139441237410d11603737231f7bd842c6806b90c5d24f35e85319a57aca72b",
    "calibrator.joblib": HISTORICAL_CALIBRATOR_SHA256,
}


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

        search_options = QHBoxLayout()
        search_options.addWidget(QLabel("Search periods (days):"))
        self.spin_min_period = QDoubleSpinBox()
        self.spin_min_period.setRange(0.1, 500.0)
        self.spin_min_period.setDecimals(2)
        self.spin_min_period.setValue(0.3)
        self.spin_min_period.setToolTip("Shortest trial period. This also affects the current transit-duration grid.")
        search_options.addWidget(self.spin_min_period)
        search_options.addWidget(QLabel("to"))
        self.spin_max_period = QDoubleSpinBox()
        self.spin_max_period.setRange(0.0, 500.0)
        self.spin_max_period.setDecimals(2)
        self.spin_max_period.setSpecialValueText("Automatic")
        self.spin_max_period.setToolTip("Automatic searches up to half the observation baseline. Longer periods may have too few events to identify aliases reliably.")
        search_options.addWidget(self.spin_max_period)
        left_layout.addLayout(search_options)

        self.check_historical_model = QCheckBox("Use historical morphology model (experimental)")
        self.check_historical_model.setToolTip(
            "Opt in to the local historical four-class model. Its performance and calibration "
            "have not been validated on the current pipeline."
        )
        left_layout.addWidget(self.check_historical_model)

        self.lbl_search_summary = QLabel("Load and analyze a light curve to see the search range.")
        self.lbl_search_summary.setWordWrap(True)
        left_layout.addWidget(self.lbl_search_summary)
        self.lbl_candidate_details = QLabel("")
        self.lbl_candidate_details.setWordWrap(True)
        left_layout.addWidget(self.lbl_candidate_details)
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

        self.btn_graph_guide = QPushButton("Graph Guide")
        self.btn_graph_guide.clicked.connect(self._on_graph_guide)
        toolbar.addWidget(self.btn_graph_guide)

        # Status Bar
        self.status_bar = self.statusBar()
        self.lbl_status = QLabel("Ready")
        self.prog_bar = QProgressBar()
        self.prog_bar.setRange(0, 100)
        self.prog_bar.setValue(0)
        self.prog_bar.setFixedWidth(200)
        self.prog_bar.setVisible(False)

        self.status_bar.addWidget(self.lbl_status, stretch=1)
        self.lbl_sample = QLabel("Sample: real TESS data")
        self.lbl_sample.setVisible(False)
        self.status_bar.addPermanentWidget(self.lbl_sample)
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

        self.shortcut_sample = QShortcut(QKeySequence("Ctrl+Shift+D"), self)
        self.shortcut_sample.activated.connect(self.start_sample)

    def _connect_signals(self) -> None:
        # Bridge signals
        self._bridge.progress_changed.connect(self._on_progress_changed)
        self._bridge.batch_progress.connect(self.batch_panel.on_batch_progress)
        self._bridge.batch_snapshot.connect(self.batch_panel.on_batch_snapshot)
        self._bridge.export_completed.connect(
            lambda path: self.lbl_status.setText(f"Exported vetting sheet: {Path(path).name}")
        )
        self._bridge.export_failed.connect(
            lambda error: QMessageBox.warning(self, "Export Failed", error)
        )
        self._bridge.analysis_completed.connect(self._on_analysis_completed)
        self._bridge.analysis_failed.connect(self._on_analysis_failed)
        self._bridge.process_state_changed.connect(self._on_process_state_changed)

        # Batch panel signals
        self.batch_panel.start_batch_requested.connect(self._on_start_batch)
        self.batch_panel.cancel_batch_requested.connect(self._bridge.cancel)
        self.batch_panel.control_requested.connect(self._on_batch_control)
        self.batch_panel.resume_requested.connect(self._on_resume_batch)
        self.batch_panel.inspect_requested.connect(self._bridge.inspect_batch)
        self.batch_panel.open_bundle_requested.connect(self._on_open_batch_bundle)

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
        custom_cand = replace(
            cand, period_days=period, trapezoid_model_curve=None, batman_model_curve=None,
        )
        self.folded_widget.set_candidate(
            custom_cand,
            self._current_bundle.times,
            self._current_bundle.detrended_flux,
        )
        self.lbl_status.setText(f"Preview folded at {period:.5f} days; no fit was performed.")

    def _on_graph_guide(self) -> None:
        """Explain the plots without changing or interpreting scientific results."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Graph Guide")
        dialog.resize(680, 560)
        layout = QVBoxLayout(dialog)
        text = QTextBrowser(dialog)
        text.setOpenExternalLinks(False)
        text.setHtml("""
        <h2>Reading the TransitEye graphs</h2>
        <p>Select a candidate in the table to see its folded profile and fit. A candidate is a
        signal to investigate, not confirmation of a planet.</p>
        <h3>Brightness and trend</h3>
        <p>Gray shows the input PDCSAP or SAP flux; red shows the estimated slow trend.
        The time axis uses BJD − 2457000 days. A line across a gap does not represent observations.</p>
        <h3>Detrended brightness</h3>
        <p>Blue shows normalized brightness after trend removal; 1 is the approximate baseline.
        Shaded windows mark predicted events for the selected candidate. Gaps and noise can make
        apparent dips misleading.</p>
        <h3>Box Least-Squares (BLS)</h3>
        <p>Peaks show periods at which a repeating box-shaped dip fits the data. Red marks the
        candidate period; orange marks half and twice that period. Power is not a planet probability.
        The period axis is logarithmic.</p>
        <h3>Lomb–Scargle</h3>
        <p>Peaks indicate approximately sinusoidal periodic variability, which may have several
        causes. The period axis is logarithmic; long periods and gaps require caution.</p>
        <h3>Phase-folded profile</h3>
        <p>Observations are aligned at the selected period. Smaller points are observations and
        larger points are binned medians. Red is the trapezoid fit; dashed yellow is the optional
        physical fit. A dip around zero can indicate a repeating signal, but aliases remain possible.</p>
        <h3>Fit residuals</h3>
        <p>Observed brightness minus the available fitted model; the physical model is used when
        present, otherwise the trapezoid. Patterns around zero can show a poor fit. A good-looking
        fit does not establish the signal's cause.</p>
        <p>Clicking a periodogram previews a different folded period; it does not perform a new fit.</p>
        """)
        layout.addWidget(text)
        dialog.exec()

    def _on_open_fits(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open TESS Light Curve FITS",
            "",
            "FITS Files (*.fits *.fits.gz);;All Files (*)",
        )
        if path:
            self._current_fits = Path(path)
            if not self.lbl_sample.isHidden():
                self.check_historical_model.setChecked(False)
            self.lbl_sample.setVisible(False)
            self.lbl_status.setText(f"Selected: {self._current_fits.name}")

    def start_sample(self, path: Path | None = None) -> None:
        """Run an actual FITS input through the ordinary analysis process."""
        if self._bridge.is_running:
            self.lbl_status.setText("Another analysis is already running.")
            return
        try:
            sample = resolve_sample(path)
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "Sample Unavailable", str(error))
            return

        previous_selection = self.check_historical_model.isChecked()
        self.check_historical_model.setChecked(True)
        if self._model_options() is None:
            self.check_historical_model.setChecked(previous_selection)
            return

        self._current_fits = sample
        self._current_tpf = None
        self.combo_device.setCurrentText("cpu")
        self.spin_min_period.setValue(1.0 if path is None else 0.3)
        self.spin_max_period.setValue(20.0 if path is None else 0.0)
        self._clear_result("Analyzing sample light curve.")
        self.lbl_sample.setVisible(True)
        self.tabs.setCurrentIndex(0)
        self._on_analyze_clicked()

    def _clear_result(self, message: str) -> None:
        self._current_bundle = None
        self._selected_candidate_index = 0
        self.lc_widget.clear()
        self.periodogram_widget.clear()
        self.candidate_table.clear()
        self.folded_widget.clear()
        self.classification_card.clear()
        self.classification_card.set_model_note(None)
        self.vetting_card.clear()
        self.lbl_candidate_details.clear()
        self.lbl_search_summary.setText(message)

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
        if self._bridge.is_running:
            self.lbl_status.setText("Another analysis is already running.")
            return
        if not self._current_fits or not self._current_fits.exists():
            QMessageBox.warning(
                self, "No Input File", "Please select a valid TESS FITS file first."
            )
            return

        dev = self.combo_device.currentText()
        minimum = self.spin_min_period.value()
        maximum = self.spin_max_period.value() or None
        if maximum is not None and maximum <= minimum:
            QMessageBox.warning(self, "Invalid Search Range", "The maximum period must exceed the minimum period.")
            return
        model_options = self._model_options()
        if model_options is None:
            return
        self._bridge.analyze_fits(
            self._current_fits,
            device=dev,
            tpf_path=self._current_tpf,
            min_period_days=minimum,
            max_period_days=maximum,
            **model_options,
        )

    def _model_options(self) -> dict[str, Any] | None:
        if not self.check_historical_model.isChecked():
            return {}
        model_dir = HISTORICAL_MODEL.resolve()
        if not all((model_dir / name).is_file() for name in HISTORICAL_MODEL_SHA256):
            QMessageBox.warning(
                self, "Classifier Unavailable",
                f"The historical classifier bundle is missing from {model_dir}. "
                "Uncheck the model option to analyze without classification.",
            )
            return None
        try:
            for name, expected in HISTORICAL_MODEL_SHA256.items():
                with (model_dir / name).open("rb") as source:
                    actual = hashlib.file_digest(source, "sha256").hexdigest()
                if actual != expected:
                    raise ValueError(f"Checksum mismatch for {name}.")
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "Classifier Unavailable", str(error))
            return None
        return {"model_dir": model_dir, "model_calibrator_sha256": HISTORICAL_CALIBRATOR_SHA256}

    def _on_start_batch(self, source: str, device: str, workers: int) -> None:
        if self._bridge.is_running:
            self.batch_panel.on_batch_finished()
            self.batch_panel.btn_start.setEnabled(False)
            self.lbl_status.setText("Another analysis is already running.")
            return
        minimum = self.spin_min_period.value()
        maximum = self.spin_max_period.value() or None
        if maximum is not None and maximum <= minimum:
            self.batch_panel.on_batch_finished()
            QMessageBox.warning(self, "Invalid Search Range", "The maximum period must exceed the minimum period.")
            return
        model_options = self._model_options()
        if model_options is None:
            self.batch_panel.on_batch_finished()
            return
        self._bridge.run_batch(
            source_path=source,
            device=device,
            workers=workers,
            min_period_days=minimum,
            max_period_days=maximum,
            **model_options,
        )
        self.lbl_sample.setVisible(False)
        self.batch_panel.txt_job_id.setText(self._bridge.batch_job_id or "")

    def _on_batch_control(self, action: str, job_id: str) -> None:
        if job_id:
            self._bridge.control_batch(action, job_id)

    def _on_resume_batch(self, job_id: str, workers: int) -> None:
        if job_id and not self._bridge.is_running:
            self.lbl_sample.setVisible(False)
            self._bridge.resume_batch(job_id, workers)

    def _on_open_batch_bundle(self, path: str) -> None:
        self.display_bundle_from_path(path)
        self.lbl_sample.setVisible(False)

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
        self.batch_panel.btn_start.setEnabled(not is_running)
        if not is_running:
            self.prog_bar.setVisible(False)
            self.batch_panel.on_batch_finished()

    def _on_analysis_completed(self, bundle_path: str, summary: dict) -> None:
        if "job_id" in summary:
            self.lbl_status.setText(
                f"Batch {summary.get('status', 'finished')}: "
                f"{summary.get('completed_targets', 0)} completed, {summary.get('failed_targets', 0)} failed."
            )
        else:
            self.lbl_status.setText("Analysis completed successfully.")
        if bundle_path:
            self.display_bundle_from_path(bundle_path)

    def _on_analysis_failed(self, error_msg: str) -> None:
        self.lbl_status.setText(f"Error: {error_msg}")
        QMessageBox.critical(
            self, "Analysis Failed", f"Backend execution error:\n\n{error_msg}"
        )

    def display_bundle_from_path(self, bundle_dir: str | Path, *, raise_on_error: bool = False) -> None:
        """Load and render bundle in the inspector tab."""
        try:
            bundle = load_result_bundle(bundle_dir)
        except Exception as e:
            if raise_on_error:
                raise ValueError(f"Could not load the saved result bundle {bundle_dir}: {e}") from e
            QMessageBox.warning(self, "Failed to Load Bundle", str(e))
            return

        self._current_bundle = bundle
        self.classification_card.set_model_note(bundle.metadata.get("model_id"))
        self.tabs.setCurrentIndex(0)  # switch to Inspector

        # Populate plots and tables
        self.lc_widget.set_bundle(bundle)
        self.periodogram_widget.set_bundle(bundle)
        self.candidate_table.set_candidates(bundle.candidates)
        if len(bundle.bls_periods):
            period_range = f"{min(bundle.bls_periods):.2f}–{max(bundle.bls_periods):.2f} days"
            if bundle.status == "no_significant_signal":
                self.lbl_search_summary.setText(
                    f"No candidate exceeded the detection threshold. Searched periods: {period_range}. "
                    "Signals outside this range were not tested."
                )
            else:
                self.lbl_search_summary.setText(f"Searched periods: {period_range}.")
        else:
            self.lbl_search_summary.setText(f"Search status: {bundle.status.replace('_', ' ')}.")

        if bundle.candidates:
            self._on_candidate_selected(0)
        else:
            self._selected_candidate_index = 0
            self.lbl_candidate_details.clear()
            self.folded_widget.clear()
            self.classification_card.clear()
            self.vetting_card.clear()

        failed_count = sum("fit_failed" in candidate.flags for candidate in bundle.candidates)
        self.lbl_status.setText(
            f"Loaded TIC {bundle.target_id} ({len(bundle.candidates)} candidates; "
            f"{failed_count} fit failures)"
        )

    def _on_candidate_selected(self, index: int) -> None:
        if not self._current_bundle or index >= len(self._current_bundle.candidates):
            return

        self._selected_candidate_index = index
        cand = self._current_bundle.candidates[index]
        alias_text = ", ".join(f"{value:.4f} d" for value in cand.aliases_days) or "none reported"
        flags = ", ".join(cand.flags) or "none"
        warning = " Two-event period may be ambiguous." if "weak_two_event" in cand.flags else ""
        if "physical_quadrature_unvalidated" in cand.flags:
            warning += " Physical-model exposure integration is not validated."
        if "physical_fit_failed" in cand.flags:
            warning += " Physical-model fit failed."
        self.lbl_candidate_details.setText(f"Candidate flags: {flags}.{warning} Aliases: {alias_text}.")
        if "fit_failed" in cand.flags:
            errors = self._current_bundle.metadata.get("candidate_errors") or {}
            detail = errors.get(cand.candidate_id, "fit failed")
            self.lbl_status.setText(f"Fit failed; detector estimates are preliminary: {detail}")
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
            self.lbl_sample.setVisible(False)

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
            "search_min_period_days": self.spin_min_period.value(),
            "search_max_period_days": self.spin_max_period.value() or None,
            "bundle_path": str(self._current_bundle.bundle_path)
            if self._current_bundle
            else None,
            "selected_candidate_index": self._selected_candidate_index,
        }
        dest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self.lbl_status.setText(f"Saved session: {dest}")
        return dest

    def load_session(self, session_file: str | Path) -> None:
        """Restore desktop workspace session from a .teye.json file."""
        src = Path(session_file).expanduser().resolve()
        doc = json.loads(src.read_text(encoding="utf-8"))
        bundle_p = doc.get("bundle_path")
        if bundle_p and not Path(bundle_p).is_dir():
            raise FileNotFoundError(f"The result bundle saved in this session was not found: {bundle_p}")

        self.check_historical_model.setChecked(False)
        self.lbl_sample.setVisible(False)

        dev = str(doc.get("device", "auto"))
        idx = self.combo_device.findText(dev)
        if idx >= 0:
            self.combo_device.setCurrentIndex(idx)
        self.spin_min_period.setValue(float(doc.get("search_min_period_days", 0.3)))
        self.spin_max_period.setValue(float(doc.get("search_max_period_days") or 0.0))

        fits_p = doc.get("fits_path")
        self._current_fits = Path(fits_p) if fits_p else None
        tpf_p = doc.get("tpf_path")
        self._current_tpf = Path(tpf_p) if tpf_p else None

        if bundle_p and Path(bundle_p).is_dir():
            self.display_bundle_from_path(bundle_p, raise_on_error=True)
            cand_idx = int(doc.get("selected_candidate_index", 0))
            if (
                self._current_bundle
                and 0 <= cand_idx < len(self._current_bundle.candidates)
            ):
                self.candidate_table.table.selectRow(cand_idx)
                self._on_candidate_selected(cand_idx)
            self.lbl_status.setText(f"Loaded session: {src.name}")
        else:
            self._clear_result("This session has no saved analysis result.")
            self.lbl_status.setText(f"Loaded session without a result: {src.name}")

    def _on_save_session_dialog(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save TransitEye Session",
            "workspace.teye.json",
            "TransitEye Session (*.teye.json *.json)",
        )
        if path:
            try:
                saved = self.save_session(path)
                QMessageBox.information(self, "Session Saved", f"Session saved to:\n{saved}")
            except (OSError, ValueError) as error:
                QMessageBox.warning(self, "Failed to Save Session", str(error))

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
            self.lbl_status.setText(f"Exporting vetting sheet: {Path(path).name}")
            self._bridge.start_vetting_sheet_export(
                self._current_bundle.bundle_path, path, self._selected_candidate_index,
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
