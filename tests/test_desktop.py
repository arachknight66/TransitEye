"""Headless GUI and component tests for the desktop package."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from astropy.io import fits
from PyQt6.QtCore import QRectF, Qt, QTimer
from PyQt6.QtWidgets import QApplication, QTextBrowser
from transiteye_desktop.bridge import BackendProcessBridge
from transiteye_desktop.bundle_loader import LoadedBundle, load_result_bundle
from transiteye_desktop.downsample import lttb_downsample, min_max_downsample
from transiteye_desktop.main_window import TransitEyeMainWindow
from transiteye_desktop.widgets.batch_panel import BatchPanel
from transiteye_desktop.widgets.safe_graphics import SafeInfiniteLine

from transiteye.characterization.fitting import evaluate_trapezoid
from transiteye.execution.service import analyze_light_curve
from transiteye.io.fits import read_tess_light_curve


def test_infinite_line_handles_view_disappearing_during_bounds_calculation(qtbot, monkeypatch) -> None:
    line = SafeInfiniteLine()
    monkeypatch.setattr(line, "viewRect", lambda: QRectF(0, 0, 10, 10))
    monkeypatch.setattr(line, "pixelVectors", lambda **kwargs: (None, None))
    monkeypatch.setattr(line, "getViewBox", lambda: None)
    assert line._computeBoundingRect().isEmpty()
    # An unrelated error while still attached must not be hidden.
    monkeypatch.setattr(line, "getViewBox", lambda: object())
    with pytest.raises(AttributeError):
        line._computeBoundingRect()


def test_graph_guide_opens_and_explains_preview(qtbot) -> None:
    window = TransitEyeMainWindow()
    qtbot.addWidget(window)
    observed = []

    def inspect_and_close() -> None:
        dialog = QApplication.activeModalWidget()
        assert dialog is not None
        observed.append(dialog.findChild(QTextBrowser).toPlainText())
        dialog.close()

    QTimer.singleShot(0, inspect_and_close)
    window.btn_graph_guide.click()
    assert "Box Least-Squares" in observed[0]
    assert "does not perform a new fit" in observed[0]


def test_repeated_window_close_with_loaded_plots(qtbot, tmp_path: Path) -> None:
    bundle_dir = _create_canonical_bundle(tmp_path / "close-reopen-bundle")
    for _ in range(5):
        window = TransitEyeMainWindow()
        window.show()
        window.display_bundle_from_path(bundle_dir)
        qtbot.wait(20)
        window.close()
        window.deleteLater()
        qtbot.wait(20)


def test_batch_panel_progress_counts_all_targets_and_failures(qtbot) -> None:
    panel = BatchPanel()
    qtbot.addWidget(panel)
    panel.on_batch_progress({"target_id": "sector1/product", "status": "failed", "error_message": "bad FITS",
                             "completed_targets": 0, "failed_targets": 1, "total_targets": 4})
    assert panel.prog_bar.value() == 25
    assert "1 failed" in panel.lbl_stats.text()
    assert panel.table.item(0, 4).text() == "bad FITS"


def test_batch_panel_restores_persisted_failures(qtbot) -> None:
    panel = BatchPanel()
    qtbot.addWidget(panel)
    panel.txt_job_id.setText("saved")
    panel.on_batch_snapshot({"job_id": "saved", "task_offset": 0, "total_targets": 2,
                             "completed_targets": 0, "failed_targets": 1,
                             "tasks": [{"target_id": "broken", "status": "failed",
                                        "error_message": "checksum mismatch"}]})
    assert panel.table.item(0, 4).text() == "checksum mismatch"
    assert panel.prog_bar.value() == 50
    assert panel.btn_more.isEnabled()


def _create_canonical_bundle(bundle_dir: Path) -> Path:
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # 1. result.json
    res_data = {
        "analysis_id": "test_analysis_101",
        "target_id": "99988877",
        "status": "complete",
        "schema_version": "2",
        "provenance": {"target_id": "99988877", "sector": 1},
        "classifications": {
            "cand_01": {
                "predicted_class": "transit_like",
                "abstained": False,
                "source_assessment": "target_consistent",
                "probabilities": {
                    "transit_like": 0.92,
                    "eclipse_like": 0.04,
                    "stellar_variability": 0.02,
                    "other_artifact": 0.02,
                },
            }
        },
        "parameter_intervals": {
            "cand_01": {
                "period_days": {
                    "68": {"lower": 2.499, "upper": 2.501, "coverage": 0.68},
                    "95": {"lower": 2.498, "upper": 2.502, "coverage": 0.95},
                },
                "depth_ppm": {
                    "68": {"lower": 9800.0, "upper": 10200.0, "coverage": 0.68},
                    "95": {"lower": 9600.0, "upper": 10400.0, "coverage": 0.95},
                },
            }
        },
    }
    (bundle_dir / "result.json").write_text(json.dumps(res_data), encoding="utf-8")

    # 2. arrays.npz with canonical backend keys
    t = np.linspace(1.0, 10.0, 2000)
    raw = np.ones_like(t)
    detrended = np.ones_like(t)
    detrended[500:550] = 0.99
    trend = np.ones_like(t)
    err = np.full_like(t, 0.001)
    exclusion_flags = np.zeros(len(t), dtype=np.int32)
    exclusion_flags[10:15] = 1
    periods = np.linspace(0.5, 5.0, 100)
    powers = np.random.default_rng(42).uniform(0.0, 10.0, 100)
    powers[30] = 50.0
    ls_periods = np.linspace(0.5, 5.0, 100)
    ls_powers = np.random.default_rng(43).uniform(0.0, 0.5, 100)
    ls_powers[40] = 0.85
    trap_model = np.ones_like(t)
    trap_model[500:550] = 0.99
    tpf = np.ones((11, 11))
    diff = np.zeros((11, 11))

    np.savez_compressed(
        bundle_dir / "arrays.npz",
        time_days=t,
        flux_raw=raw,
        flux_detrended=detrended,
        trend_flux=trend,
        flux_error=err,
        exclusion_flags=exclusion_flags,
        bls_periods=periods,
        bls_powers=powers,
        ls_periods=ls_periods,
        ls_powers=ls_powers,
        cand_00_trapezoid_model=trap_model,
        tpf_stamp=tpf,
        tpf_diff=diff,
    )

    # 3. candidates.parquet with canonical backend columns
    cand_data = {
        "candidate_id": ["cand_01"],
        "period_days": [2.5],
        "epoch_days": [3.2],
        "duration_hours": [2.4],
        "measured_depth_ppm": [10000.0],
        "dilution_corrected_depth_ppm": [10526.3],
        "observed_event_count": [3],
        "detection_snr": [15.2],
        "search_significance": [0.995],
        "phase_coverage": [0.95],
        "aliases_days": [[1.25, 5.0]],
        "predicted_class": ["transit_like"],
        "calibrated_confidence": [0.92],
        "prob_transit_like": [0.92],
        "prob_eclipse_like": [0.04],
        "prob_stellar_variability": [0.02],
        "prob_other_artifact": [0.02],
        "abstained": [False],
        "source_assessment": ["target_consistent"],
        "flags": [["FLAG_OK", "physical_quadrature_unvalidated", "physical_fit_failed"]],
        "centroid_shift_sigma": [0.8],
        "crowdsap_value": [0.95],
        "sap_to_pdcsap_depth_ratio": [1.02],
    }
    schema = pa.schema(
        [
            ("candidate_id", pa.string()),
            ("period_days", pa.float64()),
            ("epoch_days", pa.float64()),
            ("duration_hours", pa.float64()),
            ("measured_depth_ppm", pa.float64()),
            ("dilution_corrected_depth_ppm", pa.float64()),
            ("observed_event_count", pa.int64()),
            ("detection_snr", pa.float64()),
            ("search_significance", pa.float64()),
            ("phase_coverage", pa.float64()),
            ("aliases_days", pa.list_(pa.float64())),
            ("predicted_class", pa.string()),
            ("calibrated_confidence", pa.float64()),
            ("prob_transit_like", pa.float64()),
            ("prob_eclipse_like", pa.float64()),
            ("prob_stellar_variability", pa.float64()),
            ("prob_other_artifact", pa.float64()),
            ("abstained", pa.bool_()),
            ("source_assessment", pa.string()),
            ("flags", pa.list_(pa.string())),
            ("centroid_shift_sigma", pa.float64()),
            ("crowdsap_value", pa.float64()),
            ("sap_to_pdcsap_depth_ratio", pa.float64()),
        ]
    )
    tbl = pa.Table.from_pydict(cand_data, schema=schema)
    pq.write_table(tbl, bundle_dir / "candidates.parquet")

    return bundle_dir


def test_downsampling_algorithms() -> None:
    """Verify downsampling preserves extreme dips and reduces point counts."""
    t = np.linspace(0.0, 100.0, 10000)
    y = np.ones_like(t)
    y[5000:5010] = 0.95

    # 1. Min-Max
    t_mm, y_mm = min_max_downsample(t, y, target_points=1000)
    assert len(t_mm) <= 1200
    assert np.min(y_mm) <= 0.951

    # 2. LTTB
    t_lt, y_lt = lttb_downsample(t, y, target_points=1000)
    assert len(t_lt) == 1000
    assert np.min(y_lt) <= 0.96


def test_bundle_loader(tmp_path: Path) -> None:
    """Verify bundle loader parses canonical result.json, arrays.npz, and parquet tables."""
    bundle_dir = _create_canonical_bundle(tmp_path / "test_bundle")
    bundle = load_result_bundle(bundle_dir)

    assert isinstance(bundle, LoadedBundle)
    assert bundle.target_id == "99988877"
    assert bundle.analysis_id == "test_analysis_101"
    assert len(bundle.times) == 2000
    assert len(bundle.ls_periods) == 100
    assert len(bundle.candidates) == 1

    cand = bundle.candidates[0]
    assert cand.candidate_id == "cand_01"
    assert cand.period_days == 2.5
    assert cand.depth_ppm == 10000.0
    assert cand.dilution_corrected_depth_ppm == pytest.approx(10526.3)
    assert cand.predicted_class == "transit_like"
    assert cand.calibrated_confidence == 0.92
    assert cand.probabilities["transit_like"] == pytest.approx(0.92)
    assert cand.source_assessment == "target_consistent"
    assert cand.trapezoid_model_curve is not None
    assert bundle.tpf_stamp is not None


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_desktop_main_window_session_and_exports(qtbot, tmp_path: Path) -> None:
    """Verify PyQt6 main window displays canonical bundles, saves/loads sessions, and exports figures."""
    bundle_dir = _create_canonical_bundle(tmp_path / "gui_bundle")

    window = TransitEyeMainWindow()
    qtbot.addWidget(window)

    assert window.tabs.count() == 2
    assert "TransitEye" in window.windowTitle()

    # Load bundle into UI
    window.display_bundle_from_path(bundle_dir)

    # Check candidate table
    assert window.candidate_table.table.rowCount() == 1
    item = window.candidate_table.table.item(0, 1)
    assert item is not None
    assert item.text() == "cand_01"
    assert "FLAG_OK" in window.lbl_candidate_details.text()
    assert "Physical-model exposure integration is not validated" in window.lbl_candidate_details.text()
    assert "Physical-model fit failed" in window.lbl_candidate_details.text()
    assert "1.2500 d" in window.lbl_candidate_details.text()
    assert "5.0000 d" in window.lbl_candidate_details.text()

    # Check classification card & probabilities
    assert window.classification_card.lbl_badge.text() == "TRANSIT-LIKE"
    assert window.classification_card.bars["transit_like"][1].value() == 92

    # Check vetting card intervals and dilution depth
    assert window.vetting_card.lbl_badge.text() == "TARGET CONSISTENT"
    assert "10526.3" in window.vetting_card.lbl_depths.text()
    assert "68%" in window.vetting_card.lbl_period_ci.text()
    assert "95%" in window.vetting_card.lbl_period_ci.text()

    # Check BLS and LS periodogram lines
    window.candidate_table.table.selectRow(0)
    assert window.periodogram_widget.peak_line.isVisible()
    assert window.periodogram_widget.ls_peak_line.isVisible()
    assert window.periodogram_widget.peak_line.value() == pytest.approx(math.log10(2.5))
    ls = window._current_bundle
    assert window.periodogram_widget.ls_peak_line.value() == pytest.approx(
        math.log10(float(ls.ls_periods[ls.ls_powers.argmax()]))
    )

    # Folding an alternate period is a visual preview, not a refit of the old model.
    window._on_period_picked(5.0)
    assert "no fit was performed" in window.lbl_status.text()
    assert window.folded_widget.model_curve.getData()[0] is None
    assert len(window.folded_widget.res_scatter.points()) == 0
    assert window._current_bundle.candidates[0].period_days == 2.5
    window._on_candidate_selected(0)

    # Verify session save and load (.teye.json)
    window.spin_min_period.setValue(1.0)
    window.spin_max_period.setValue(20.0)
    session_file = tmp_path / "test_session.teye.json"
    saved_path = window.save_session(session_file)
    assert saved_path.is_file()

    window2 = TransitEyeMainWindow()
    qtbot.addWidget(window2)
    window2.load_session(saved_path)
    assert window2._current_bundle is not None
    assert window2._current_bundle.target_id == "99988877"
    assert window2.spin_min_period.value() == 1.0
    assert window2.spin_max_period.value() == 20.0

    # Verify PNG/PDF vetting sheet export via subprocess bridge
    out_png = tmp_path / "exported_vetting.png"
    assert window.export_vetting_sheet(out_png) is True
    assert out_png.is_file()
    assert out_png.stat().st_size > 10000


def test_no_candidate_explains_search_range_and_missing_probabilities(qtbot, tmp_path: Path) -> None:
    bundle = _create_canonical_bundle(tmp_path / "empty")
    doc_path = bundle / "result.json"
    doc = json.loads(doc_path.read_text())
    doc["status"] = "no_significant_signal"
    doc_path.write_text(json.dumps(doc))
    parquet = bundle / "candidates.parquet"
    pq.write_table(pq.read_table(parquet).slice(0, 0), parquet)
    window = TransitEyeMainWindow()
    qtbot.addWidget(window)
    window.display_bundle_from_path(bundle)
    assert "No candidate exceeded" in window.lbl_search_summary.text()
    assert "Signals outside this range were not tested" in window.lbl_search_summary.text()
    assert all(bar.format() == "—" for _, bar in window.classification_card.bars.values())


@pytest.fixture
def sample_model(tmp_path: Path, monkeypatch) -> Path:
    import hashlib

    model = tmp_path / "model"
    model.mkdir()
    names = ("model_metadata.json", "estimator.joblib", "calibrator.joblib")
    for name in names:
        (model / name).touch()
    monkeypatch.setattr("transiteye_desktop.main_window.HISTORICAL_MODEL", model)
    monkeypatch.setattr("transiteye_desktop.main_window.HISTORICAL_MODEL_SHA256",
                        dict.fromkeys(names, hashlib.sha256(b"").hexdigest()))
    return model


def test_sample_runs_normal_backend_and_clears_old_results(qtbot, tmp_path: Path, monkeypatch, sample_model: Path) -> None:
    from transiteye_desktop.main_window import HISTORICAL_CALIBRATOR_SHA256

    window = TransitEyeMainWindow()
    qtbot.addWidget(window)
    window.display_bundle_from_path(_create_canonical_bundle(tmp_path / "old"))
    sample = tmp_path / "sample.fits"
    sample.write_bytes(b"example input")
    calls = []
    monkeypatch.setattr(window._bridge, "analyze_fits", lambda *args, **kwargs: calls.append((args, kwargs)))
    window._current_tpf = tmp_path / "old_tpf.fits"

    window.start_sample(sample)

    assert calls == [((sample.resolve(),), {"device": "cpu", "tpf_path": None,
                      "min_period_days": 0.3, "max_period_days": None,
                      "model_dir": sample_model, "model_calibrator_sha256": HISTORICAL_CALIBRATOR_SHA256})]
    assert window.check_historical_model.isChecked()
    assert window._current_bundle is None
    assert window._current_tpf is None
    assert not window.lbl_sample.isHidden()
    assert window.lbl_sample.text() == "Sample: real TESS data"
    assert window.candidate_table.table.rowCount() == 0


def test_missing_sample_preserves_existing_result(qtbot, tmp_path: Path, monkeypatch) -> None:
    window = TransitEyeMainWindow()
    qtbot.addWidget(window)
    window.display_bundle_from_path(_create_canonical_bundle(tmp_path / "old"))
    original = window._current_bundle
    warnings = []
    monkeypatch.setattr("transiteye_desktop.main_window.QMessageBox.warning",
                        lambda *args: warnings.append(args))
    window.start_sample(tmp_path / "missing.fits")
    assert window._current_bundle is original
    assert warnings and "not found" in warnings[0][2]
    assert window.lbl_sample.isHidden()


def test_default_sample_range_and_busy_guard(qtbot, tmp_path: Path, monkeypatch, sample_model: Path) -> None:
    window = TransitEyeMainWindow()
    qtbot.addWidget(window)
    sample = tmp_path / "sample.fits"
    sample.write_bytes(b"input")
    monkeypatch.setattr("transiteye_desktop.main_window.resolve_sample", lambda path: sample)
    calls = []
    monkeypatch.setattr(window._bridge, "analyze_fits", lambda *args, **kwargs: calls.append(kwargs))
    window.start_sample()
    assert calls[0]["min_period_days"] == 1.0
    assert calls[0]["max_period_days"] == 20.0
    monkeypatch.setattr(type(window._bridge), "is_running", property(lambda self: True))
    window.start_sample()
    assert len(calls) == 1
    assert "already running" in window.lbl_status.text()


def test_sample_keyboard_shortcut(qtbot, tmp_path: Path, monkeypatch, sample_model: Path) -> None:
    window = TransitEyeMainWindow()
    qtbot.addWidget(window)
    sample = tmp_path / "sample.fits"
    sample.write_bytes(b"input")
    monkeypatch.setattr("transiteye_desktop.main_window.resolve_sample", lambda path: sample)
    calls = []
    monkeypatch.setattr(window._bridge, "analyze_fits", lambda *args, **kwargs: calls.append(args))
    window.show()
    window.activateWindow()
    qtbot.waitUntil(window.isActiveWindow)
    qtbot.keyClick(window, Qt.Key.Key_D, Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier)
    assert calls == [(sample,)]


def test_default_sample_rejects_changed_data(tmp_path: Path, monkeypatch) -> None:
    from transiteye_desktop import sample

    path = tmp_path / ".transiteye-workspace" / sample.SAMPLE_RELATIVE_PATH
    path.parent.mkdir(parents=True)
    path.write_bytes(b"unexpected content")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="checksum"):
        sample.resolve_sample()


def test_historical_model_requires_explicit_opt_in_for_single_and_batch(qtbot, tmp_path: Path, monkeypatch, sample_model: Path) -> None:
    from transiteye_desktop.main_window import HISTORICAL_CALIBRATOR_SHA256

    window = TransitEyeMainWindow()
    qtbot.addWidget(window)
    model = sample_model
    sample = tmp_path / "sample.fits"
    sample.touch()
    window._current_fits = sample
    single, batches = [], []
    monkeypatch.setattr(window._bridge, "analyze_fits", lambda *args, **kwargs: single.append(kwargs))
    monkeypatch.setattr(window._bridge, "run_batch", lambda *args, **kwargs: batches.append(kwargs))
    window._on_analyze_clicked()
    assert "model_dir" not in single[-1]
    window.check_historical_model.setChecked(True)
    window._on_analyze_clicked()
    window._on_start_batch("input", "cpu", 1)
    for options in (single[-1], batches[-1]):
        assert options["model_dir"] == model
        assert options["model_calibrator_sha256"] == HISTORICAL_CALIBRATOR_SHA256


def test_missing_selected_historical_model_reports_error(qtbot, tmp_path: Path, monkeypatch) -> None:
    window = TransitEyeMainWindow()
    qtbot.addWidget(window)
    sample = tmp_path / "sample.fits"
    sample.touch()
    window._current_fits = sample
    monkeypatch.setattr("transiteye_desktop.main_window.HISTORICAL_MODEL", tmp_path / "missing")
    warnings, calls = [], []
    monkeypatch.setattr("transiteye_desktop.main_window.QMessageBox.warning", lambda *args: warnings.append(args))
    monkeypatch.setattr(window._bridge, "analyze_fits", lambda *args, **kwargs: calls.append(args))
    window.check_historical_model.setChecked(True)
    window._on_analyze_clicked()
    assert not calls
    assert "missing" in warnings[0][2]


def test_demo_rejects_changed_model_before_clearing_result(qtbot, tmp_path: Path, monkeypatch, sample_model: Path) -> None:
    window = TransitEyeMainWindow()
    qtbot.addWidget(window)
    window.display_bundle_from_path(_create_canonical_bundle(tmp_path / "old"))
    old_result = window._current_bundle
    sample = tmp_path / "sample.fits"
    sample.touch()
    (sample_model / "estimator.joblib").write_bytes(b"changed")
    warnings = []
    monkeypatch.setattr("transiteye_desktop.main_window.QMessageBox.warning", lambda *args: warnings.append(args))
    window.start_sample(sample)
    assert window._current_bundle is old_result
    assert not window.check_historical_model.isChecked()
    assert "Checksum mismatch" in warnings[0][2]


def test_desktop_passes_search_range_to_backend(qtbot, tmp_path: Path, monkeypatch) -> None:
    fits_path = tmp_path / "input.fits"
    fits_path.touch()
    window = TransitEyeMainWindow()
    qtbot.addWidget(window)
    window._current_fits = fits_path
    window.spin_min_period.setValue(1.0)
    window.spin_max_period.setValue(20.0)
    calls = []
    monkeypatch.setattr(window._bridge, "analyze_fits", lambda *args, **kwargs: calls.append((args, kwargs)))
    window.btn_analyze.click()
    assert calls[0][1]["min_period_days"] == 1.0
    assert calls[0][1]["max_period_days"] == 20.0


def test_desktop_batch_uses_inspector_period_range(qtbot, monkeypatch) -> None:
    window = TransitEyeMainWindow()
    qtbot.addWidget(window)
    window.spin_min_period.setValue(1.0)
    window.spin_max_period.setValue(20.0)
    calls = []
    monkeypatch.setattr(window._bridge, "run_batch", lambda *args, **kwargs: calls.append(kwargs))
    window._on_start_batch("/test", "cpu", 2)
    assert calls[0]["min_period_days"] == 1.0
    assert calls[0]["max_period_days"] == 20.0


def test_batch_start_is_blocked_while_another_analysis_runs(qtbot, monkeypatch) -> None:
    window = TransitEyeMainWindow()
    qtbot.addWidget(window)
    monkeypatch.setattr(BackendProcessBridge, "is_running", property(lambda self: True))
    calls = []
    monkeypatch.setattr(window._bridge, "run_batch", lambda *args, **kwargs: calls.append(kwargs))
    window._on_process_state_changed(True)
    assert not window.batch_panel.btn_start.isEnabled()
    window._on_start_batch("/test", "cpu", 1)
    assert not calls
    assert "already running" in window.lbl_status.text()
    window._on_analyze_clicked()
    assert "already running" in window.lbl_status.text()
    window._on_process_state_changed(False)
    assert window.batch_panel.btn_start.isEnabled()


def test_desktop_batch_keeps_failure_visible_on_retry(qtbot, tmp_path: Path, monkeypatch) -> None:
    from transiteye.execution.store import JobStatus, JobStore

    monkeypatch.chdir(tmp_path)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "bad_lc.fits").write_bytes(b"not a FITS file")
    window = TransitEyeMainWindow()
    qtbot.addWidget(window)
    panel = window.batch_panel
    panel.txt_source.setText(str(inputs))
    panel.combo_device.setCurrentText("cpu")
    panel.spin_workers.setValue(1)
    with qtbot.waitSignal(window._bridge.analysis_completed, timeout=30000) as finished:
        panel.btn_start.click()
    assert finished.args[1]["failed_targets"] == 1
    assert panel.table.rowCount() == 1
    assert panel.table.item(0, 1).text() == "failed"
    assert panel.table.item(0, 4).text()

    job_id = panel.txt_job_id.text()
    store = JobStore(tmp_path / ".transiteye-workspace" / "state.db")
    window._on_batch_control("retry", job_id)
    qtbot.waitUntil(lambda: store.get_job(job_id).status is JobStatus.PAUSED, timeout=30000)
    with qtbot.waitSignal(window._bridge.analysis_completed, timeout=30000) as retried:
        window._on_resume_batch(job_id, 1)
    assert retried.args[1]["failed_targets"] == 1
    assert store.list_tasks(job_id)[0].attempt_count == 2
    assert panel.table.item(0, 4).text()


def test_bridge_single_and_batch_send_same_search_arguments(qtbot) -> None:
    from PyQt6.QtCore import QProcess

    class FakeProcess:
        def __init__(self) -> None:
            self.arguments: list[str] = []

        def state(self):
            return QProcess.ProcessState.NotRunning

        def start(self, program: str, arguments: list[str]) -> None:
            self.arguments = arguments

    bridge = BackendProcessBridge()
    process = FakeProcess()
    bridge._process = process
    bridge.analyze_fits("target.fits", min_period_days=1.0, max_period_days=20.0,
                        model_dir="model", model_calibrator_sha256="abc")
    single = process.arguments
    bridge.run_batch("target.fits", min_period_days=1.0, max_period_days=20.0,
                     model_dir="model", model_calibrator_sha256="abc")
    batch = process.arguments
    for option in ("--snr-threshold", "--max-candidates", "--bls-min-period-days", "--bls-max-period-days",
                   "--model-dir", "--model-calibrator-sha256"):
        assert single[single.index(option) + 1] == batch[batch.index(option) + 1]


def test_missing_session_bundle_is_reported(qtbot, tmp_path: Path) -> None:
    session = tmp_path / "missing.teye.json"
    session.write_text(json.dumps({"bundle_path": str(tmp_path / "missing_bundle")}))
    window = TransitEyeMainWindow()
    qtbot.addWidget(window)
    with pytest.raises(FileNotFoundError, match="result bundle"):
        window.load_session(session)


def test_invalid_session_bundle_is_not_reported_as_loaded(qtbot, tmp_path: Path) -> None:
    broken_bundle = tmp_path / "broken"
    broken_bundle.mkdir()
    session = tmp_path / "broken.teye.json"
    session.write_text(json.dumps({"bundle_path": str(broken_bundle)}))
    window = TransitEyeMainWindow()
    qtbot.addWidget(window)
    with pytest.raises(ValueError, match="Could not load the saved result bundle"):
        window.load_session(session)
    assert window._current_bundle is None


def test_session_without_result_does_not_leave_old_result_visible(qtbot, tmp_path: Path) -> None:
    session = tmp_path / "empty.teye.json"
    session.write_text(json.dumps({"bundle_path": None}))
    window = TransitEyeMainWindow()
    qtbot.addWidget(window)
    window.check_historical_model.setChecked(True)
    window.display_bundle_from_path(_create_canonical_bundle(tmp_path / "bundle"))
    window.load_session(session)
    assert not window.check_historical_model.isChecked()
    assert window._current_bundle is None
    assert window.candidate_table.table.rowCount() == 0
    assert "no saved analysis result" in window.lbl_search_summary.text()


def test_session_buttons_save_and_reopen_result(qtbot, tmp_path: Path, monkeypatch) -> None:
    from PyQt6.QtWidgets import QFileDialog, QMessageBox

    bundle = _create_canonical_bundle(tmp_path / "bundle")
    session = tmp_path / "session.teye.json"
    window = TransitEyeMainWindow()
    qtbot.addWidget(window)
    window.display_bundle_from_path(bundle)
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args, **kwargs: (str(session), ""))
    monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: QMessageBox.StandardButton.Ok)
    window.btn_save_session.click()
    assert session.exists()
    empty = tmp_path / "empty.teye.json"
    empty.write_text(json.dumps({"bundle_path": None}))
    window.load_session(empty)
    assert window._current_bundle is None
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args, **kwargs: (str(session), ""))
    window.btn_open_session.click()
    assert window._current_bundle is not None
    assert window._current_bundle.target_id == "99988877"


def test_vetting_export_runs_asynchronously(qtbot, tmp_path: Path) -> None:
    bundle = _create_canonical_bundle(tmp_path / "bundle")
    bridge = BackendProcessBridge()
    output = tmp_path / "sheet.png"
    with qtbot.waitSignal(bridge.export_completed, timeout=30000) as signal:
        bridge.start_vetting_sheet_export(bundle, output)
    assert signal.args == [str(output)]
    assert output.stat().st_size > 10000


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_end_to_end_backend_bundle_loads_in_desktop_ui(qtbot, tmp_path: Path) -> None:
    """Verify a real bundle generated by analyze_light_curve loads cleanly into Desktop UI."""
    t = np.linspace(1.0, 14.0, 2000)
    model = evaluate_trapezoid(
        t,
        epoch_days=2.0,
        period_days=3.0,
        duration_days=2.5 / 24.0,
        depth=0.008,
        ingress_fraction=0.2,
    )
    f = model + np.random.default_rng(77).normal(0, 0.0003, size=len(t))

    fits_path = tmp_path / "e2e_target_lc.fits"
    cols = [
        fits.Column(name="TIME", format="D", array=t),
        fits.Column(name="PDCSAP_FLUX", format="D", array=f),
        fits.Column(name="PDCSAP_FLUX_ERR", format="D", array=np.full_like(f, 0.0005)),
        fits.Column(name="QUALITY", format="K", array=np.zeros(len(t), dtype=np.int64)),
        fits.Column(name="CADENCENO", format="K", array=np.arange(len(t), dtype=np.int64)),
        fits.Column(name="MOM_CENTR1", format="D", array=np.zeros(len(t))),
        fits.Column(name="MOM_CENTR2", format="D", array=np.zeros(len(t))),
    ]
    table = fits.BinTableHDU.from_columns(cols)
    table.header["TIMESYS"] = "TDB"
    table.header["BJDREFI"] = 2457000
    table.header["CROWDSAP"] = 0.92
    primary = fits.PrimaryHDU()
    primary.header["TICID"] = 55566677
    primary.header["SECTOR"] = 1
    fits.HDUList([primary, table]).writeto(fits_path)

    lc = read_tess_light_curve(fits_path)
    res, bundle_dir = analyze_light_curve(
        lc,
        max_candidates=1,
        snr_threshold=6.0,
        device="cpu",
        output_dir=tmp_path / "real_bundle",
    )
    assert res.status.value == "complete"

    window = TransitEyeMainWindow()
    qtbot.addWidget(window)
    window.display_bundle_from_path(bundle_dir)

    assert window._current_bundle is not None
    assert window._current_bundle.target_id == "55566677"
    assert window.candidate_table.table.rowCount() == 1
    assert window._current_bundle.candidates[0].trapezoid_model_curve is not None
    assert window._current_bundle.candidates[0].search_significance is None
    assert window._current_bundle.candidates[0].calibrated_confidence is None
    assert window.candidate_table.table.item(0, 9).text() == "--"
    assert window.candidate_table.table.item(0, 11).text() == "--"
    assert "unverified assumption" in window.vetting_card.lbl_exposure.text()
    assert "68%" in window.vetting_card.lbl_period_ci.text()


def test_backend_bridge_multiline_json_parsing(qtbot: pytest.QtBot) -> None:
    from PyQt6.QtCore import QProcess

    bridge = BackendProcessBridge()
    completed_bundles: list[tuple[str, dict]] = []
    bridge.analysis_completed.connect(lambda p, s: completed_bundles.append((p, s)))

    # Simulate multi-line indented JSON summary
    multiline_summary = json.dumps(
        {"bundle_path": "/tmp/test_bundle", "status": "complete", "candidate_count": 1},
        indent=2,
    )
    bridge._full_stdout = multiline_summary
    bridge._stdout_buffer = multiline_summary

    bridge._on_process_finished(0, QProcess.ExitStatus.NormalExit)

    assert len(completed_bundles) == 1
    assert completed_bundles[0][0] == "/tmp/test_bundle"
    assert completed_bundles[0][1]["status"] == "complete"


def test_backend_bridge_exit_code_zero_error_message(qtbot: pytest.QtBot) -> None:
    from PyQt6.QtCore import QProcess

    bridge = BackendProcessBridge()
    failed_msgs: list[str] = []
    bridge.analysis_failed.connect(lambda msg: failed_msgs.append(msg))

    bridge._on_process_finished(0, QProcess.ExitStatus.NormalExit)

    assert len(failed_msgs) == 1
    assert "non-zero code 0" not in failed_msgs[0]
    assert "code 0" in failed_msgs[0]

