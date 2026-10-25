"""Headless GUI and component tests for the desktop package."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from astropy.io import fits
from transiteye_desktop.bundle_loader import LoadedBundle, load_result_bundle
from transiteye_desktop.downsample import lttb_downsample, min_max_downsample
from transiteye_desktop.main_window import TransitEyeMainWindow

from transiteye.characterization.fitting import evaluate_trapezoid
from transiteye.execution.service import analyze_light_curve
from transiteye.io.fits import read_tess_light_curve


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
        "flags": [["FLAG_OK"]],
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

    # Verify session save and load (.teye.json)
    session_file = tmp_path / "test_session.teye.json"
    saved_path = window.save_session(session_file)
    assert saved_path.is_file()

    window2 = TransitEyeMainWindow()
    qtbot.addWidget(window2)
    window2.load_session(saved_path)
    assert window2._current_bundle is not None
    assert window2._current_bundle.target_id == "99988877"

    # Verify PNG/PDF vetting sheet export via subprocess bridge
    out_png = tmp_path / "exported_vetting.png"
    assert window.export_vetting_sheet(out_png) is True
    assert out_png.is_file()
    assert out_png.stat().st_size > 10000


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
    assert "68%" in window.vetting_card.lbl_period_ci.text()
