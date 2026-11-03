"""Tests for result bundle generation, persistence, and diagnostic figure vetting sheets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from transiteye.reporting.figures import generate_vetting_sheet


def _create_mock_bundle(bundle_dir: Path) -> Path:
    bundle_dir.mkdir(parents=True, exist_ok=True)

    res_data = {
        "analysis_id": "test_reporting_01",
        "target_id": "88877766",
        "status": "completed",
        "created_at": "2026-10-24T12:00:00Z",
    }
    (bundle_dir / "result.json").write_text(json.dumps(res_data), encoding="utf-8")

    t = np.linspace(1.0, 10.0, 1000)
    raw = np.ones_like(t)
    detrended = np.ones_like(t)
    detrended[200:230] = 0.99
    trend = np.ones_like(t)
    err = np.full_like(t, 0.001)
    mask = np.ones(len(t), dtype=bool)
    periods = np.linspace(0.5, 5.0, 50)
    powers = np.random.default_rng(42).uniform(0.0, 10.0, 50)
    powers[20] = 40.0
    tpf = np.ones((9, 9))
    diff = np.zeros((9, 9))

    np.savez_compressed(
        bundle_dir / "arrays.npz",
        time=t,
        raw_flux=raw,
        flux=detrended,
        trend=trend,
        flux_err=err,
        quality_mask=mask,
        bls_periods=periods,
        bls_powers=powers,
        tpf_stamp=tpf,
        tpf_diff=diff,
    )

    cand_data = {
        "candidate_id": ["cand_01"],
        "period_days": [2.5],
        "epoch_days": [2.2],
        "duration_hours": [2.0],
        "depth_ppm": [10000.0],
        "detection_snr": [18.4],
        "search_significance": [7.2],
        "predicted_class": ["transit_like"],
        "calibrated_confidence": [0.94],
        "abstained": [False],
        "source_assessment": ["target_consistent"],
        "flags": [["FLAG_OK"]],
        "centroid_shift_sigma": [0.6],
        "crowdsap_value": [0.96],
        "sap_to_pdcsap_depth_ratio": [1.01],
    }
    schema = pa.schema([
        ("candidate_id", pa.string()),
        ("period_days", pa.float64()),
        ("epoch_days", pa.float64()),
        ("duration_hours", pa.float64()),
        ("depth_ppm", pa.float64()),
        ("detection_snr", pa.float64()),
        ("search_significance", pa.float64()),
        ("predicted_class", pa.string()),
        ("calibrated_confidence", pa.float64()),
        ("abstained", pa.bool_()),
        ("source_assessment", pa.string()),
        ("flags", pa.list_(pa.string())),
        ("centroid_shift_sigma", pa.float64()),
        ("crowdsap_value", pa.float64()),
        ("sap_to_pdcsap_depth_ratio", pa.float64()),
    ])
    tbl = pa.Table.from_pydict(cand_data, schema=schema)
    pq.write_table(tbl, bundle_dir / "candidates.parquet")

    return bundle_dir


def test_generate_vetting_sheet_png(tmp_path: Path) -> None:
    """Verify diagnostic vetting sheet is rendered to 300dpi PNG without GUI display."""
    bundle_dir = _create_mock_bundle(tmp_path / "mock_bundle")
    out_png = tmp_path / "vetting_sheet.png"

    res = generate_vetting_sheet(bundle_dir, out_png, candidate_index=0)
    assert res.exists()
    assert res.stat().st_size > 10000  # Non-trivial image file


def test_vetting_sheet_displays_missing_source_measurements(tmp_path: Path, monkeypatch) -> None:
    from matplotlib.figure import Figure

    bundle = _create_mock_bundle(tmp_path / "bundle")
    parquet = bundle / "candidates.parquet"
    table = pq.read_table(parquet)
    for name in ("centroid_shift_sigma", "crowdsap_value"):
        index = table.schema.get_field_index(name)
        table = table.set_column(index, name, pa.array([float("nan")]))
    pq.write_table(table, parquet)
    captured = []
    original = Figure.savefig

    def capture(figure, *args, **kwargs):
        captured.extend(text.get_text() for axis in figure.axes for text in axis.texts)
        return original(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    generate_vetting_sheet(bundle, tmp_path / "sheet.png")
    card = next(text for text in captured if "CANDIDATE CHARACTERIZATION" in text)
    assert "Centroid Shift:      unavailable" in card
    assert "CROWDSAP:            unavailable" in card
    assert "NO (High Confidence)" not in card
