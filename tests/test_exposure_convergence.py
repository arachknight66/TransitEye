"""Checks for the development-only exposure quadrature experiment."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq

from transiteye.evaluation.exposure_convergence import (
    ExposureCase,
    _parameters,
    reference_flux,
    run,
    study,
)


def _config() -> dict:
    return {
        "schema_version": "1", "period_days": 3.0, "semi_major_axis_stellar_radii": 10.0,
        "radius_ratios": [0.1], "impact_parameters": [0.0], "exposure_minutes": [30.0],
        "limb_darkening": [0.3, 0.22], "sample_factors": [5, 101], "reference_points": 1001,
        "reference_check_points": 501, "time_start_days": -0.2, "time_end_days": 0.2,
        "time_points": 101,
    }


def test_reference_integrates_constant_out_of_transit_flux() -> None:
    params = _parameters(_config(), ExposureCase(0.1, 0.0, 30))
    flux = reference_flux(params, np.array([0.4, 0.5]), 30 / 1440, 101)
    np.testing.assert_allclose(flux, 1.0, atol=1e-12)


def test_exposure_study_resolves_expected_quadrature_error_and_preserves_failures() -> None:
    config = _config()
    config["radius_ratios"] = [0.1, -1.0]
    rows = study(config)
    assert len(rows) == 4
    assert rows[0]["max_error_ppm"] > rows[1]["max_error_ppm"]
    assert rows[0]["reference_difference_ppm"] < 0.1
    assert all(row["error"] and row["max_error_ppm"] is None for row in rows[2:])


def test_simpson_study_improves_long_exposure_integration() -> None:
    config = _config()
    config["sample_factors"] = [51]
    baseline = study(config)[0]
    config["integration_method"] = "simpson"
    improved = study(config)[0]
    assert improved["error"] is None
    assert improved["max_error_ppm"] < 1.0
    assert improved["max_error_ppm"] < baseline["max_error_ppm"]


def test_frozen_study_persists_provenance_and_all_rows(tmp_path: Path) -> None:
    config = Path("configs/exposure_convergence.v1.toml")
    summary = run(config, tmp_path)
    rows = pq.read_table(tmp_path / "cases.parquet").to_pylist()
    assert len(rows) == summary["n_rows"] == 60
    assert summary["n_failures"] == 0
    assert summary["n_unusable"] == sum(not row["usable"] for row in rows)
    assert summary["n_outside_tolerance"] == sum(not row["within_tolerance"] for row in rows)
    assert all(row["native_diagnostics"] for row in rows if not row["usable"])
    assert len(summary["config_sha256"]) == len(summary["backend_sha256"]) == 64
    assert (tmp_path / "summary.json").is_file()


def test_native_warning_marks_finite_results_unusable(tmp_path: Path, monkeypatch) -> None:
    row = {"radius_ratio": 0.1, "impact_parameter": 0.9, "exposure_minutes": 2,
           "sample_factor": 5, "reference_difference_ppm": 0.001, "depth_ppm": 8000,
           "max_error_ppm": 0.01, "error": None}
    def fake_worker(args, **kwargs):
        case = json.loads(args[-1])
        rows = [{**row, "radius_ratio": case["radius_ratios"][0],
                 "impact_parameter": case["impact_parameters"][0],
                 "exposure_minutes": case["exposure_minutes"][0], "sample_factor": factor}
                for factor in case["sample_factors"]]
        output = "TRANSITEYE_ROWS=" + json.dumps(rows) + "\nConvergence failure in ellpic_bulirsch\n"
        return SimpleNamespace(returncode=0, stdout=output, stderr="")
    monkeypatch.setattr(
        "transiteye.evaluation.exposure_convergence.subprocess.run",
        fake_worker,
    )
    summary = run(Path("configs/exposure_convergence.v1.toml"), tmp_path)
    rows = pq.read_table(tmp_path / "cases.parquet").to_pylist()
    assert summary["n_unusable"] == len(rows)
    assert all(not row["usable"] and not row["within_tolerance"] for row in rows)


def test_incomplete_worker_output_is_counted_as_failure(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "transiteye.evaluation.exposure_convergence.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="TRANSITEYE_ROWS=[]\n", stderr=""),
    )
    summary = run(Path("configs/exposure_convergence.v1.toml"), tmp_path)
    rows = pq.read_table(tmp_path / "cases.parquet").to_pylist()
    assert summary["n_rows"] == summary["n_failures"] == len(rows) == 60
    assert all("invalid or incomplete output" in row["error"] for row in rows)
