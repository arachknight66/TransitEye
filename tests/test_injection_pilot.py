"""Truth matching and accounting for the synthetic-only recovery pilot."""

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
from scipy.integrate import trapezoid

from transiteye.characterization.fitting import evaluate_trapezoid
from transiteye.domain.models import AnalysisSettings
from transiteye.evaluation.injection_pilot import (
    _trial,
    injection_flux,
    matches_truth,
    run,
    summarize,
)


def test_independent_injection_matches_high_resolution_integral() -> None:
    times = np.array([-0.05, -0.04, 0.0, 0.04, 0.05])
    exposure = 10 / 1440
    offsets = np.linspace(-exposure / 2, exposure / 2, 10001)
    flux = evaluate_trapezoid(
        (times[:, None] + offsets).ravel(), 0.0, 2.0, 0.1, 0.001, 0.2, supersample=1,
    ).reshape(len(times), -1)
    expected = 1 - trapezoid(flux, x=offsets, axis=1) / exposure
    actual = injection_flux(times, 2.0, 0.0, 2.4, 1000.0, 0.2, exposure, 101)
    np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=0)


def test_truth_match_rejects_harmonics_and_checks_wrapped_epoch() -> None:
    assert matches_truth(2.01, 4.01, 2.0, 0.0, 0.02, 0.5)
    assert not matches_truth(1.0, 0.0, 2.0, 0.0, 0.02, 0.5)
    assert not matches_truth(2.0, 0.1, 2.0, 0.0, 0.02, 0.5)


def test_summary_counts_failures_and_unavailable_intervals() -> None:
    def row(kind: str, matched: bool, detected: bool, status: str) -> dict:
        value = {"kind": kind, "matched": matched, "detected": detected, "status": status,
                 "truth_period_days": 2.0 if kind == "injection" else None,
                 "truth_depth_ppm": 1000.0 if kind == "injection" else None,
                 "harmonic": False, "fit_failed": False, "physical_fit_failed": False}
        for name in ("period_days", "depth_ppm", "duration_hours"):
            value[f"error_{name}"] = 0.1 if matched else None
            value[f"covered_{name}_68"] = True if matched else None
            value[f"covered_{name}_95"] = None
        return value
    result = summarize([row("injection", True, True, "complete"),
                        row("injection", False, False, "error"),
                        row("null", False, True, "complete")])
    assert result["injected_trials"] == 2
    assert result["recovered"] == 1
    assert result["null_alerts"] == result["null_trials"] == 1
    assert result["trial_errors"] == 1
    assert result["parameters"]["depth_ppm"]["95"] == {
        "covered": 0, "available": 0, "matched": 1, "all_injections": 2,
    }


def test_one_synthetic_trial_uses_backend_and_records_input(tmp_path: Path) -> None:
    config = {"seed": 1, "cadence_minutes": 10, "exposure_minutes": 10, "baseline_days": 8,
              "noise_ppm": 500, "duration_hours": 2.5, "ingress_fraction": 0.2,
              "injection_integration_points": 101, "match_period_fraction": 0.02,
              "match_epoch_hours": 0.5}
    settings = AnalysisSettings(max_candidates=1, snr_threshold=6, screen_snr_threshold=5,
                                device="cpu", bootstrap_replicates=10,
                                bls_min_period_days=0.5, bls_max_period_days=3.0,
                                trapezoid_exposure_days=10 / 1440)
    row = _trial(config, "pilot-test", 2.0, 5000.0, settings, tmp_path / "bundle")
    assert row["status"] == "complete", row["error"]
    assert row["detected"] and row["matched"]
    assert len(row["input_sha256"]) == 64
    assert row["truth_depth_ppm"] == 5000.0


def test_invalid_injection_grid_is_rejected() -> None:
    with pytest.raises(ValueError):
        injection_flux(np.array([0.0]), 2.0, 0.0, 2.4, 1000, 0.2, 0.01, 100)


def test_resume_counts_each_checkpoint_once(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "pilot.toml"
    config.write_text('''schema_version = "1"
seed = 1
exposure_minutes = 10.0
periods_days = [2.0]
depths_ppm = [500.0]
replicates = 1
null_trials = 1
[analysis]
max_candidates = 1
snr_threshold = 6.0
screen_snr_threshold = 5.0
device = "cpu"
''')
    calls = []

    def fail_trial(config, trial_id, period, depth, settings, output_dir):
        calls.append(trial_id)
        row = {"trial_id": trial_id, "seed": 2**64 - 1,
               "kind": "null" if period is None else "injection",
               "truth_period_days": period, "truth_depth_ppm": depth, "status": "error",
               "error": "deliberate", "detected": False, "fit_failed": False,
               "physical_fit_failed": False,
               "matched": False, "harmonic": False}
        for name in ("period_days", "depth_ppm", "duration_hours"):
            row[f"error_{name}"] = None
            for level in ("68", "95"):
                row[f"covered_{name}_{level}"] = None
        return row

    monkeypatch.setattr("transiteye.evaluation.injection_pilot._trial", fail_trial)
    first = run(config, tmp_path / "output")
    second = run(config, tmp_path / "output")
    assert first == second
    assert len(calls) == 2
    assert second["injected_trials"] == second["null_trials"] == 1
    assert second["trial_errors"] == 2
    assert pq.read_table(tmp_path / "output" / "trials.parquet").column("seed")[0].as_py() == 2**64 - 1
