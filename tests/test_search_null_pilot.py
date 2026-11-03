"""Experimental IID null testing must repeat the complete primary BLS grid."""

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from transiteye.detection.bls import run_bls
from transiteye.evaluation.search_null_pilot import iid_primary_bls_permutation, run, summarize


def _search() -> dict:
    return {"min_period": 0.5, "max_period": 1.5, "min_duration_hours": 0.5,
            "max_duration_hours": 2.0, "num_durations": 3, "oversample": 5.0, "device": "cpu"}


def test_permutation_repeats_full_bls_grid_and_uses_conservative_count() -> None:
    times = np.arange(150, dtype=np.float64) / 48
    flux = 1 + np.random.default_rng(43).normal(0, 0.001, len(times))
    result = iid_primary_bls_permutation(times, flux, _search(), 9, 7, assume_iid=True)
    observed = float(np.max(run_bls(times, flux, **_search()).powers))
    rng = np.random.default_rng(7)
    powers = [float(np.max(run_bls(times, rng.permutation(flux), **_search()).powers)) for _ in range(9)]
    tolerance = 32 * np.finfo(np.float64).eps * max(1.0, abs(observed))
    expected = sum(value >= observed - tolerance for value in powers)
    assert result.exceedances == expected
    assert result.p_value == (expected + 1) / 10
    assert result == iid_primary_bls_permutation(times, flux, _search(), 9, 7, assume_iid=True)


def test_permutation_requires_explicit_iid_and_counts_ties() -> None:
    times = np.arange(150, dtype=np.float64) / 48
    with pytest.raises(ValueError, match="explicitly assumed"):
        iid_primary_bls_permutation(times, np.ones_like(times), _search(), 4, 2)
    result = iid_primary_bls_permutation(times, np.ones_like(times), _search(), 4, 2, assume_iid=True)
    assert result.exceedances == 4
    assert result.p_value == 1.0
    bad = times.copy()
    bad[20] = np.nan
    with pytest.raises(ValueError, match="finite, ordered"):
        iid_primary_bls_permutation(bad, np.ones_like(times), _search(), 4, 2, assume_iid=True)


def test_summary_never_passes_with_missing_trials() -> None:
    config = {"alpha": 0.05, "white_trials": 2, "max_white_alerts": 1, "dkw_confidence": 0.95}
    rows = [{"kind": "white", "status": "complete", "p_value": 0.5},
            {"kind": "white", "status": "error", "p_value": None},
            {"kind": "correlated", "status": "complete", "p_value": 0.01}]
    result = summarize(rows, config)
    assert result["white"]["errors"] == 1
    assert not result["white"]["pilot_passed"]
    assert result["correlated"]["alerts_at_alpha"] == 1


def test_null_pilot_reuses_checkpoints_and_preserves_unsigned_seeds(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "null.toml"
    config.write_text('''schema_version = "1"
seed = 3
white_trials = 1
correlated_trials = 1
permutations = 9
alpha = 0.05
max_white_alerts = 1
dkw_confidence = 0.95
[search]
device = "cpu"
''')
    calls = []

    def fake_trial(config, kind, index):
        calls.append((kind, index))
        return {"kind": kind, "trial_index": index, "status": "complete", "error": None,
                "noise_seed": 2**64 - 1, "permutation_seed": 2**64 - 1,
                "input_sha256": "a" * 64, "observed_max_power": 0.1,
                "exceedances": 5, "p_value": 0.6}

    monkeypatch.setattr("transiteye.evaluation.search_null_pilot._trial", fake_trial)
    first = run(config, tmp_path / "output")
    second = run(config, tmp_path / "output")
    assert first == second
    assert len(calls) == 2
    table = pq.read_table(tmp_path / "output" / "trials.parquet")
    assert table.column("noise_seed")[0].as_py() == 2**64 - 1
