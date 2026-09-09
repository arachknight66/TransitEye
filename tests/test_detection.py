from __future__ import annotations

import numpy as np
import pandas as pd

from transiteye.config import BlsSettings
from transiteye.detection.bls import run_bls
from transiteye.detection.matching import match_candidates
from transiteye.detection.peaks import extract_peaks


def _settings() -> BlsSettings:
    return BlsSettings(
        min_period_days=0.5,
        max_period_days=8,
        min_transits=2,
        durations_days=(0.05, 0.1, 0.15),
        frequency_factor=5,
        top_k=4,
        local_peak_fraction=0.01,
        harmonic_tolerance=0.03,
        period_match_tolerance=0.03,
        phase_match_tolerance=0.2,
    )


def _curve(period: float = 2.5) -> pd.DataFrame:
    time = np.arange(0, 20, 0.02)
    time = time[~((time > 8) & (time < 9))]
    flux = np.ones_like(time)
    phase = ((time - 0.3 + 0.5 * period) % period) - 0.5 * period
    flux[np.abs(phase) < 0.05] -= 0.01
    return pd.DataFrame({"time": time, "detrended_flux": flux, "valid": True})


def test_blind_synthetic_recovery_is_deterministic() -> None:
    result = run_bls(
        _curve(),
        settings=_settings(),
        observation_group_id="og-synthetic",
        preprocessing_hash="a" * 20,
    )
    peaks = extract_peaks(result, _settings())
    assert any(abs(float(p) / 2.5 - 1) < 0.03 for p in peaks.period)
    assert (
        result.config_hash
        == run_bls(
            _curve(),
            settings=_settings(),
            observation_group_id="og-synthetic",
            preprocessing_hash="a" * 20,
        ).config_hash
    )


def test_post_detection_matching_recognizes_harmonic() -> None:
    candidates = pd.DataFrame([{"candidate_id": "cand-x", "period": 1.25, "epoch": 0.3}])
    events = pd.DataFrame([{"toi_id": "1.01", "period_days": 2.5, "transit_epoch_bjd": 0.3}])
    match = match_candidates(candidates, events, _settings())
    assert match.loc[0, "match_type"] == "half_period" and bool(match.loc[0, "matched"])
