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


def _curve(period: float = 2.5, *, epoch: float = 0.3, depth: float = 0.01) -> pd.DataFrame:
    time = np.arange(0, 20, 0.02)
    time = time[~((time > 8) & (time < 9))]
    flux = np.ones_like(time)
    phase = ((time - epoch + 0.5 * period) % period) - 0.5 * period
    flux[np.abs(phase) < 0.05] -= depth
    return pd.DataFrame({"time": time, "detrended_flux": flux, "valid": True})


def _outcome(
    period: float, *, gap: bool = False, irregular: bool = False, depth: float = 0.01
) -> str:
    frame = _curve(period, depth=depth)
    if gap:
        frame = frame.loc[~frame.time.between(2.4, 2.6)].reset_index(drop=True)
    if irregular:
        frame = frame.iloc[np.arange(len(frame)) % 7 != 0].reset_index(drop=True)
    result = run_bls(
        frame, settings=_settings(), observation_group_id="og-matrix", preprocessing_hash="b" * 20
    )
    peaks = extract_peaks(result, _settings())
    ratios = [float(value) / period for value in peaks.period]
    if any(abs(ratio - 1) < 0.03 for ratio in ratios):
        return "fundamental"
    if any(abs(ratio - 0.5) < 0.03 or abs(ratio - 2) < 0.03 for ratio in ratios):
        return "harmonic"
    return "not_recovered"


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


def test_synthetic_recovery_matrix_is_deterministic() -> None:
    outcomes = [_outcome(2.5), _outcome(3.1, gap=True), _outcome(2.7, irregular=True)]
    assert outcomes == [_outcome(2.5), _outcome(3.1, gap=True), _outcome(2.7, irregular=True)]
    assert outcomes[0] == "fundamental"


def test_no_transit_is_not_claimed_as_injected_recovery() -> None:
    frame = _curve(2.5)
    frame["detrended_flux"] = 1.0
    result = run_bls(
        frame, settings=_settings(), observation_group_id="og-null", preprocessing_hash="c" * 20
    )
    candidates = extract_peaks(result, _settings())
    events = pd.DataFrame([{"toi_id": "synthetic", "period_days": 2.5, "transit_epoch_bjd": 0.3}])
    assert not match_candidates(candidates, events, _settings()).matched.any()


def test_weak_transit_has_deterministic_graceful_outcome() -> None:
    frame = _curve(depth=0.001)
    first = _outcome(2.5, depth=0.001)
    result = run_bls(
        frame, settings=_settings(), observation_group_id="og-weak", preprocessing_hash="d" * 20
    )
    peaks = extract_peaks(result, _settings())
    assert np.isfinite(result.periodogram.power).all()
    assert len(peaks) > 0
    assert first == _outcome(2.5, depth=0.001)
    assert first in {"fundamental", "harmonic", "not_recovered"}


def test_long_period_baseline_limit_is_deterministic() -> None:
    settings = _settings().model_copy(update={"max_period_days": 20.0})
    frame = _curve(period=9.5)
    result = run_bls(
        frame, settings=settings, observation_group_id="og-baseline", preprocessing_hash="e" * 20
    )
    effective_maximum = min(
        settings.max_period_days, (frame.time.max() - frame.time.min()) / settings.min_transits
    )
    assert effective_maximum == 9.99
    assert result.periodogram.period.max() <= effective_maximum
    assert len(extract_peaks(result, settings)) > 0


def test_noninteger_cadence_epoch_alignment_recovers_period() -> None:
    frame = _curve(period=2.7, epoch=0.337)
    result = run_bls(
        frame,
        settings=_settings(),
        observation_group_id="og-noninteger",
        preprocessing_hash="f" * 20,
    )
    peaks = extract_peaks(result, _settings())
    assert any(abs(float(period) / 2.7 - 1) < 0.03 for period in peaks.period)


def test_double_period_matching_is_not_fundamental() -> None:
    candidates = pd.DataFrame([{"candidate_id": "cand-double", "period": 5.0, "epoch": 5.3}])
    events = pd.DataFrame([{"toi_id": "1.01", "period_days": 2.5, "transit_epoch_bjd": 0.3}])
    match = match_candidates(candidates, events, _settings())
    assert match.loc[0, "match_type"] == "double_period"
    assert bool(match.loc[0, "matched"])
    assert match.loc[0, "harmonic_ratio"] == 2.0
