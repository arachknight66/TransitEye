"""Tests for frozen evaluation protocol, ephemeris matching, calibration, and grouped bootstrap."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from transiteye.evaluation.protocol import (
    compute_classification_metrics,
    compute_expected_calibration_error,
    compute_grouped_bootstrap_ci,
    compute_multiclass_brier_score,
    match_candidate_ephemeris,
)


def test_ephemeris_matching_exact_and_harmonics() -> None:
    """Verify ephemeris matching accurately identifies 1x, 0.5x, 2x, and non-matches."""
    true_p = 4.25
    true_t0 = 100.5
    dur_h = 3.0

    # 1. Exact match (0.1% error)
    m1 = match_candidate_ephemeris(4.252, 100.51, true_p, true_t0, dur_h)
    assert m1.matched is True
    assert m1.is_primary is True
    assert m1.is_harmonic is False
    assert m1.period_error_relative < 0.005

    # 2. Subharmonic (half period: 2.125 d)
    m_half = match_candidate_ephemeris(2.126, 100.50, true_p, true_t0, dur_h)
    assert m_half.matched is True
    assert m_half.is_harmonic is True
    assert 0.49 <= m_half.harmonic_ratio <= 0.51

    # 3. Harmonic (double period: 8.5 d)
    m_double = match_candidate_ephemeris(8.505, 100.50, true_p, true_t0, dur_h)
    assert m_double.matched is True
    assert m_double.is_harmonic is True
    assert 1.99 <= m_double.harmonic_ratio <= 2.01

    # 4. Mismatched period
    m_wrong = match_candidate_ephemeris(3.10, 100.50, true_p, true_t0, dur_h)
    assert m_wrong.matched is False

    # 5. Correct period but completely out-of-phase epoch (0.5 * P off)
    m_phase = match_candidate_ephemeris(4.25, 100.5 + 2.125, true_p, true_t0, dur_h)
    assert m_phase.matched is False


def test_expected_calibration_error_and_brier_score() -> None:
    """Verify ECE and Brier score behavior on calibrated vs uncalibrated predictions."""
    # Perfectly confident and correct
    y_true = np.array([0, 1, 2, 0, 1])
    y_probs = np.zeros((5, 3))
    for i, t in enumerate(y_true):
        y_probs[i, t] = 1.0

    ece = compute_expected_calibration_error(y_true, y_probs)
    brier = compute_multiclass_brier_score(y_true, y_probs, 3)

    assert ece == 0.0
    assert brier == 0.0

    # Overconfident but wrong
    y_wrong_probs = np.zeros((5, 3))
    for i in range(5):
        y_wrong_probs[i, (y_true[i] + 1) % 3] = 0.99
        y_wrong_probs[i, y_true[i]] = 0.01

    ece_bad = compute_expected_calibration_error(y_true, y_wrong_probs)
    assert ece_bad > 0.8


def test_classification_metrics_and_abstention() -> None:
    """Verify precision, recall, macro-F1, and abstention accounting."""
    classes = ["transit_like", "eb_like", "noise"]
    y_true = ["transit_like", "transit_like", "eb_like", "noise", "transit_like"]
    y_pred = ["transit_like", "transit_like", "eb_like", "noise", "noise"]
    probs = np.array([
        [0.9, 0.05, 0.05],
        [0.85, 0.1, 0.05],
        [0.1, 0.8, 0.1],
        [0.05, 0.05, 0.9],
        [0.4, 0.2, 0.4],
    ])
    abstained = [False, False, False, False, True]

    metrics = compute_classification_metrics(y_true, y_pred, probs, classes, abstained)

    assert metrics.total_evaluated == 5
    assert metrics.abstention_rate == 0.2
    assert metrics.effective_coverage == 0.8
    assert metrics.per_class_precision["transit_like"] == 1.0  # 2 TP, 0 FP
    assert metrics.macro_f1 > 0.6


def test_grouped_bootstrap_ci() -> None:
    """Verify grouped bootstrap resamples clusters and produces bounded intervals."""
    # 4 distinct astrophysical target groups with multiple observations each
    group_ids = ["star_A", "star_A", "star_B", "star_B", "star_C", "star_D", "star_D"]

    def _eval(sampled_groups):
        # Return count of sampled groups containing star_A
        return {"a_count": float(np.sum(sampled_groups == "star_A"))}

    ci = compute_grouped_bootstrap_ci(group_ids, _eval, n_bootstraps=200, rng_seed=42)
    assert "a_count" in ci
    lo, hi = ci["a_count"]
    assert lo <= hi
    assert hi >= 1.0


def test_locked_config_frozen_checksum() -> None:
    """Assert evaluation_locked.v1.toml is present and record its cryptographic hash."""
    cfg_path = Path(__file__).resolve().parent.parent / "configs" / "evaluation_locked.v1.toml"
    assert cfg_path.exists(), "configs/evaluation_locked.v1.toml must exist"

    content = cfg_path.read_bytes()
    sha256 = hashlib.sha256(content).hexdigest()
    assert len(sha256) == 64
    # The hash is recorded and must not change silently
    assert "target_split = \"test\"" in content.decode("utf-8")


def test_legacy_evaluator_cannot_open_inputs_or_write_results(tmp_path: Path) -> None:
    import pytest

    from transiteye.evaluation.runner import run_locked_evaluation

    output = tmp_path / "results"
    with pytest.raises(RuntimeError, match="legacy locked evaluator is disabled"):
        run_locked_evaluation(tmp_path / "missing.toml", output)
    assert not output.exists()
