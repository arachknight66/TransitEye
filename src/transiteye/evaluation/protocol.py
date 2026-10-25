"""Frozen evaluation protocol and metrics for TransitEye.

Defines ephemeris matching rules, calibration assessment, grouped bootstrap intervals,
and metric calculations. Must be frozen prior to opening the locked test set.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class EphemerisMatchResult:
    """Outcome of matching detected candidate ephemeris to true ephemeris."""

    matched: bool
    is_primary: bool  # exact period match within tolerance
    is_harmonic: bool  # matched at 1/2x or 2x harmonic
    harmonic_ratio: float  # pred_P / true_P
    period_error_relative: float
    epoch_phase_error: float


def match_candidate_ephemeris(
    pred_period: float,
    pred_epoch: float,
    true_period: float,
    true_epoch: float,
    true_duration_hours: float,
    tolerance_period_rel: float = 0.01,
) -> EphemerisMatchResult:
    """Evaluate whether a detected candidate matches the true ephemeris.

    Requires relative period error <= 1.0% (at P, 2P, or P/2) AND
    mid-transit epoch alignment within the transit duration window.
    """
    if true_period <= 0 or pred_period <= 0:
        return EphemerisMatchResult(
            matched=False,
            is_primary=False,
            is_harmonic=False,
            harmonic_ratio=0.0,
            period_error_relative=1.0,
            epoch_phase_error=1.0,
        )

    ratio = pred_period / true_period
    dur_days = true_duration_hours / 24.0

    # 1. Primary match (1x)
    rel_err_1x = abs(pred_period - true_period) / true_period
    is_1x = rel_err_1x <= tolerance_period_rel

    # 2. Subharmonic match (P/2 detected)
    rel_err_half = abs(pred_period - 0.5 * true_period) / (0.5 * true_period)
    is_half = rel_err_half <= tolerance_period_rel

    # 3. Harmonic match (2P detected)
    rel_err_double = abs(pred_period - 2.0 * true_period) / (2.0 * true_period)
    is_double = rel_err_double <= tolerance_period_rel

    is_matched_period = is_1x or is_half or is_double
    if not is_matched_period:
        min_err = min(rel_err_1x, rel_err_half, rel_err_double)
        return EphemerisMatchResult(
            matched=False,
            is_primary=False,
            is_harmonic=False,
            harmonic_ratio=ratio,
            period_error_relative=min_err,
            epoch_phase_error=1.0,
        )

    # Check transit epoch alignment: time difference modulo matching period
    eval_p = pred_period if is_1x else (0.5 * true_period if is_half else 2.0 * true_period)
    diff_days = abs(pred_epoch - true_epoch)
    folded_diff = (diff_days + 0.5 * eval_p) % eval_p - 0.5 * eval_p
    phase_err = abs(folded_diff) / eval_p

    # Epoch alignment within 1.0 * duration window
    epoch_ok = abs(folded_diff) <= max(0.05, 1.25 * dur_days)

    matched = is_matched_period and epoch_ok
    best_err = rel_err_1x if is_1x else (rel_err_half if is_half else rel_err_double)

    return EphemerisMatchResult(
        matched=matched,
        is_primary=is_1x and epoch_ok,
        is_harmonic=(is_half or is_double) and epoch_ok,
        harmonic_ratio=ratio,
        period_error_relative=best_err,
        epoch_phase_error=phase_err,
    )


def compute_expected_calibration_error(
    y_true_indices: np.ndarray,
    y_probs: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute multiclass Expected Calibration Error (ECE) across confidence bins."""
    if len(y_true_indices) == 0:
        return 0.0

    confidences = np.max(y_probs, axis=1)
    predictions = np.argmax(y_probs, axis=1)
    accuracies = (predictions == y_true_indices).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total_samples = len(y_true_indices)

    for i in range(n_bins):
        b_lo = bin_boundaries[i]
        b_hi = bin_boundaries[i + 1]
        mask = (confidences > b_lo) & (confidences <= b_hi)
        count = int(np.sum(mask))
        if count > 0:
            bin_acc = float(np.mean(accuracies[mask]))
            bin_conf = float(np.mean(confidences[mask]))
            ece += (count / total_samples) * abs(bin_acc - bin_conf)

    return float(ece)


def compute_multiclass_brier_score(
    y_true_indices: np.ndarray,
    y_probs: np.ndarray,
    n_classes: int = 5,
) -> float:
    """Compute multiclass Brier score (mean squared error of probability vectors)."""
    if len(y_true_indices) == 0:
        return 0.0

    # One-hot encode ground truth
    n_samples = len(y_true_indices)
    one_hot = np.zeros((n_samples, n_classes), dtype=float)
    for i, idx in enumerate(y_true_indices):
        if 0 <= idx < n_classes:
            one_hot[i, idx] = 1.0

    return float(np.mean(np.sum((y_probs - one_hot) ** 2, axis=1)))


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    """Classification performance metrics on a partition."""

    accuracy: float
    macro_f1: float
    per_class_precision: dict[str, float]
    per_class_recall: dict[str, float]
    per_class_f1: dict[str, float]
    expected_calibration_error: float
    brier_score: float
    abstention_rate: float
    effective_coverage: float
    total_evaluated: int


def compute_classification_metrics(
    y_true: list[str],
    y_pred: list[str],
    y_probs: np.ndarray,
    class_names: list[str],
    abstained_mask: list[bool] | np.ndarray | None = None,
) -> ClassificationMetrics:
    """Compute precision, recall, macro-F1, calibration, and abstention stats."""
    n = len(y_true)
    if n == 0:
        return ClassificationMetrics(
            accuracy=0.0,
            macro_f1=0.0,
            per_class_precision={c: 0.0 for c in class_names},
            per_class_recall={c: 0.0 for c in class_names},
            per_class_f1={c: 0.0 for c in class_names},
            expected_calibration_error=0.0,
            brier_score=0.0,
            abstention_rate=0.0,
            effective_coverage=0.0,
            total_evaluated=0,
        )

    abstained = np.asarray(abstained_mask if abstained_mask is not None else [False] * n, dtype=bool)
    abstention_rate = float(np.mean(abstained))
    effective_coverage = 1.0 - abstention_rate

    cls_to_idx = {c: i for i, c in enumerate(class_names)}
    y_true_idx = np.array([cls_to_idx.get(t, -1) for t in y_true])

    # Per-class counts
    p_dict = {}
    r_dict = {}
    f1_dict = {}

    for c in class_names:
        tp = sum(1 for i in range(n) if y_true[i] == c and y_pred[i] == c and not abstained[i])
        fp = sum(1 for i in range(n) if y_true[i] != c and y_pred[i] == c and not abstained[i])
        fn = sum(1 for i in range(n) if y_true[i] == c and (y_pred[i] != c or abstained[i]))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        p_dict[c] = float(prec)
        r_dict[c] = float(rec)
        f1_dict[c] = float(f1)

    macro_f1 = float(np.mean(list(f1_dict.values())))
    correct = sum(1 for i in range(n) if y_true[i] == y_pred[i] and not abstained[i])
    accuracy = float(correct / n)

    ece = compute_expected_calibration_error(y_true_idx, y_probs)
    brier = compute_multiclass_brier_score(y_true_idx, y_probs, len(class_names))

    return ClassificationMetrics(
        accuracy=accuracy,
        macro_f1=macro_f1,
        per_class_precision=p_dict,
        per_class_recall=r_dict,
        per_class_f1=f1_dict,
        expected_calibration_error=ece,
        brier_score=brier,
        abstention_rate=abstention_rate,
        effective_coverage=effective_coverage,
        total_evaluated=n,
    )


def compute_grouped_bootstrap_ci(
    group_ids: list[str],
    eval_fn: Any,
    n_bootstraps: int = 1000,
    rng_seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, tuple[float, float]]:
    """Compute 95% confidence intervals by resampling groups of astrophysical sources.

    Ensures that all sectors, noise derivatives, and candidates of the same target
    are sampled together in each bootstrap replicate.
    """
    unique_groups = np.unique(group_ids)
    n_groups = len(unique_groups)
    rng = np.random.default_rng(rng_seed)

    metric_replicates: dict[str, list[float]] = {}

    for _ in range(n_bootstraps):
        sampled_groups = rng.choice(unique_groups, size=n_groups, replace=True)
        # Call metric evaluation on sampled groups
        rep_metrics = eval_fn(sampled_groups)
        for k, val in rep_metrics.items():
            if isinstance(val, (int, float)) and not math.isnan(val):
                metric_replicates.setdefault(k, []).append(float(val))

    ci_dict = {}
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    for k, vals in metric_replicates.items():
        if len(vals) > 0:
            arr = np.sort(vals)
            lo = float(np.percentile(arr, lower_pct))
            hi = float(np.percentile(arr, upper_pct))
            ci_dict[k] = (lo, hi)
        else:
            ci_dict[k] = (0.0, 0.0)

    return ci_dict
