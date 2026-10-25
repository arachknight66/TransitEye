"""Probability calibration, Expected Calibration Error (ECE), and abstention tuning."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.calibration import CalibratedClassifierCV


def compute_multiclass_brier_score(
    y_true_indices: NDArray[np.int64],
    y_probs: NDArray[np.float64],
    n_classes: int,
) -> float:
    """Compute multiclass Brier score: (1/N) * sum_i sum_c (p_ic - y_ic)^2."""
    n_samples = len(y_true_indices)
    if n_samples == 0:
        return 0.0

    y_one_hot = np.zeros((n_samples, n_classes), dtype=np.float64)
    y_one_hot[np.arange(n_samples), y_true_indices] = 1.0

    diff = y_probs - y_one_hot
    return float(np.mean(np.sum(diff**2, axis=1)))


def compute_expected_calibration_error(
    y_true_indices: NDArray[np.int64],
    y_probs: NDArray[np.float64],
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE) across confidence bins."""
    n_samples = len(y_true_indices)
    if n_samples == 0:
        return 0.0

    confidences = np.max(y_probs, axis=1)
    predictions = np.argmax(y_probs, axis=1)
    accuracies = (predictions == y_true_indices).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


def fit_platt_calibrator(
    base_estimator: Any,
    x_cal: NDArray[np.float64],
    y_cal_indices: NDArray[np.int64],
) -> CalibratedClassifierCV:
    """Fit sigmoid (Platt) regularized calibration on held-out calibration partition."""
    min_class_count = int(np.min(np.bincount(y_cal_indices)))
    n_splits = max(2, min(5, min_class_count))

    try:
        from sklearn.frozen import FrozenEstimator

        cal_estimator = FrozenEstimator(base_estimator)
        calibrator = CalibratedClassifierCV(
            estimator=cal_estimator,
            method="sigmoid",
            cv=n_splits,
        )
    except ImportError:
        calibrator = CalibratedClassifierCV(
            estimator=base_estimator,
            method="sigmoid",
            cv="prefit",
        )
    calibrator.fit(x_cal, y_cal_indices)
    return calibrator


def tune_abstention_thresholds(
    y_true_indices: NDArray[np.int64],
    y_probs: NDArray[np.float64],
    min_coverage: float = 0.75,
) -> tuple[float, float]:
    """Select optimal (tau_conf, tau_margin) on selection partition.

    Balances classification accuracy on covered candidates with minimum coverage constraint.
    """
    sorted_probs = np.sort(y_probs, axis=1)[:, ::-1]
    top_probs = sorted_probs[:, 0]
    second_probs = sorted_probs[:, 1]
    margins = top_probs - second_probs
    predictions = np.argmax(y_probs, axis=1)

    cand_conf_thresholds = [0.4, 0.5, 0.6, 0.7, 0.8]
    cand_margin_thresholds = [0.1, 0.15, 0.2, 0.25, 0.3]

    best_score = -1.0
    best_tau_conf = 0.5
    best_tau_margin = 0.15

    for tau_c in cand_conf_thresholds:
        for tau_m in cand_margin_thresholds:
            covered = (top_probs >= tau_c) & (margins >= tau_m)
            coverage = float(np.mean(covered))

            if coverage < min_coverage:
                continue

            if np.sum(covered) > 0:
                acc = float(np.mean(predictions[covered] == y_true_indices[covered]))
                # Harmonic mean of accuracy and coverage
                score = (2.0 * acc * coverage) / (acc + coverage + 1e-6)
            else:
                score = 0.0

            if score > best_score:
                best_score = score
                best_tau_conf = tau_c
                best_tau_margin = tau_m

    return best_tau_conf, best_tau_margin
