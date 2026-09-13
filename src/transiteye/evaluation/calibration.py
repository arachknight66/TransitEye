"""B052 descriptive calibration diagnostics; no calibration transform is fitted."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def calibration_diagnostics(labels: np.ndarray, scores: np.ndarray, *, bins: int) -> dict[str, Any]:
    """Return deterministic equal-frequency reliability statistics."""
    y = np.asarray(labels, dtype=int)
    p = np.asarray(scores, dtype=float)
    if y.shape != p.shape or not y.size or not np.isfinite(p).all():
        raise ValueError("Calibration inputs must be finite, non-empty, and aligned.")
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("Calibration requires binary labels.")
    order = np.argsort(p, kind="stable")
    bin_rows: list[dict[str, float | int]] = []
    weighted_gap = 0.0
    for index, members in enumerate(np.array_split(order, min(bins, len(order)))):
        bin_scores = p[members]
        bin_labels = y[members]
        gap = float(np.mean(bin_scores) - np.mean(bin_labels))
        weighted_gap += len(members) * abs(gap)
        bin_rows.append(
            {
                "bin": index,
                "count": int(len(members)),
                "score_min": float(np.min(bin_scores)),
                "score_max": float(np.max(bin_scores)),
                "mean_predicted_score": float(np.mean(bin_scores)),
                "observed_positive_fraction": float(np.mean(bin_labels)),
                "calibration_gap": gap,
            }
        )
    clipped = np.clip(p, 1e-15, 1 - 1e-15)
    mean_score = float(np.mean(p))
    observed = float(np.mean(y))
    signed_gap = mean_score - observed
    return {
        "candidate_count": int(len(y)),
        "positive_count": int(y.sum()),
        "binning": "equal_frequency_stable_score_order",
        "requested_bins": bins,
        "actual_bins": len(bin_rows),
        "brier_score": float(np.mean((p - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(clipped) + (1 - y) * np.log(1 - clipped))),
        "expected_calibration_error": float(weighted_gap / len(y)),
        "mean_predicted_score": mean_score,
        "observed_positive_fraction": observed,
        "mean_calibration_gap": signed_gap,
        "direction": "overconfident"
        if signed_gap > 0
        else "underconfident"
        if signed_gap < 0
        else "aligned",
        "reliability_bins": bin_rows,
    }


def group_bootstrap_calibration(
    rows: pd.DataFrame, *, bins: int, replicates: int, seed: int
) -> dict[str, Any]:
    """Describe TIC-level resampling variability without population claims."""
    groups = sorted(rows["object_id"].astype(str).unique())
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {"brier_score": [], "expected_calibration_error": []}
    for _ in range(replicates):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        pieces = [rows.loc[rows["object_id"].astype(str) == group] for group in sampled]
        sample = pd.concat(pieces, ignore_index=True)
        result = calibration_diagnostics(
            sample["label"].to_numpy(), sample["score"].to_numpy(), bins=bins
        )
        for metric in values:
            values[metric].append(float(result[metric]))
    return {
        "independent_tic_count": len(groups),
        "requested_replicates": replicates,
        "valid_replicates": replicates,
        "intervals_are_descriptive": True,
        "metrics": {
            metric: {
                "lower": float(np.quantile(samples, 0.025)),
                "median": float(np.quantile(samples, 0.5)),
                "upper": float(np.quantile(samples, 0.975)),
            }
            for metric, samples in values.items()
        },
    }


def scientific_score_support(
    scientific_scores: np.ndarray, demo_negative: np.ndarray, demo_positive: np.ndarray
) -> dict[str, Any]:
    """Compare unlabeled scientific scores with demo score regions."""
    science = np.asarray(scientific_scores, dtype=float)
    negative = np.asarray(demo_negative, dtype=float)
    positive = np.asarray(demo_positive, dtype=float)
    if not science.size or not negative.size or not positive.size:
        raise ValueError("Score support comparison requires all three non-empty samples.")
    neg_min, neg_max = float(negative.min()), float(negative.max())
    pos_min, pos_max = float(positive.min()), float(positive.max())
    ambiguous_low, ambiguous_high = sorted((neg_max, pos_min))
    fractions = {
        "below_demo_negative_support": float(np.mean(science < neg_min)),
        "overlapping_demo_negative_support": float(
            np.mean((science >= neg_min) & (science <= neg_max))
        ),
        "overlapping_ambiguous_demo_region": float(
            np.mean((science > ambiguous_low) & (science < ambiguous_high))
        ),
        "overlapping_demo_positive_support": float(
            np.mean((science >= pos_min) & (science <= pos_max))
        ),
        "above_demo_positive_support": float(np.mean(science > pos_max)),
    }
    return {
        "labels_used": False,
        "calibration_metrics_computed": False,
        "interpretation": "score_distribution_overlap_only_not_planet_probability",
        "demo_negative_support": [neg_min, neg_max],
        "demo_positive_support": [pos_min, pos_max],
        "ambiguous_demo_region": [ambiguous_low, ambiguous_high],
        "fractions": fractions,
    }
