"""B050 unlabeled demo-to-scientific covariate-shift diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _ks_distance(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.sort(left[np.isfinite(left)])
    right = np.sort(right[np.isfinite(right)])
    if not left.size or not right.size:
        return None
    grid = np.unique(np.concatenate([left, right]))
    left_cdf = np.searchsorted(left, grid, side="right") / left.size
    right_cdf = np.searchsorted(right, grid, side="right") / right.size
    return float(np.max(np.abs(left_cdf - right_cdf)))


def feature_shift_table(
    demo_train: pd.DataFrame,
    scientific: pd.DataFrame,
    feature_names: tuple[str, ...],
    *,
    iqr_multiplier: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare distributions and count per-candidate support violations."""
    shift_rows: list[dict[str, object]] = []
    outside_range = pd.DataFrame(index=scientific.index)
    outside_robust = pd.DataFrame(index=scientific.index)
    for feature in feature_names:
        train = demo_train[feature].to_numpy(dtype=float)
        science = scientific[feature].to_numpy(dtype=float)
        finite_train = train[np.isfinite(train)]
        finite_science = science[np.isfinite(science)]
        if finite_train.size:
            q1, median, q3 = np.quantile(finite_train, [0.25, 0.5, 0.75])
            minimum, maximum = float(finite_train.min()), float(finite_train.max())
            iqr = float(q3 - q1)
            low = float(q1 - iqr_multiplier * iqr)
            high = float(q3 + iqr_multiplier * iqr)
            outside_range[feature] = np.isfinite(science) & (
                (science < minimum) | (science > maximum)
            )
            outside_robust[feature] = np.isfinite(science) & ((science < low) | (science > high))
        else:
            q1 = median = q3 = minimum = maximum = iqr = low = high = np.nan
            outside_range[feature] = False
            outside_robust[feature] = False
        science_median = float(np.median(finite_science)) if finite_science.size else None
        science_q1 = float(np.quantile(finite_science, 0.25)) if finite_science.size else None
        science_q3 = float(np.quantile(finite_science, 0.75)) if finite_science.size else None
        science_iqr = (
            float(science_q3 - science_q1)
            if science_q1 is not None and science_q3 is not None
            else None
        )
        shift_rows.append(
            {
                "feature": feature,
                "demo_train_median": None if not finite_train.size else float(median),
                "scientific_median": science_median,
                "median_difference": (
                    None
                    if science_median is None or not finite_train.size
                    else science_median - median
                ),
                "standardized_median_difference": (
                    None
                    if science_median is None or not finite_train.size or iqr == 0
                    else float((science_median - median) / iqr)
                ),
                "demo_train_iqr": None if not finite_train.size else iqr,
                "scientific_iqr": science_iqr,
                "iqr_ratio": (
                    None
                    if science_iqr is None or not finite_train.size or iqr == 0
                    else science_iqr / iqr
                ),
                "demo_train_missing_fraction": float(np.mean(~np.isfinite(train))),
                "scientific_missing_fraction": float(np.mean(~np.isfinite(science))),
                "missingness_difference": float(
                    np.mean(~np.isfinite(science)) - np.mean(~np.isfinite(train))
                ),
                "outside_train_range_fraction": float(outside_range[feature].mean()),
                "outside_robust_support_fraction": float(outside_robust[feature].mean()),
                "ks_distance": _ks_distance(train, science),
                "demo_train_min": None if not finite_train.size else minimum,
                "demo_train_max": None if not finite_train.size else maximum,
                "robust_support_low": None if not finite_train.size else low,
                "robust_support_high": None if not finite_train.size else high,
            }
        )
    shift = pd.DataFrame(shift_rows)
    candidate_support = pd.DataFrame(
        {
            "candidate_id": scientific["candidate_id"].astype(str).to_numpy(),
            "object_id": scientific["object_id"].astype(str).to_numpy(),
            "outside_train_range_count": outside_range.sum(axis=1).to_numpy(dtype=int),
            "outside_robust_support_count": outside_robust.sum(axis=1).to_numpy(dtype=int),
            "missing_feature_count": scientific.loc[:, feature_names]
            .isna()
            .sum(axis=1)
            .to_numpy(dtype=int),
        }
    )
    return shift, candidate_support


def score_summary(scores: pd.Series | np.ndarray) -> dict[str, float]:
    values = np.asarray(scores, dtype=float)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("Score summaries require finite non-empty values.")
    return dict(
        zip(
            ("min", "q1", "median", "q3", "max"),
            (float(value) for value in np.quantile(values, [0.0, 0.25, 0.5, 0.75, 1.0])),
            strict=True,
        )
    )
