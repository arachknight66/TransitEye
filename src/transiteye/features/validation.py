"""Feature-table leakage, numerical, and QA validation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from transiteye.features.registry import FEATURE_REGISTRY, model_feature_names
from transiteye.features.schemas import ColumnRole


def validate_feature_matrix(frame: pd.DataFrame) -> dict[str, Any]:
    """Validate one row per candidate and return portable feature QA."""
    feature_columns = list(model_feature_names())
    required = [
        entry.name
        for entry in FEATURE_REGISTRY
        if entry.role in {ColumnRole.IDENTITY, ColumnRole.PROVENANCE, ColumnRole.FEATURE}
        and entry.group != "label_provenance"
    ]
    if list(frame.columns) != required:
        raise ValueError("Feature matrix columns/order do not match the registry.")
    if frame["candidate_id"].duplicated().any():
        raise ValueError("Feature matrix contains duplicate candidate IDs.")
    numeric = frame[feature_columns].to_numpy(dtype=float)
    infinite_count = int(np.isinf(numeric).sum())
    if infinite_count:
        raise ValueError("Feature matrix contains infinite model-feature values.")
    if (frame["bls_candidate_period"] <= 0).any():
        raise ValueError("Candidate periods must be positive.")
    if (frame["bls_candidate_duration"] <= 0).any():
        raise ValueError("Candidate durations must be positive.")
    if (frame["bls_candidate_duration"] >= frame["bls_candidate_period"]).any():
        raise ValueError("Candidate durations must be shorter than periods.")
    duty_cycle = frame["bls_duty_cycle"].dropna()
    if ((duty_cycle <= 0) | (duty_cycle >= 1)).any():
        raise ValueError("BLS duty cycles must be between zero and one.")
    if (frame["bls_estimated_transit_count"] < 0).any():
        raise ValueError("Estimated transit counts cannot be negative.")
    for frequency in ("ls_dominant_frequency_per_day", "fft_dominant_frequency_per_day"):
        if (frame[frequency].dropna() <= 0).any():
            raise ValueError(f"{frequency} must be positive when defined.")
    if not frame["fft_usable"].dropna().isin([0.0, 1.0]).all():
        raise ValueError("FFT usability must be a binary indicator.")
    for entropy in ("ls_spectral_entropy", "fft_spectral_entropy"):
        values = frame[entropy].dropna()
        if ((values < -1e-12) | (values > 1 + 1e-12)).any():
            raise ValueError(f"{entropy} is outside its mathematical range.")
    missing = frame[feature_columns].isna().sum()
    finite = frame[feature_columns].replace([np.inf, -np.inf], np.nan)
    statistics: dict[str, dict[str, float | None]] = {}
    for column in feature_columns:
        values = finite[column].dropna()
        statistics[column] = {
            "min": float(values.min()) if not values.empty else None,
            "median": float(values.median()) if not values.empty else None,
            "max": float(values.max()) if not values.empty else None,
        }
    return {
        "candidate_rows": int(len(frame)),
        "feature_column_count": len(feature_columns),
        "identity_column_count": sum(
            entry.role is ColumnRole.IDENTITY for entry in FEATURE_REGISTRY
        ),
        "label_evaluation_column_count": sum(
            entry.role in {ColumnRole.LABEL, ColumnRole.EVALUATION} for entry in FEATURE_REGISTRY
        ),
        "missing_count": {column: int(missing[column]) for column in feature_columns},
        "missing_fraction": {
            column: float(missing[column] / len(frame)) if len(frame) else 0.0
            for column in feature_columns
        },
        "statistics": statistics,
        "constant_feature_count": int(
            sum(frame[column].nunique(dropna=True) == 1 for column in feature_columns)
        ),
        "all_null_feature_count": int(
            sum(frame[column].isna().all() for column in feature_columns)
        ),
        "infinite_value_count": infinite_count,
        "duplicate_candidate_ids": int(frame["candidate_id"].duplicated().sum()),
        "tic_count": int(frame["object_id"].nunique()),
        "observation_group_count": int(frame["observation_group_id"].nunique()),
    }
