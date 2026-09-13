"""B047 group-aware uncertainty over TIC-level prediction clusters."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from transiteye.evaluation.metrics import safe_binary_metrics


def group_bootstrap_intervals(
    predictions: pd.DataFrame,
    *,
    threshold: float,
    replicates: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap TIC groups, preserving every row belonging to a sampled group."""
    groups = tuple(sorted(predictions["object_id"].astype(str).unique()))
    if len(groups) < 2:
        raise ValueError("Group bootstrap requires at least two TIC groups.")
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {
        "pr_auc": [],
        "roc_auc": [],
        "balanced_accuracy": [],
        "f1": [],
    }
    valid = 0
    for _ in range(replicates):
        selected = rng.choice(groups, size=len(groups), replace=True)
        parts = [
            predictions.loc[predictions["object_id"].astype(str) == group] for group in selected
        ]
        replicate = pd.concat(parts, ignore_index=True)
        result = safe_binary_metrics(
            replicate["label"].to_numpy(dtype=int),
            replicate["score"].to_numpy(dtype=float),
            threshold,
        )
        if result["pr_auc"] is None:
            continue
        valid += 1
        for metric in samples:
            samples[metric].append(float(result[metric]))
    alpha = (1.0 - confidence_level) / 2.0
    summaries: dict[str, Any] = {}
    for metric, values in samples.items():
        array = np.asarray(values, dtype=float)
        summaries[metric] = {
            "mean": float(np.mean(array)) if array.size else None,
            "median": float(np.median(array)) if array.size else None,
            "lower": float(np.quantile(array, alpha)) if array.size else None,
            "upper": float(np.quantile(array, 1.0 - alpha)) if array.size else None,
        }
    return {
        "independent_group_count": len(groups),
        "requested_replicates": replicates,
        "valid_replicates": valid,
        "confidence_level": confidence_level,
        "metrics": summaries,
    }
