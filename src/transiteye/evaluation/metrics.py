"""Metrics that make the independent TIC group explicit."""

from __future__ import annotations

from typing import Any

import numpy as np

from transiteye.modeling.metrics import compute_metrics


def safe_binary_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    """Return null rate metrics for a single-class group, never fabricated values."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(labels) != len(scores) or not len(labels):
        raise ValueError("Metrics require non-empty aligned labels and scores.")
    predictions = (scores >= threshold).astype(int)
    tp = int(((labels == 1) & (predictions == 1)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    tn = int(((labels == 0) & (predictions == 0)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())
    base: dict[str, Any] = {
        "positive_count": int((labels == 1).sum()),
        "negative_count": int((labels == 0).sum()),
        "candidate_count": int(len(labels)),
        "pr_auc": None,
        "roc_auc": None,
        "balanced_accuracy": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }
    if set(np.unique(labels)) == {0, 1}:
        base.update(compute_metrics(labels, scores, threshold).to_dict())
    return base
