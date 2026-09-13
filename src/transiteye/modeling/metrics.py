"""Validation-only selection and locked binary metric evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (  # type: ignore[import-untyped]
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class BinaryMetrics:
    pr_auc: float
    roc_auc: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    tn: int
    fn: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def choose_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """Choose the highest validation-score threshold among maximum-F1 ties."""
    if len(labels) != len(scores) or not len(labels):
        raise ValueError("Threshold selection requires aligned validation labels and scores.")
    candidates = np.unique(scores)
    candidates = np.append(candidates, np.nextafter(float(np.min(scores)), -np.inf))
    ranked = [
        (float(f1_score(labels, scores >= threshold, zero_division=0)), float(threshold))
        for threshold in candidates
    ]
    return max(ranked, key=lambda value: (value[0], value[1]))[1]


def compute_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> BinaryMetrics:
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Binary metrics require both classes.")
    prediction = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, prediction, labels=[0, 1]).ravel()
    return BinaryMetrics(
        pr_auc=float(average_precision_score(labels, scores)),
        roc_auc=float(roc_auc_score(labels, scores)),
        balanced_accuracy=float(balanced_accuracy_score(labels, prediction)),
        precision=float(precision_score(labels, prediction, zero_division=0)),
        recall=float(recall_score(labels, prediction, zero_division=0)),
        f1=float(f1_score(labels, prediction, zero_division=0)),
        tp=int(tp),
        fp=int(fp),
        tn=int(tn),
        fn=int(fn),
    )


class LockedTestEvaluator:
    """One-use guard that prevents iterative inspection of a locked test partition."""

    def __init__(self, labels: np.ndarray) -> None:
        self._labels = labels.copy()
        self._used = False

    def evaluate(self, scores: np.ndarray, threshold: float) -> BinaryMetrics:
        if self._used:
            raise RuntimeError("Locked test partition has already been evaluated.")
        self._used = True
        return compute_metrics(self._labels, scores, threshold)
