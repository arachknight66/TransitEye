"""Shared B045 inference contract for conforming feature matrices."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from transiteye.modeling.models import TrainedClassifier
from transiteye.modeling.transforms import TrainOnlyTransformer


@dataclass(frozen=True)
class FrozenModel:
    model_version: str
    feature_names: tuple[str, ...]
    transformer: TrainOnlyTransformer
    classifier: TrainedClassifier
    threshold: float
    statistical_role: str


def predict_candidates(feature_matrix: pd.DataFrame, model: FrozenModel) -> pd.DataFrame:
    """Score candidates without branching on demo/scientific dataset role."""
    if "candidate_id" not in feature_matrix:
        raise ValueError("Inference input lacks candidate identity.")
    if missing := set(model.feature_names).difference(feature_matrix.columns):
        raise ValueError(f"Inference input lacks model features: {sorted(missing)}")
    transformed = model.transformer.transform(feature_matrix.loc[:, model.feature_names])
    scores = model.classifier.scores(transformed)
    if len(scores) != len(feature_matrix) or not np.isfinite(scores).all():
        raise ValueError("Inference produced invalid scores.")
    return pd.DataFrame(
        {
            "candidate_id": feature_matrix["candidate_id"].astype(str).to_numpy(),
            "score": scores,
            "prediction": (scores >= model.threshold).astype(int),
            "model_version": model.model_version,
        }
    )
