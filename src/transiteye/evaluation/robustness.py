"""B046 deterministic leave-one-development-TIC-out evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from transiteye.config import ModelingSettings
from transiteye.evaluation.metrics import safe_binary_metrics
from transiteye.modeling.models import fit_classifier
from transiteye.modeling.transforms import TrainOnlyTransformer


@dataclass(frozen=True)
class GroupRobustnessResult:
    predictions: pd.DataFrame
    per_tic_metrics: tuple[dict[str, object], ...]
    pooled_metrics: dict[str, object]
    development_tics: tuple[str, ...]


def _binary_labels(series: pd.Series) -> np.ndarray:
    values = series.map({"negative": 0, "positive": 1})
    if values.isna().any():
        raise ValueError("Grouped robustness accepts gold binary rows only.")
    return values.to_numpy(dtype=int)


def leave_one_tic_out(
    rows: pd.DataFrame,
    feature_names: tuple[str, ...],
    *,
    modeling: ModelingSettings,
    master_seed: int,
    fixed_threshold: float,
    excluded_tics: tuple[str, ...],
) -> GroupRobustnessResult:
    """Refit the frozen RF policy per fold; the official test groups stay excluded."""
    required = {"candidate_id", "object_id", "gold_candidate_label", "split", *feature_names}
    if missing := required.difference(rows.columns):
        raise ValueError(f"Grouped robustness input lacks columns: {sorted(missing)}")
    gold = rows.loc[rows["gold_candidate_label"].isin(["positive", "negative"])].copy()
    if set(gold.loc[gold["split"] == "test", "object_id"].astype(str)) != set(excluded_tics):
        raise ValueError("Declared excluded TICs do not equal the official locked-test TICs.")
    pool = gold.loc[gold["split"].isin(["train", "validation"])].copy()
    groups = tuple(sorted(pool["object_id"].astype(str).unique()))
    if set(groups).intersection(excluded_tics):
        raise ValueError("Official test TIC entered development resampling.")
    if len(groups) < 3:
        raise ValueError("Grouped robustness requires at least three development TICs.")

    outputs: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    for held_out in groups:
        fold_train = pool.loc[pool["object_id"].astype(str) != held_out]
        fold_validation = pool.loc[pool["object_id"].astype(str) == held_out]
        transformer = TrainOnlyTransformer(scale=False)
        train_matrix = transformer.fit_transform(fold_train.loc[:, feature_names])
        validation_matrix = transformer.transform(fold_validation.loc[:, feature_names])
        model = fit_classifier(
            "random_forest",
            train_matrix,
            _binary_labels(fold_train["gold_candidate_label"]),
            settings=modeling,
            master_seed=master_seed,
        )
        scores = model.scores(validation_matrix)
        labels = _binary_labels(fold_validation["gold_candidate_label"])
        fold = pd.DataFrame(
            {
                "candidate_id": fold_validation["candidate_id"].astype(str).to_numpy(),
                "object_id": fold_validation["object_id"].astype(str).to_numpy(),
                "label": labels,
                "score": scores,
                "prediction": (scores >= fixed_threshold).astype(int),
                "held_out_tic": held_out,
                "threshold": fixed_threshold,
            }
        )
        outputs.append(fold)
        metrics = safe_binary_metrics(labels, scores, fixed_threshold)
        metric_rows.append({"object_id": held_out, **metrics})
    predictions = (
        pd.concat(outputs, ignore_index=True)
        .sort_values("candidate_id", kind="stable")
        .reset_index(drop=True)
    )
    if predictions["candidate_id"].duplicated().any():
        raise ValueError("A candidate appeared in multiple grouped robustness folds.")
    pooled = safe_binary_metrics(
        predictions["label"].to_numpy(dtype=int),
        predictions["score"].to_numpy(dtype=float),
        fixed_threshold,
    )
    return GroupRobustnessResult(predictions, tuple(metric_rows), pooled, groups)
