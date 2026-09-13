"""B053 deterministic interpretation of the frozen random forest."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance  # type: ignore[import-untyped]

from transiteye.features.registry import FEATURE_REGISTRY
from transiteye.features.schemas import ColumnRole
from transiteye.modeling.inference import FrozenModel


def feature_group(feature: str) -> str:
    groups = {
        definition.name: definition.group
        for definition in FEATURE_REGISTRY
        if definition.role is ColumnRole.FEATURE
    }
    return groups[feature]


def global_feature_importance(
    model: FrozenModel,
    validation: pd.DataFrame,
    labels: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Measure validation permutation importance and secondary RF impurity importance."""
    if model.classifier.family != "random_forest":
        raise ValueError("Interpretability protocol requires the frozen random forest.")
    if tuple(validation.loc[:, model.feature_names].columns) != model.feature_names:
        raise ValueError("Attribution schema differs from frozen model features.")
    transformed = model.transformer.transform(validation.loc[:, model.feature_names])
    result = permutation_importance(
        model.classifier.estimator,
        transformed,
        labels,
        scoring="average_precision",
        n_repeats=repeats,
        random_state=seed,
        n_jobs=1,
    )
    impurity = np.asarray(model.classifier.estimator.feature_importances_, dtype=float)
    table = (
        pd.DataFrame(
            {
                "feature": model.feature_names,
                "feature_group": [feature_group(name) for name in model.feature_names],
                "permutation_importance_mean": result.importances_mean,
                "permutation_importance_std": result.importances_std,
                "impurity_importance_secondary": impurity,
            }
        )
        .sort_values(
            ["permutation_importance_mean", "feature"], ascending=[False, True], kind="stable"
        )
        .reset_index(drop=True)
    )
    table["permutation_rank"] = np.arange(1, len(table) + 1)
    positive = table["permutation_importance_mean"].clip(lower=0)
    group_totals = positive.groupby(table["feature_group"], sort=True).sum()
    total = float(group_totals.sum())
    grouped = {
        str(group): {
            "positive_permutation_importance_sum": float(value),
            "positive_importance_fraction": float(value / total) if total else 0.0,
        }
        for group, value in group_totals.items()
    }
    frequency_fraction = sum(
        details["positive_importance_fraction"]
        for group, details in grouped.items()
        if group in {"lomb_scargle", "fft"}
    )
    summary = {
        "primary_method": "validation_permutation_importance_average_precision",
        "secondary_method": "random_forest_mean_decrease_impurity",
        "secondary_method_caution": (
            "impurity importance can favor continuous or high-cardinality features"
        ),
        "locked_test_used_for_ranking": False,
        "feature_groups": grouped,
        "frequency_positive_importance_fraction": frequency_fraction,
        "frequency_features_materially_contribute": frequency_fraction >= 0.10,
    }
    return table, summary


def local_median_perturbation(
    model: FrozenModel,
    rows: pd.DataFrame,
    examples: Iterable[dict[str, str]],
    *,
    top_k: int,
) -> pd.DataFrame:
    """Explain selected scores by replacing one raw feature with its frozen train median."""
    if tuple(rows.loc[:, model.feature_names].columns) != model.feature_names:
        raise ValueError("Attribution input schema must exactly match frozen model features.")
    if model.transformer.medians is None:
        raise ValueError("Frozen transformer medians are unavailable.")
    medians = dict(zip(model.transformer.retained_features, model.transformer.medians, strict=True))
    output: list[dict[str, object]] = []
    by_id = rows.set_index("candidate_id", drop=False)
    for example in examples:
        candidate_id = example["candidate_id"]
        original = by_id.loc[[candidate_id]].copy()
        score = float(
            model.classifier.scores(
                model.transformer.transform(original.loc[:, model.feature_names])
            )[0]
        )
        contributions: list[tuple[str, float, float]] = []
        for feature in model.feature_names:
            perturbed = original.copy()
            raw = float(perturbed.iloc[0][feature])
            perturbed.loc[:, feature] = float(medians[feature])
            perturbed_score = float(
                model.classifier.scores(
                    model.transformer.transform(perturbed.loc[:, model.feature_names])
                )[0]
            )
            contributions.append((feature, score - perturbed_score, raw))
        contributions.sort(key=lambda item: (-abs(item[1]), item[0]))
        for rank, (feature, contribution, raw_value) in enumerate(contributions[:top_k], start=1):
            output.append(
                {
                    "candidate_id": candidate_id,
                    "object_id": str(original.iloc[0]["object_id"]),
                    "dataset_role": example["dataset_role"],
                    "example_role": example["example_role"],
                    "selection_source": example["selection_source"],
                    "model_score": score,
                    "feature": feature,
                    "feature_group": feature_group(feature),
                    "local_rank": rank,
                    "raw_feature_value": raw_value,
                    "frozen_train_median": float(medians[feature]),
                    "score_contribution_approximation": contribution,
                    "method": "single_feature_frozen_median_perturbation",
                }
            )
    return pd.DataFrame(output)


def shift_importance_risk(importance: pd.DataFrame, shift: pd.DataFrame) -> pd.DataFrame:
    """Rank plausible transfer risks using a documented deterministic rule."""
    merged = importance.merge(shift, on="feature", validate="one_to_one")
    high_importance = merged["permutation_importance_mean"] >= max(
        float(merged["permutation_importance_mean"].quantile(0.75)), 0.0
    )
    shift_magnitude = merged["standardized_median_difference"].abs().fillna(0.0)
    strong_shift = (
        (shift_magnitude >= 1.0)
        | (merged["outside_robust_support_fraction"] >= 0.25)
        | (merged["missingness_difference"].abs() >= 0.10)
    )
    merged["risk_flag"] = np.select(
        [high_importance & strong_shift, high_importance | strong_shift],
        ["high", "moderate"],
        default="low",
    )
    merged["risk_score"] = merged["permutation_importance_mean"].clip(lower=0) * (
        1
        + shift_magnitude
        + merged["outside_robust_support_fraction"]
        + merged["missingness_difference"].abs()
    )
    columns = [
        "feature",
        "feature_group",
        "permutation_importance_mean",
        "permutation_importance_std",
        "standardized_median_difference",
        "outside_robust_support_fraction",
        "missingness_difference",
        "risk_flag",
        "risk_score",
    ]
    return (
        merged.loc[:, columns]
        .sort_values(["risk_score", "feature"], ascending=[False, True], kind="stable")
        .reset_index(drop=True)
    )
