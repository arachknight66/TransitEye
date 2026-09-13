"""B049 fixed-policy feature-group and missing-feature stability replicas."""

from __future__ import annotations

import numpy as np
import pandas as pd

from transiteye.config import ModelingSettings
from transiteye.evaluation.metrics import safe_binary_metrics
from transiteye.features.registry import feature_names_by_groups
from transiteye.modeling.inference import FrozenModel, predict_candidates
from transiteye.modeling.models import fit_classifier
from transiteye.modeling.transforms import TrainOnlyTransformer

ABLATIONS: dict[str, tuple[str, ...]] = {
    "full": ("time_domain", "bls", "lomb_scargle", "fft"),
    "remove_lomb_scargle": ("time_domain", "bls", "fft"),
    "remove_fft": ("time_domain", "bls", "lomb_scargle"),
    "remove_frequency": ("time_domain", "bls"),
    "time_only": ("time_domain",),
    "bls_only": ("bls",),
}


def feature_ablation_stability(
    rows: pd.DataFrame,
    *,
    modeling: ModelingSettings,
    master_seed: int,
    fixed_threshold: float,
    official_model: FrozenModel,
    period_perturbation_fraction: float,
) -> pd.DataFrame:
    """Refit predefined RF replicas on official train and assess validation only."""
    gold = rows.loc[rows["gold_candidate_label"].isin(["positive", "negative"])]
    train = gold.loc[gold["split"] == "train"]
    validation = gold.loc[gold["split"] == "validation"]
    labels_train = train["gold_candidate_label"].map({"negative": 0, "positive": 1}).to_numpy()
    labels_validation = (
        validation["gold_candidate_label"].map({"negative": 0, "positive": 1}).to_numpy()
    )
    official = predict_candidates(validation, official_model)
    rows_out: list[dict[str, object]] = []
    for name, groups in ABLATIONS.items():
        feature_names = feature_names_by_groups(groups)
        transformer = TrainOnlyTransformer(scale=False)
        train_matrix = transformer.fit_transform(train.loc[:, feature_names])
        validation_matrix = transformer.transform(validation.loc[:, feature_names])
        model = fit_classifier(
            "random_forest",
            train_matrix,
            labels_train,
            settings=modeling,
            master_seed=master_seed,
        )
        scores = model.scores(validation_matrix)
        metrics = safe_binary_metrics(labels_validation, scores, fixed_threshold)
        rows_out.append(
            {
                "analysis": "refit_feature_ablation",
                "variant": name,
                "feature_count": len(feature_names),
                "prediction_agreement_with_official": float(
                    np.mean((scores >= fixed_threshold) == official["prediction"].to_numpy())
                ),
                "mean_absolute_score_difference": float(
                    np.mean(np.abs(scores - official["score"].to_numpy()))
                ),
                **metrics,
            }
        )

    for name, groups in {
        "missing_lomb_scargle": ("lomb_scargle",),
        "missing_fft": ("fft",),
        "missing_frequency": ("lomb_scargle", "fft"),
    }.items():
        stressed = validation.copy()
        stressed.loc[:, feature_names_by_groups(groups)] = np.nan
        result = predict_candidates(stressed, official_model)
        rows_out.append(
            {
                "analysis": "frozen_model_missing_feature_stress",
                "variant": name,
                "feature_count": len(official_model.feature_names),
                "prediction_agreement_with_official": float(
                    np.mean(result["prediction"].to_numpy() == official["prediction"].to_numpy())
                ),
                "mean_absolute_score_difference": float(
                    np.mean(np.abs(result["score"].to_numpy() - official["score"].to_numpy()))
                ),
                **safe_binary_metrics(
                    labels_validation,
                    result["score"].to_numpy(dtype=float),
                    fixed_threshold,
                ),
            }
        )
    for direction in (-1.0, 1.0):
        fraction = direction * period_perturbation_fraction
        stressed = validation.copy()
        stressed["bls_candidate_period"] *= 1.0 + fraction
        stressed["bls_duty_cycle"] /= 1.0 + fraction
        stressed["bls_period_baseline_ratio"] *= 1.0 + fraction
        result = predict_candidates(stressed, official_model)
        rows_out.append(
            {
                "analysis": "frozen_model_candidate_period_perturbation",
                "variant": f"candidate_period_{fraction:+.3%}",
                "feature_count": len(official_model.feature_names),
                "prediction_agreement_with_official": float(
                    np.mean(result["prediction"].to_numpy() == official["prediction"].to_numpy())
                ),
                "mean_absolute_score_difference": float(
                    np.mean(np.abs(result["score"].to_numpy() - official["score"].to_numpy()))
                ),
                **safe_binary_metrics(
                    labels_validation,
                    result["score"].to_numpy(dtype=float),
                    fixed_threshold,
                ),
            }
        )
    return pd.DataFrame(rows_out)
