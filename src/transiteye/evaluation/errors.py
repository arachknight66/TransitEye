"""B051 evaluation-only error rows; truth never enters model inputs."""

from __future__ import annotations

import pandas as pd

MAJOR_FEATURES = (
    "bls_candidate_period",
    "bls_candidate_power",
    "bls_candidate_rank",
    "td_depth_significance",
    "ls_dominant_power",
    "fft_power_concentration",
)


def error_analysis_table(
    prediction_sets: tuple[tuple[str, pd.DataFrame], ...],
    features: pd.DataFrame,
    candidates: pd.DataFrame,
    events: pd.DataFrame,
    *,
    high_missingness_count: int,
) -> pd.DataFrame:
    """Describe mistakes from fixed protocols using evaluation metadata only."""
    truth = candidates[
        [
            "candidate_id",
            "object_id",
            "variant_id",
            "match_type",
            "relation_to_stronger",
            "gold_candidate_label",
        ]
    ].merge(
        events[["variant_id", "synthetic_event_class", "difficulty", "variant_name"]],
        on="variant_id",
        validate="many_to_one",
    )
    feature_columns = [name for name in MAJOR_FEATURES if name in features.columns]
    truth = truth.merge(
        features[["candidate_id", *feature_columns]], on="candidate_id", validate="one_to_one"
    )
    identities = {"candidate_id", "object_id", "observation_group_id"}
    missingness = pd.DataFrame(
        {
            "candidate_id": features["candidate_id"].astype(str),
            "missing_feature_count": features.drop(
                columns=[column for column in features.columns if column in identities],
                errors="ignore",
            )
            .isna()
            .sum(axis=1),
        }
    )
    truth = truth.merge(missingness, on="candidate_id", validate="one_to_one")
    outputs: list[pd.DataFrame] = []
    for source, predictions in prediction_sets:
        merged = predictions.merge(truth, on=["candidate_id", "object_id"], validate="one_to_one")
        merged["expected"] = merged["gold_candidate_label"].map({"negative": 0, "positive": 1})
        errors = merged.loc[merged["prediction"] != merged["expected"]].copy()
        if errors.empty:
            continue
        errors["evaluation_source"] = source
        errors["error_type"] = errors["prediction"].map({1: "false_positive", 0: "false_negative"})
        errors["error_category"] = errors.apply(
            lambda row: _category(row, high_missingness_count), axis=1
        )
        outputs.append(errors)
    columns = [
        "evaluation_source",
        "candidate_id",
        "object_id",
        "synthetic_event_class",
        "match_type",
        "difficulty",
        "variant_name",
        "error_type",
        "error_category",
        "score",
        "prediction",
        "expected",
        "missing_feature_count",
        *feature_columns,
    ]
    if not outputs:
        return pd.DataFrame(columns=columns)
    return (
        pd.concat(outputs, ignore_index=True)
        .loc[:, columns]
        .sort_values(["evaluation_source", "candidate_id"], kind="stable")
        .reset_index(drop=True)
    )


def _category(row: pd.Series, high_missingness_count: int) -> str:
    if int(row["missing_feature_count"]) >= high_missingness_count:
        return "high_missingness"
    if str(row["match_type"]) not in {"fundamental", "none", "nan"}:
        return "harmonic_candidate"
    if str(row["difficulty"]) == "weak":
        return "weak_signal"
    if str(row["synthetic_event_class"]) == "eclipsing_binary_like":
        return "eb_like_confusion"
    if str(row["synthetic_event_class"]) == "sinusoidal_variability_like":
        return "sinusoidal_confusion"
    return "other"
