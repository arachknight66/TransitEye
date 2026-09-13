"""B054 candidate scorecards and neutral scientific triage."""

from __future__ import annotations

import pandas as pd

TRIAGE_VOCABULARY = {
    "model_high_score",
    "model_mid_score",
    "model_low_score",
    "out_of_support",
    "insufficient_support",
}


def scientific_triage(score: float, threshold: float, robust_violations: int, missing: int) -> str:
    if missing >= 10:
        return "insufficient_support"
    if robust_violations > 0:
        return "out_of_support"
    if score >= threshold:
        return "model_high_score"
    if score >= threshold / 2:
        return "model_mid_score"
    return "model_low_score"


def build_scorecards(
    features: pd.DataFrame,
    candidates: pd.DataFrame,
    predictions: pd.DataFrame,
    support: pd.DataFrame,
    local: pd.DataFrame,
    *,
    dataset_role: str,
    threshold: float,
) -> pd.DataFrame:
    """Create one label-free interpretation row for every candidate."""
    provenance = [
        "candidate_id",
        "object_id",
        "observation_group_id",
        "dataset_version",
        "feature_version",
        "preprocessing_config_hash",
        "bls_config_hash",
    ]
    diagnostics = [
        "candidate_id",
        "bls_candidate_period",
        "bls_candidate_duration",
        "bls_candidate_depth",
        "bls_candidate_power",
        "bls_candidate_rank",
        "bls_power_ratio_to_strongest",
        "bls_local_peak_prominence",
        "td_observed_depth",
        "td_depth_significance",
        "td_in_out_scatter_ratio",
        "ls_dominant_power",
        "ls_power_at_bls_frequency",
        "fft_power_concentration",
        "fft_power_at_bls_frequency",
    ]
    candidate_fields = candidates[
        [
            "candidate_id",
            "candidate_rank",
            "candidate_period",
            "candidate_duration",
            "candidate_depth",
            "bls_power",
            "source_product_id",
            "source_raw_checksum",
        ]
    ]
    frame = (
        features[provenance + diagnostics[1:]]
        .merge(candidate_fields, on="candidate_id", validate="one_to_one")
        .merge(
            predictions[["candidate_id", "score", "prediction", "model_version"]],
            on="candidate_id",
            validate="one_to_one",
        )
        .merge(support, on=["candidate_id", "object_id"], validate="one_to_one")
    )
    local_summary = (
        local.sort_values(["candidate_id", "local_rank"], kind="stable")
        .groupby("candidate_id", sort=False)["feature"]
        .agg(lambda values: "|".join(values.astype(str)))
        .rename("top_local_attribution_features")
    )
    frame = frame.join(local_summary, on="candidate_id")
    frame["top_local_attribution_features"] = frame["top_local_attribution_features"].fillna("")
    frame["dataset_role"] = dataset_role
    frame["frozen_threshold"] = threshold
    frame["triage_status"] = "not_applicable_demo"
    if dataset_role == "scientific_unlabeled":
        frame["triage_status"] = [
            scientific_triage(float(score), threshold, int(outside), int(missing))
            for score, outside, missing in zip(
                frame["score"],
                frame["outside_robust_support_count"],
                frame["missing_feature_count"],
                strict=True,
            )
        ]
    frame["candidate_identity_preserved"] = True
    return frame.sort_values("candidate_id", kind="stable").reset_index(drop=True)
