"""Detailed inspection of one frozen candidate scorecard."""

import streamlit as st

from transiteye.ui.charts import candidate_figure
from transiteye.ui.components import configure_page, header, scientific_warning
from transiteye.ui.data import (
    ArtifactUnavailableError,
    candidate_scorecard,
    candidates_with_scorecards,
    lazy_candidate_artifacts,
    load_local_explanations,
)
from transiteye.ui.formatting import humanize, number

configure_page("Candidate Explorer")
header()
dataset = st.selectbox("Dataset", ["demo", "scientific"], format_func=str.title)
try:
    candidates = candidates_with_scorecards(dataset)
    candidate_id = st.selectbox("Candidate ID", candidates.candidate_id.sort_values().tolist())
    scorecard = candidate_scorecard(dataset, candidate_id)
except ArtifactUnavailableError as error:
    st.error(f"Project artifacts are incomplete. Run verification. Details: {error}")
    st.stop()

if dataset == "scientific":
    scientific_warning()

st.subheader("Frozen candidate scorecard")
identity = st.columns(4)
for column, label, key in zip(
    identity,
    ["Candidate", "TIC", "Model score", "Prediction"],
    ["candidate_id", "object_id", "score", "prediction"],
    strict=True,
):
    value = number(scorecard[key]) if key == "score" else scorecard[key]
    column.metric(label, value)
st.caption(
    f"Dataset role: `{scorecard['dataset_role']}` · Model: `{scorecard['model_version']}` · Frozen threshold: `{scorecard['frozen_threshold']}`"
)

st.subheader("BLS information")
bls = st.columns(5)
for column, label, key in zip(
    bls,
    ["Period (days)", "Duration", "Depth", "BLS power", "Rank"],
    [
        "bls_candidate_period",
        "bls_candidate_duration",
        "bls_candidate_depth",
        "bls_candidate_power",
        "bls_candidate_rank",
    ],
    strict=True,
):
    column.metric(label, number(scorecard[key]))

st.subheader("Selected features")
feature_keys = [
    "td_observed_depth",
    "td_depth_significance",
    "td_in_out_scatter_ratio",
    "ls_dominant_power",
    "ls_power_at_bls_frequency",
    "fft_power_concentration",
    "fft_power_at_bls_frequency",
]
feature_columns = st.columns(4)
for index, key in enumerate(feature_keys):
    feature_columns[index % 4].metric(humanize(key), number(scorecard.get(key)))

st.subheader("Model interpretation")
try:
    explanations = load_local_explanations()
    local = explanations.loc[explanations.candidate_id == candidate_id].sort_values("local_rank")
    if local.empty:
        st.caption("No frozen local explanation was selected for this candidate.")
    else:
        st.dataframe(
            local[
                [
                    "local_rank",
                    "feature",
                    "feature_group",
                    "raw_feature_value",
                    "score_contribution_approximation",
                ]
            ],
            hide_index=True,
            width="stretch",
        )
except ArtifactUnavailableError as error:
    st.warning(str(error))
if dataset == "scientific":
    st.caption("These are contributors to the model score, not evidence of planet status.")
    if scorecard["outside_robust_support_count"] > 0:
        st.warning(
            f"Out of robust demo support on {int(scorecard['outside_robust_support_count'])} feature(s)."
        )
    st.write(
        f"Missing feature count: {int(scorecard['missing_feature_count'])} · Triage: `{scorecard['triage_status']}`"
    )

st.subheader("Light curve and frozen BLS artifacts")
with st.spinner("Loading selected candidate artifacts..."):
    cadences, periodogram = lazy_candidate_artifacts(scorecard)
if cadences is None:
    st.info(
        "A matching processed light curve is not available in the frozen bundle for this candidate."
    )
else:
    st.pyplot(
        candidate_figure(cadences, periodogram, float(scorecard["bls_candidate_period"])),
        width="stretch",
    )
    st.caption(
        "The displayed periodogram is loaded from frozen BLS output; the UI does not recompute BLS."
    )
