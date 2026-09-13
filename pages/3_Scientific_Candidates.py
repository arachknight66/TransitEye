"""Unlabeled scientific-candidate explorer."""

import streamlit as st

from transiteye.ui.components import configure_page, header, scientific_warning
from transiteye.ui.data import (
    ArtifactUnavailableError,
    candidates_with_scorecards,
    filter_candidates,
)

configure_page("Scientific Candidates")
header()
try:
    scientific = candidates_with_scorecards("scientific")
except ArtifactUnavailableError as error:
    st.error(f"Project artifacts are incomplete. Run verification. Details: {error}")
    st.stop()

st.subheader("Unlabeled Scientific Candidates")
cards = st.columns(3)
cards[0].metric("Unlabeled candidates", len(scientific))
cards[1].metric("Demo-trained model-predicted positive", int(scientific["prediction"].sum()))
cards[2].metric(
    "Outside robust demo support",
    f"{int((scientific.outside_robust_support_count > 0).sum())} / {len(scientific)}",
)
scientific_warning()
st.caption("A model-predicted positive does not mean that an exoplanet was detected.")

tic_filter = st.multiselect("TIC", sorted(scientific.object_id.unique()))
score_filter = st.slider("Model score", 0.0, 1.0, (0.0, 1.0))
rank_filter = st.slider(
    "BLS rank",
    1,
    int(scientific.bls_candidate_rank.max()),
    (1, int(scientific.bls_candidate_rank.max())),
)
period_filter = st.slider(
    "Period (days)",
    float(scientific.bls_candidate_period.min()),
    float(scientific.bls_candidate_period.max()),
    (float(scientific.bls_candidate_period.min()), float(scientific.bls_candidate_period.max())),
)
triage_filter = st.multiselect("Triage status", sorted(scientific.triage_status.dropna().unique()))
support_filter = st.selectbox("Demo support", ["all", "in support", "out of support"])
filtered = filter_candidates(
    scientific,
    tics=tic_filter,
    score_range=score_filter,
    ranks=rank_filter,
    period_range=period_filter,
    triage=triage_filter,
    support=None if support_filter == "all" else support_filter,
)
columns = [
    "candidate_id",
    "object_id",
    "observation_group_id",
    "bls_candidate_rank",
    "bls_candidate_period",
    "bls_candidate_duration",
    "bls_candidate_depth",
    "bls_candidate_power",
    "score",
    "prediction",
    "triage_status",
    "outside_robust_support_count",
    "missing_feature_count",
]
st.dataframe(filtered[columns], width="stretch", hide_index=True)
