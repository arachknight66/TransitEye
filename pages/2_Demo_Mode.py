"""Controlled demo-mode explorer."""

import pandas as pd
import streamlit as st

from transiteye.ui.components import configure_page, header
from transiteye.ui.data import (
    ArtifactUnavailableError,
    candidates_with_scorecards,
    filter_candidates,
    load_result_table,
)

configure_page("Demo Mode")
header()
try:
    demo = candidates_with_scorecards("demo")
    recovery = load_result_table("table_6_injection_recovery.csv")
except ArtifactUnavailableError as error:
    st.error(f"Project artifacts are incomplete. Run verification. Details: {error}")
    st.stop()

label_column = "gold_candidate_label" if "gold_candidate_label" in demo else None
labels = demo[label_column] if label_column else pd.Series(index=demo.index, dtype="object")
metrics = st.columns(5)
for column, label, value in zip(
    metrics,
    ["Demo candidates", "Positive", "Negative", "Unlabeled", "Source TICs"],
    [
        len(demo),
        int((labels == "positive").sum()),
        int((labels == "negative").sum()),
        int((labels == "unlabeled").sum() if label_column else len(demo)),
        demo.object_id.nunique(),
    ],
    strict=True,
):
    column.metric(label, value)

st.subheader("Explore controlled demo candidates")
tic_filter = st.multiselect("Source TIC", sorted(demo.object_id.unique()))
rank_max = int(demo["bls_candidate_rank"].max())
rank_filter = st.slider("BLS rank", 1, rank_max, (1, rank_max))
score_filter = st.slider("Frozen model score", 0.0, 1.0, (0.0, 1.0))
filtered = filter_candidates(demo, tics=tic_filter, ranks=rank_filter, score_range=score_filter)
show = [
    column
    for column in [
        "candidate_id",
        "object_id",
        label_column,
        "candidate_rank",
        "score",
        "prediction",
        "bls_candidate_period",
        "bls_candidate_rank",
    ]
    if column
]
st.dataframe(filtered[show], width="stretch", hide_index=True)

st.subheader("Injection recovery")
st.dataframe(recovery, width="stretch", hide_index=True)
st.caption(
    "Ground truth below is demo-only evaluation metadata; it never alters blind detection or model inputs."
)
