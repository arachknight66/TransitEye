"""TransitEye interactive frontend entry point."""

from __future__ import annotations

import streamlit as st

from transiteye.ui.components import (
    configure_page,
    header,
    image_or_warning,
    scientific_warning,
    status_cards,
)
from transiteye.ui.data import ArtifactUnavailableError, figure_path, load_candidates, project_state

configure_page("Overview")
header()

try:
    index, status, _ = project_state()
    scientific = load_candidates("scientific")
    demo = load_candidates("demo")
except ArtifactUnavailableError as error:
    st.error(f"Project artifacts are incomplete. Run verification. Details: {error}")
    st.stop()

st.subheader("Project dashboard")
cards = st.columns(6)
for card, label, value in zip(
    cards,
    [
        "Scientific candidates",
        "Demo candidates",
        "Model features",
        "Selected model",
        "Demo source TICs",
        "Official threshold",
    ],
    [
        len(scientific),
        len(demo),
        65,
        "Random Forest",
        demo["object_id"].nunique(),
        status["official_threshold"],
    ],
    strict=True,
):
    card.metric(label, value)

st.subheader("Pipeline status")
status_cards(status)

st.subheader("Architecture preview")
image_or_warning(figure_path("figure_1_system_architecture"), "TransitEye system architecture")

st.subheader("Controlled demo results")
left, right = st.columns(2)
left.metric("Grouped LOTO PR-AUC", "0.974")
right.metric("Grouped LOTO F1", "0.871")
st.caption(
    "Per-TIC PR-AUC: 0.846–1.000 · Per-TIC F1: 0.625–1.000. These are controlled demo metrics."
)

scientific_warning()
st.caption(
    f"Frozen result index selects model `{index['official_model']}` and final evaluation `{index['final_evaluation']}`."
)
