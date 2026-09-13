"""Overview page for Streamlit multipage navigation."""

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
    _, status, _ = project_state()
    scientific, demo = load_candidates("scientific"), load_candidates("demo")
except ArtifactUnavailableError as error:
    st.error(f"Project artifacts are incomplete. Run verification. Details: {error}")
    st.stop()

st.subheader("Project dashboard")
metrics = st.columns(6)
values = [
    len(scientific),
    len(demo),
    65,
    "Random Forest",
    demo.object_id.nunique(),
    status["official_threshold"],
]
for column, label, value in zip(
    metrics,
    [
        "Scientific candidates",
        "Demo candidates",
        "Model features",
        "Selected model",
        "Demo source TICs",
        "Official threshold",
    ],
    values,
    strict=True,
):
    column.metric(label, value)
status_cards(status)
image_or_warning(figure_path("figure_1_system_architecture"), "TransitEye system architecture")
st.subheader("Controlled-demo results")
st.write("Grouped LOTO PR-AUC ≈ **0.974** · F1 ≈ **0.871**")
st.caption("Per-TIC PR-AUC: 0.846–1.000 · F1: 0.625–1.000.")
scientific_warning()
