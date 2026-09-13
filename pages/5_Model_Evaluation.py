"""Frozen model-evaluation views."""

import streamlit as st

from transiteye.ui.components import configure_page, header, image_or_warning
from transiteye.ui.data import (
    ArtifactUnavailableError,
    figure_path,
    load_result_table,
    project_state,
)

configure_page("Model Evaluation")
header()
try:
    index, status, _ = project_state()
    comparison = load_result_table("table_3_model_comparison.csv")
    ablation = load_result_table("table_4_feature_ablation.csv")
    robustness = load_result_table("table_5_grouped_robustness.csv")
    recovery = load_result_table("table_6_injection_recovery.csv")
except ArtifactUnavailableError as error:
    st.error(f"Project artifacts are incomplete. Run verification. Details: {error}")
    st.stop()

st.subheader("Model comparison")
st.dataframe(comparison, width="stretch", hide_index=True)
st.subheader("Selected model")
st.write(
    f"**Random Forest** · Frozen model `{index['official_model']}` · Threshold `{status['official_threshold']}`"
)

st.subheader("Grouped robustness")
st.dataframe(robustness, width="stretch", hide_index=True)
image_or_warning(figure_path("figure_5_grouped_robustness"), "Grouped leave-one-TIC-out robustness")

st.subheader("Feature ablation")
st.dataframe(ablation, width="stretch", hide_index=True)
image_or_warning(figure_path("figure_4_feature_ablation"), "Frozen feature ablation")

st.subheader("Injection recovery")
st.dataframe(recovery, width="stretch", hide_index=True)
image_or_warning(figure_path("figure_6_injection_recovery"), "Controlled injection recovery")

st.subheader("Calibration and feature importance")
st.info("Model scores are not calibrated scientific planet probabilities.")
image_or_warning(figure_path("figure_9_feature_importance"), "Frozen feature importance")
