"""Architecture and truth-firewall explanation."""

import streamlit as st

from transiteye.ui.components import configure_page, header, image_or_warning
from transiteye.ui.data import ArtifactUnavailableError, figure_path

configure_page("Architecture")
header()
try:
    image_or_warning(figure_path("figure_1_system_architecture"), "Final TransitEye architecture")
except ArtifactUnavailableError as error:
    st.error(f"Project artifacts are incomplete. Run verification. Details: {error}")
    st.stop()

left, right = st.columns(2)
left.subheader("Scientific path")
left.write(
    "Real TESS → preprocessing → blind BLS → candidates → shared features → frozen model score"
)
right.subheader("Demo path")
right.write(
    "Real TESS substrate + synthetic injection → preprocessing → blind BLS → candidate matching → shared features → ML training and evaluation"
)
st.subheader("Shared downstream infrastructure")
st.write(
    "Candidate = ML unit. TIC = split and uncertainty unit. Both paths use the same feature and model interfaces after the dataset boundary."
)
st.subheader("Truth firewall")
st.warning(
    "Catalog and synthetic truth are used for labeling and evaluation only. They never enter model features."
)
