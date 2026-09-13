"""Shared Streamlit page elements and claim-bounded content."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from .data import ArtifactUnavailableError, project_state


def configure_page(title: str) -> None:
    st.set_page_config(page_title=f"TransitEye | {title}", page_icon="✦", layout="wide")
    st.markdown(
        """<style>
        .stApp { background: #0b1220; }
        [data-testid="stMetric"] { background: #142033; border: 1px solid #263a55;
          border-radius: 10px; padding: 12px; }
        .status-card { border: 1px solid #263a55; border-radius: 10px; padding: 15px;
          background: #142033; min-height: 105px; }
        .status-ok { color: #8bd3c7; font-weight: 700; }
        .status-warn { color: #f6c177; font-weight: 700; }
        </style>""",
        unsafe_allow_html=True,
    )


def header() -> None:
    st.title("TransitEye")
    st.caption("Machine Learning-Based Exoplanet Detection from Stellar Light Curves")
    try:
        _, status, release = project_state()
    except ArtifactUnavailableError as error:
        st.error(f"Project artifacts are incomplete. Run verification. Details: {error}")
        return
    st.sidebar.title("TransitEye")
    st.sidebar.caption(f"Version {release['package_version']}")
    st.sidebar.success("Software ✓  Demo ✓")
    st.sidebar.warning("Scientific validation pending")
    st.caption(
        f"Release `{release['release_version_id']}` · Model `{status['official_model']}` · "
        f"Threshold `{status['official_threshold']}`"
    )


def status_cards(status: dict[str, Any]) -> None:
    cards = st.columns(3)
    values = [
        ("Software Pipeline", str(status["software_pipeline_status"]).upper(), "status-ok"),
        ("Demo Validation", str(status["demo_pipeline_status"]).upper(), "status-ok"),
        ("Scientific Readiness", str(status["scientific_readiness"]).upper(), "status-warn"),
    ]
    for card, (label, value, css) in zip(cards, values, strict=True):
        card.markdown(
            f'<div class="status-card"><small>{label}</small><br><span class="{css}">{value}</span></div>',
            unsafe_allow_html=True,
        )
    st.info(
        "Scientific readiness is blocked because independently labeled real candidate coverage is "
        "currently insufficient and significant demo-to-scientific domain shift remains."
    )


def image_or_warning(path: Path, caption: str) -> None:
    if path.is_file():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.warning(f"Optional figure unavailable: {caption}")


def scientific_warning() -> None:
    st.warning(
        "The 75 scientific candidates are unlabeled. Demo-trained model predictions are exploratory "
        "scores, not confirmed exoplanets or scientifically validated classifications."
    )
