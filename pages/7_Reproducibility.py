"""Release status and safe existing reproducibility operations."""

import streamlit as st

from transiteye.ui.components import configure_page, header
from transiteye.ui.data import (
    ArtifactUnavailableError,
    project_state,
    run_demo_smoke,
    run_verification,
)

configure_page("Reproducibility")
header()
try:
    index, status, release = project_state()
except ArtifactUnavailableError as error:
    st.error(f"Project artifacts are incomplete. Run verification. Details: {error}")
    st.stop()

st.subheader("Frozen release identities")
st.code(
    f"release = {release['release_version_id']}\nrepro = {status['repro_version']}\nmodel = {status['official_model']}\nfinal_evaluation = {status['final_evaluation']}"
)
st.subheader("Artifact verification")
st.write(
    "The frozen package registers 613 artifacts. Verification uses the existing backend verifier."
)
if st.button("Verify Frozen Artifacts", type="primary"):
    with st.spinner("Verifying frozen artifacts..."):
        try:
            result = run_verification()
            st.success(
                f"Verified {result['verified']} artifacts; missing: {len(result['missing'])}; checksum mismatches: {len(result['checksum_mismatch'])}; metadata mismatches: {len(result['metadata_mismatch'])}."
            )
        except Exception as error:  # Streamlit must present a useful failure rather than crash.
            st.error(f"Verification failed: {error}")

st.subheader("Demo smoke")
st.write("Runs the compact, offline existing smoke path. It does not train a model.")
if st.button("Run Demo Smoke Test"):
    with st.spinner("Running the deterministic smoke test..."):
        try:
            result = run_demo_smoke()
            st.success(
                f"{result['status'].title()}: expected period {result['expected_period_days']} days; recovered {result['bls_best_period_days']:.4f} days; candidates {result['bls_candidate_count']}; inference rows {result['frozen_inference_rows']}."
            )
        except Exception as error:  # Streamlit must present a useful failure rather than crash.
            st.error(f"Demo smoke failed: {error}")
