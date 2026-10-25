#!/usr/bin/env python3
"""TransitEye Quickstart Tutorial: End-to-end Python API Walkthrough.

Demonstrates loading a TESS light curve, two-pass duration-aware detrending,
blind BLS frequency grid search, candidate refinement, and two-tier source vetting.
"""

from __future__ import annotations

import sys
from pathlib import Path

from transiteye.characterization.fitting import refine_candidate
from transiteye.detection.peaks import find_candidates
from transiteye.io.fits import read_tess_light_curve
from transiteye.preprocessing.detrend import two_pass_detrend
from transiteye.preprocessing.representation import prepare_light_curve
from transiteye.vetting.source import vet_source


def main() -> int:
    # 1. Locate pilot test data
    pilot_dir = Path(".transiteye-workspace/data/pilot")
    sample_files = list(pilot_dir.glob("*.fits")) if pilot_dir.is_dir() else []

    if not sample_files:
        print("No pilot FITS files found in .transiteye-workspace/data/pilot.")
        print("Run `python scripts/download_pilot_dataset.py` to populate.")
        return 1

    sample_fits = sample_files[0]
    print(f"[1/5] Loading TESS Light Curve: {sample_fits.name}")
    lc = read_tess_light_curve(sample_fits)
    print(f"      Target: TIC {lc.provenance.target_id}, Sector: {lc.provenance.sector}")
    print(f"      Cadences: {len(lc.time_days)} points, baseline: {lc.time_days[-1] - lc.time_days[0]:.2f} days")

    # 2. Quality assessment and representation
    print("[2/5] Preparing quality mask and baseline normalization...")
    prepared = prepare_light_curve(lc)

    # 3. Two-pass detrending
    print("[3/5] Applying duration-aware two-pass biweight detrending...")
    dual_lc = two_pass_detrend(prepared, baseline_window_days=0.75)

    # 4. Blind BLS candidate search
    print("[4/5] Running Box Least-Squares (BLS) and Lomb-Scargle search...")
    detection = find_candidates(dual_lc, max_candidates=2, snr_threshold=6.0)
    print(f"      Detected {len(detection.candidates)} significant candidate(s).")

    for idx, cand in enumerate(detection.candidates, start=1):
        print(f"\n--- Candidate {idx}: {cand.candidate_id} ---")
        print(f"    Period: {cand.period_days:.5f} days, Epoch: {cand.epoch_days:.4f} BJD")
        print(f"    Duration: {cand.duration_hours:.2f} hours, Depth: {cand.depth_ppm:.1f} ppm")
        print(f"    BLS Detection SNR: {cand.detection_snr:.2f}")

        # Refinement with physical limb-darkening
        refined = refine_candidate(
            cand,
            dual_lc.prepared.time_days,
            dual_lc.detrended_flux,
            dual_lc.prepared.flux_error_normalized,
            crowdsap=float(lc.metadata.get("crowdsap", 1.0) or 1.0),
            teff=float(lc.metadata.get("teff", 5778.0) or 5778.0),
        )
        print(f"    Refined Depth: {refined.measured_depth_ppm:.1f} ppm (BIC: {refined.bic:.1f})")

        # Source vetting
        vetting = vet_source(dual_lc, refined)
        print(f"    Source Vetting Assessment: {vetting.assessment.value.upper()}")
        print(f"    Centroid Shift: {vetting.centroid_shift_pixels:.3f} px ({vetting.centroid_shift_arcsec:.1f}\", {vetting.centroid_shift_sigma:.1f} σ)")
        print(f"    Vetting Summary: {vetting.summary_reason}")

    print("\n[5/5] Quickstart analysis completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
