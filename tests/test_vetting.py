"""Tests for independent spatial source vetting (Tier 1 LC and Tier 2 TPF)."""

from __future__ import annotations

import numpy as np

from transiteye.characterization.fitting import RefinedCandidate
from transiteye.domain.models import Candidate, Provenance, SourceAssessment
from transiteye.io.tpf import TargetPixelData
from transiteye.preprocessing.detrend import DualLightCurve
from transiteye.preprocessing.representation import PreparedLightCurve
from transiteye.vetting.source import (
    run_tier1_source_vetting,
    run_tier2_source_vetting,
    vet_source,
)


def _make_vetting_fixture(
    crowdsap: float = 0.95,
    centroid_shift_sigma: float = 0.5,
) -> tuple[DualLightCurve, RefinedCandidate]:
    """Helper to construct light curve and candidate for vetting tests."""
    time = np.linspace(0.0, 20.0, 1000)
    period = 4.0
    epoch = 1.0
    dur_days = 0.2
    depth = 0.01

    flux = np.ones_like(time)
    phases = ((time - epoch + 0.5 * period) % period) / period - 0.5
    in_transit = np.abs(phases) <= (0.5 * dur_days / period)
    flux[in_transit] -= depth

    # Centroid columns
    col = np.full_like(time, 500.0)
    row = np.full_like(time, 500.0)

    if centroid_shift_sigma > 2.0:
        # Induce an in-transit centroid shift
        col[in_transit] += 0.8
        row[in_transit] += 0.8

    provenance = Provenance(
        source_uri="file:///test_vetting.fits",
        source_path=None,
        checksum_sha256=None,
        product_type="tess_light_curve",
        target_id="12345678",
        sector=1,
        time_system="TDB",
        time_reference="BJD",
    )

    prepared = PreparedLightCurve(
        time_days=time,
        flux_normalized=flux,
        flux_error_normalized=np.full_like(time, 0.0005),
        quality=np.zeros(len(time), dtype=np.int64),
        cadence_number=np.arange(len(time), dtype=np.int64),
        centroid_column=col,
        centroid_row=row,
        raw_cadence_mask=np.ones(len(time), dtype=bool),
        exclusion_flags=np.zeros(len(time), dtype=np.int64),
        segment_slices=((0, len(time)),),
        scatter_ptp_ppm=500.0,
        cdpp_1h_ppm=300.0,
        cdpp_2h_ppm=200.0,
        cdpp_6h_ppm=100.0,
        provenance=provenance,
        metadata={"crowdsap": str(crowdsap)},
    )

    dual_lc = DualLightCurve(
        prepared=prepared,
        detrended_flux=flux.copy(),
        trend_flux=np.ones_like(flux),
        in_transit_mask=in_transit,
        window_days=0.5,
        pass_number=2,
    )

    cand = Candidate(
        candidate_id="vet_01",
        period_days=period,
        epoch_days=epoch,
        duration_hours=dur_days * 24.0,
        depth_ppm=depth * 1e6,
        observed_event_count=5,
        detection_snr=25.0,
        search_significance=0.99,
    )

    refined = RefinedCandidate(
        candidate=cand,
        period_days=period,
        epoch_days=epoch,
        duration_hours=dur_days * 24.0,
        measured_depth_ppm=depth * 1e6,
        dilution_corrected_depth_ppm=depth * 1e6 / crowdsap,
        ingress_fraction=0.2,
        chi2_reduced=1.0,
        bic=1000.0,
        batman_delta_bic=10.0,
        phase_coverage=0.95,
        trapezoid_model_curve=np.ones_like(time),
        batman_model_curve=np.ones_like(time),
    )

    return dual_lc, refined


def test_tier1_target_consistent() -> None:
    """Verify Tier 1 confirms TARGET_CONSISTENT for uncrowded, stable star."""
    dual_lc, refined = _make_vetting_fixture(crowdsap=0.95, centroid_shift_sigma=0.5)
    evidence = run_tier1_source_vetting(dual_lc, refined)

    assert evidence.assessment == SourceAssessment.TARGET_CONSISTENT
    assert evidence.tier == 1
    assert evidence.crowdsap == 0.95
    assert evidence.centroid_shift_sigma < 3.0


def test_tier1_blend_suspected_due_to_centroid_shift() -> None:
    """Verify Tier 1 detects BLEND_SUSPECTED when centroid shifts significantly in transit."""
    dual_lc, refined = _make_vetting_fixture(crowdsap=0.95, centroid_shift_sigma=5.0)
    evidence = run_tier1_source_vetting(dual_lc, refined)

    assert evidence.assessment == SourceAssessment.BLEND_SUSPECTED
    assert evidence.centroid_shift_pixels > 0.5
    assert evidence.centroid_shift_sigma >= 3.5


def test_tier1_unresolved_in_crowded_field_without_pixels() -> None:
    """Verify Tier 1 yields UNRESOLVED when CROWDSAP < 0.85 without TPF pixel data."""
    dual_lc, refined = _make_vetting_fixture(crowdsap=0.65, centroid_shift_sigma=0.5)
    evidence = run_tier1_source_vetting(dual_lc, refined)

    assert evidence.assessment == SourceAssessment.UNRESOLVED
    assert evidence.crowdsap == 0.65


def test_tier2_target_consistent_with_tpf() -> None:
    """Verify Tier 2 difference image confirms TARGET_CONSISTENT when drop is on target."""
    dual_lc, refined = _make_vetting_fixture(crowdsap=0.65, centroid_shift_sigma=0.5)

    time = dual_lc.prepared.time_days
    # Create 9x9 pixel stamp
    rows, cols = 9, 9
    flux_cube = np.zeros((len(time), rows, cols), dtype=np.float64)

    # Put bright target star at center (4.0, 4.0)
    target_coord = (4.0, 4.0)
    y_g, x_g = np.mgrid[0:rows, 0:cols]
    r_sq = (x_g - 4.0) ** 2 + (y_g - 4.0) ** 2
    psf = 1000.0 * np.exp(-0.5 * r_sq / 1.2)  # Gaussian PSF

    # Add PSF to all frames
    flux_cube += psf[np.newaxis, :, :]

    # Add transit dip exactly at center
    period = 4.0
    epoch = 1.0
    dur_days = 0.2
    phases = ((time - epoch + 0.5 * period) % period) / period - 0.5
    in_transit = np.abs(phases) <= (0.5 * dur_days / period)

    flux_cube[in_transit, 4, 4] -= 50.0

    tpf_data = TargetPixelData(
        time_days=time,
        flux_cube=flux_cube,
        flux_err_cube=None,
        aperture_mask=r_sq <= 3.0,
        target_pixel_coord=target_coord,
        provenance=dual_lc.prepared.provenance,
        metadata={},
    )

    evidence = run_tier2_source_vetting(dual_lc, refined, tpf_data)

    assert evidence.tier == 2
    assert evidence.assessment == SourceAssessment.TARGET_CONSISTENT
    assert evidence.difference_image_offset_pixels is not None
    assert evidence.difference_image_offset_pixels < 0.5


def test_tier2_blend_suspected_with_offset_tpf() -> None:
    """Verify Tier 2 detects off-center difference image centroid (background blend)."""
    dual_lc, refined = _make_vetting_fixture(crowdsap=0.95, centroid_shift_sigma=0.5)

    time = dual_lc.prepared.time_days
    rows, cols = 9, 9
    flux_cube = np.zeros((len(time), rows, cols), dtype=np.float64)

    target_coord = (4.0, 4.0)
    y_g, x_g = np.mgrid[0:rows, 0:cols]
    r_sq = (x_g - 4.0) ** 2 + (y_g - 4.0) ** 2
    psf = 1000.0 * np.exp(-0.5 * r_sq / 1.2)
    flux_cube += psf[np.newaxis, :, :]

    # In-transit flux drop happens at off-center pixel (7, 7) - 4.2 pixels away!
    period = 4.0
    epoch = 1.0
    dur_days = 0.2
    phases = ((time - epoch + 0.5 * period) % period) / period - 0.5
    in_transit = np.abs(phases) <= (0.5 * dur_days / period)

    flux_cube[in_transit, 7, 7] -= 80.0

    tpf_data = TargetPixelData(
        time_days=time,
        flux_cube=flux_cube,
        flux_err_cube=None,
        aperture_mask=r_sq <= 3.0,
        target_pixel_coord=target_coord,
        provenance=dual_lc.prepared.provenance,
        metadata={},
    )

    evidence = vet_source(dual_lc, refined, tpf_data)

    assert evidence.tier == 2
    assert evidence.assessment == SourceAssessment.BLEND_SUSPECTED
    assert evidence.difference_image_offset_pixels is not None
    assert evidence.difference_image_offset_pixels > 1.0
