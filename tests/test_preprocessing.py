"""Tests for quality assessment, cadence filtering, gap segmentation, and normalization."""

from __future__ import annotations

import numpy as np
import pytest

from transiteye.domain.models import LightCurve, Provenance
from transiteye.preprocessing.quality import (
    DEFAULT_QUALITY_MASK,
    KEPLER_DEFAULT_QUALITY_MASK,
    ExclusionReason,
    assess_quality,
    quality_mask_for_mission,
)
from transiteye.preprocessing.representation import (
    compute_cdpp_proxy,
    compute_robust_ptp_scatter,
    compute_segment_slices,
    prepare_light_curve,
)


def _make_light_curve(
    time_days: np.ndarray,
    flux: np.ndarray,
    flux_error: np.ndarray | None = None,
    quality: np.ndarray | None = None,
) -> LightCurve:
    length = len(time_days)
    provenance = Provenance(
        source_uri="file:///test/lc.fits",
        source_path="/test/lc.fits",
        checksum_sha256="0" * 64,
        product_type="tess_light_curve",
        target_id="123456",
        sector=1,
        time_system="TDB",
        time_reference="2457000",
    )
    return LightCurve(
        time_days=time_days,
        flux=flux,
        flux_error=flux_error if flux_error is not None else np.ones(length) * 0.01,
        quality=quality if quality is not None else np.zeros(length, dtype=np.int64),
        cadence_number=np.arange(length, dtype=np.int64),
        centroid_column=np.zeros(length),
        centroid_row=np.zeros(length),
        flux_kind="PDCSAP_FLUX",
        flux_unit="e-/s",
        provenance=provenance,
    )


def test_assess_quality_flags_reasons_accurately() -> None:
    time_days = np.array([1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0])
    flux = np.array([100.0, 100.0, -5.0, np.nan, 100.0, 100.0, 500.0])  # 500 is positive outlier
    flux_error = np.array([1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0])  # index 4 has negative error
    quality = np.array([0, 0, 0, 0, 0, DEFAULT_QUALITY_MASK, 0])  # index 5 has TESS quality bit

    lc = _make_light_curve(time_days, flux, flux_error=flux_error, quality=quality)
    reasons = assess_quality(lc, quality_mask=DEFAULT_QUALITY_MASK, reject_positive_outliers_sigma=3.0)

    assert reasons[0] == ExclusionReason.VALID
    assert bool(reasons[1] & ExclusionReason.NAN_OR_INF)
    assert bool(reasons[2] & ExclusionReason.NON_POSITIVE_FLUX)
    assert bool(reasons[3] & ExclusionReason.NAN_OR_INF)
    assert bool(reasons[4] & ExclusionReason.INVALID_ERROR)
    assert bool(reasons[5] & ExclusionReason.TESS_QUALITY_FLAG)


def test_negative_dips_are_never_filtered_as_outliers() -> None:
    # 200 normal points around 100.0
    rng = np.random.default_rng(42)
    time_days = np.linspace(1.0, 10.0, 200)
    flux = 100.0 + rng.normal(0, 0.1, size=200)

    # Inject a deep transit dip of 5% (well beyond 10 sigma negative)
    flux[50:55] = 95.0

    lc = _make_light_curve(time_days, flux)
    reasons = assess_quality(lc, reject_positive_outliers_sigma=3.0)

    # In-dip cadences MUST remain VALID
    assert np.all(reasons[50:55] == ExclusionReason.VALID)


def test_mission_quality_masks_keep_informational_kepler_flags() -> None:
    assert quality_mask_for_mission("TESS") == DEFAULT_QUALITY_MASK
    assert quality_mask_for_mission("Kepler") == KEPLER_DEFAULT_QUALITY_MASK
    assert quality_mask_for_mission("Kepler") != DEFAULT_QUALITY_MASK
    with pytest.raises(ValueError, match="Unsupported mission"):
        quality_mask_for_mission("K2")

    # Kepler aperture cosmic ray correction is informational; manual exclusion is not.
    lc = _make_light_curve(
        np.array([1.0, 2.0, 3.0]),
        np.array([100.0, 100.0, 100.0]),
        quality=np.array([0, 128, 256], dtype=np.int64),
    )
    reasons = assess_quality(
        lc, quality_mask=quality_mask_for_mission("Kepler"),
        reject_positive_outliers_sigma=None,
    )
    assert reasons[0] == ExclusionReason.VALID
    assert reasons[1] == ExclusionReason.VALID
    assert reasons[2] & ExclusionReason.QUALITY_FLAG


def test_gap_segmentation() -> None:
    # Two clusters separated by a 1-day gap (> 6 hours)
    t1 = np.linspace(1.0, 5.0, 50)
    t2 = np.linspace(6.5, 10.5, 50)
    time_days = np.concatenate([t1, t2])

    slices = compute_segment_slices(time_days, gap_threshold_days=0.25)
    assert len(slices) == 2
    assert slices[0] == (0, 50)
    assert slices[1] == (50, 100)


def test_prepare_light_curve_normalizes_per_segment() -> None:
    t1 = np.linspace(1.0, 5.0, 50)
    t2 = np.linspace(6.5, 10.5, 50)
    time_days = np.concatenate([t1, t2])

    # Segment 1 has median 1000, Segment 2 has median 500
    flux1 = np.full(50, 1000.0)
    flux2 = np.full(50, 500.0)
    flux = np.concatenate([flux1, flux2])

    lc = _make_light_curve(time_days, flux)
    prepared = prepare_light_curve(lc, gap_threshold_hours=6.0)

    assert len(prepared.segment_slices) == 2
    # Both segments must be normalized to median ~1.0
    np.testing.assert_allclose(np.median(prepared.flux_normalized[:50]), 1.0, atol=1e-7)
    np.testing.assert_allclose(np.median(prepared.flux_normalized[50:]), 1.0, atol=1e-7)

    assert prepared.scatter_ptp_ppm >= 0.0
    assert prepared.cdpp_1h_ppm >= 0.0
    assert prepared.cdpp_2h_ppm >= 0.0
    assert prepared.cdpp_6h_ppm >= 0.0
    assert prepared.metadata["segment_count"] == 2
    assert prepared.metadata["valid_cadences"] == 100

    # Test direct metric computations
    direct_ptp = compute_robust_ptp_scatter(prepared.flux_normalized, prepared.segment_slices)
    assert direct_ptp == prepared.scatter_ptp_ppm
    direct_cdpp = compute_cdpp_proxy(
        prepared.flux_normalized, prepared.time_days, prepared.segment_slices, duration_hours=1.0
    )
    assert direct_cdpp == prepared.cdpp_1h_ppm



def test_prepare_light_curve_rejects_empty_valid_set() -> None:
    time_days = np.array([1.0, 2.0])
    flux = np.array([np.nan, np.nan])
    lc = _make_light_curve(time_days, flux)

    with pytest.raises(ValueError, match="No valid cadences"):
        prepare_light_curve(lc)
