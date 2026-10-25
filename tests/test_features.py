"""Tests for audited feature definitions, extractors, and leakage prevention."""

from __future__ import annotations

import numpy as np
import pytest

from transiteye.characterization.fitting import RefinedCandidate
from transiteye.detection.variability import VariabilitySummary
from transiteye.domain.models import Candidate, LightCurve, Provenance
from transiteye.features.extractors import extract_features
from transiteye.features.registry import (
    FEATURE_DEFINITIONS,
    FEATURE_NAMES,
    FeatureFamily,
    audit_feature_leakage,
    get_feature_definition,
    validate_feature_vector,
)
from transiteye.preprocessing.detrend import DualLightCurve
from transiteye.preprocessing.representation import PreparedLightCurve


def test_feature_registry_schema_and_completeness() -> None:
    """Verify registry has all 25 features across the 6 families with valid bounds."""
    assert len(FEATURE_DEFINITIONS) == 25
    assert len(FEATURE_NAMES) == 25

    families = {f.family for f in FEATURE_DEFINITIONS}
    assert families == {
        FeatureFamily.MORPHOLOGY,
        FeatureFamily.CONSISTENCY,
        FeatureFamily.ECLIPSE,
        FeatureFamily.VARIABILITY,
        FeatureFamily.NOISE_QUALITY,
        FeatureFamily.SOURCE_EVIDENCE,
    }

    for f_def in FEATURE_DEFINITIONS:
        assert f_def.min_value <= f_def.default_value <= f_def.max_value
        retrieved = get_feature_definition(f_def.name)
        assert retrieved == f_def


def test_leakage_audit_detects_prohibited_terms() -> None:
    """Verify audit_feature_leakage rejects label, target, and split metadata."""
    # Clean dictionary should pass
    clean_dict = {"v_shape_ratio": 0.5, "detection_snr": 12.0}
    audit_feature_leakage(clean_dict)

    # Prohibited keys should raise ValueError
    prohibited_keys = [
        "target_id",
        "tic_id",
        "label",
        "dataset_split",
        "sector",
        "ground_truth_class",
        "split_name",
        "parent_source_group_id",
    ]
    for key in prohibited_keys:
        with pytest.raises(ValueError, match="Prohibited leakage term"):
            audit_feature_leakage({key: 1.0})


def test_validate_feature_vector_imputes_and_clips() -> None:
    """Verify missing, NaN, Inf, and out-of-bounds features are sanitized."""
    raw_dict = {
        "v_shape_ratio": float("nan"),
        "detection_snr": float("inf"),
        "duration_to_period_ratio": 5.0,  # exceeds max 1.0
        "boxiness_index": -0.5,  # below min 0.0
    }
    sanitized = validate_feature_vector(raw_dict)
    assert len(sanitized) == 25

    # NaN imputed to default
    assert sanitized["v_shape_ratio"] == get_feature_definition("v_shape_ratio").default_value
    # Inf imputed to default
    assert sanitized["detection_snr"] == get_feature_definition("detection_snr").default_value
    # Clipped to max
    assert sanitized["duration_to_period_ratio"] == 1.0
    # Clipped to min
    assert sanitized["boxiness_index"] == 0.0


def _build_test_light_curve_and_candidate(
    transit_type: str = "transit",
) -> tuple[DualLightCurve, RefinedCandidate, VariabilitySummary, LightCurve]:
    """Helper to synthesize test data for feature extraction."""
    rng = np.random.default_rng(42)
    n_pts = 1000
    time = np.linspace(0.0, 20.0, n_pts)
    period = 4.0
    epoch = 1.0
    dur_days = 0.2  # 4.8 hours
    depth = 0.01  # 10,000 ppm

    flux = np.ones_like(time)
    phases = ((time - epoch + 0.5 * period) % period) / period - 0.5

    if transit_type == "transit":
        # Flat-bottomed U-shape transit
        in_transit = np.abs(phases) <= (0.5 * dur_days / period)
        flux[in_transit] -= depth
    elif transit_type == "odd_even_eb":
        # Eclipsing binary with alternating depths
        cycle = np.round((time - epoch) / period).astype(int)
        in_transit = np.abs(phases) <= (0.5 * dur_days / period)
        flux[in_transit & (cycle % 2 == 0)] -= depth
        flux[in_transit & (cycle % 2 != 0)] -= 0.5 * depth

    # Add small Gaussian noise
    flux += rng.normal(0.0, 0.0005, size=len(time))
    detrended_flux = flux.copy()
    trend_flux = np.ones_like(flux)

    provenance = Provenance(
        source_uri="file:///test.fits",
        source_path=None,
        checksum_sha256=None,
        product_type="tess_light_curve",
        target_id="12345678",
        sector=1,
        time_system="TDB",
        time_reference="BJD",
    )

    raw_lc = LightCurve(
        time_days=time,
        flux=flux,
        flux_error=np.full_like(time, 0.0005),
        quality=np.zeros(len(time), dtype=np.int64),
        cadence_number=np.arange(len(time), dtype=np.int64),
        centroid_column=np.full_like(time, 500.0) + rng.normal(0, 0.01, len(time)),
        centroid_row=np.full_like(time, 500.0) + rng.normal(0, 0.01, len(time)),
        flux_kind="PDCSAP_FLUX",
        flux_unit="e-/s",
        provenance=provenance,
        metadata={"crowdsap": "0.95"},
    )

    prepared = PreparedLightCurve(
        time_days=time,
        flux_normalized=flux,
        flux_error_normalized=np.full_like(time, 0.0005),
        quality=raw_lc.quality,
        cadence_number=raw_lc.cadence_number,
        centroid_column=raw_lc.centroid_column,
        centroid_row=raw_lc.centroid_row,
        raw_cadence_mask=np.ones(len(time), dtype=bool),
        exclusion_flags=np.zeros(len(time), dtype=np.int64),
        segment_slices=((0, len(time)),),
        scatter_ptp_ppm=500.0,
        cdpp_1h_ppm=300.0,
        cdpp_2h_ppm=200.0,
        cdpp_6h_ppm=100.0,
        provenance=provenance,
        metadata={"crowdsap": "0.95"},
    )

    dual_lc = DualLightCurve(
        prepared=prepared,
        detrended_flux=detrended_flux,
        trend_flux=trend_flux,
        in_transit_mask=np.abs(phases) <= (0.5 * dur_days / period),
        window_days=0.5,
        pass_number=2,
    )

    cand = Candidate(
        candidate_id="test_01",
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
        dilution_corrected_depth_ppm=depth * 1e6 / 0.95,
        ingress_fraction=0.15,
        chi2_reduced=1.05,
        bic=1020.0,
        batman_delta_bic=15.0,
        phase_coverage=0.98,
        trapezoid_model_curve=np.ones_like(time),
        batman_model_curve=np.ones_like(time),
    )

    var_summary = VariabilitySummary(
        best_period_days=10.0,
        peak_power=0.05,
        false_alarm_prob=0.8,
        periods=np.linspace(0.5, 10.0, 100),
        powers=np.full(100, 0.05),
    )

    return dual_lc, refined, var_summary, raw_lc


def test_extract_features_transit_like() -> None:
    """Verify feature extraction on a clean planetary transit signal."""
    dual_lc, refined, var_summary, raw_lc = _build_test_light_curve_and_candidate("transit")
    features = extract_features(dual_lc, refined, var_summary, raw_lc)

    assert len(features) == 25
    # Verify no NaN or Inf
    for name, val in features.items():
        assert not np.isnan(val), f"Feature {name} is NaN"
        assert not np.isinf(val), f"Feature {name} is Inf"

    # Verify physical expectations for clean transit
    assert features["boxiness_index"] > 0.8
    assert features["v_shape_ratio"] < 0.2
    assert features["odd_even_depth_sigma"] < 3.0  # odd and even transits have equal depth
    assert features["crowdsap_value"] == pytest.approx(0.95, rel=1e-3)
    assert features["batman_vs_trapezoid_delta_bic"] == 15.0
    assert features["centroid_shift_sigma"] < 3.0  # no centroid shift


def test_extract_features_odd_even_eb() -> None:
    """Verify feature extraction on an alternating-depth eclipsing binary."""
    dual_lc, refined, var_summary, raw_lc = _build_test_light_curve_and_candidate(
        "odd_even_eb"
    )
    features = extract_features(dual_lc, refined, var_summary, raw_lc)

    assert len(features) == 25
    # Should detect strong odd-even depth difference
    assert features["odd_even_depth_sigma"] > 5.0
