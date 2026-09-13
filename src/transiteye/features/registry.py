"""Ordered, explicit feature registry and leakage firewall."""

from __future__ import annotations

from collections.abc import Iterable

from transiteye.datasets.schemas import (
    FORBIDDEN_MODEL_INPUT_COLUMNS,
    assert_model_input_boundary,
)
from transiteye.features.schemas import ColumnRole, FeatureDefinition
from transiteye.serialization import content_hash


def _definitions(
    names: Iterable[tuple[str, str]], role: ColumnRole, group: str, dtype: str = "float64"
) -> tuple[FeatureDefinition, ...]:
    return tuple(
        FeatureDefinition(name, role, group, description, dtype) for name, description in names
    )


IDENTITY_COLUMNS = _definitions(
    (
        ("candidate_id", "Deterministic blind-BLS candidate identity."),
        ("object_id", "Normalized TIC object identity."),
        ("observation_group_id", "Processed observation-group identity."),
    ),
    ColumnRole.IDENTITY,
    "identity",
    "string",
)

PROVENANCE_COLUMNS = _definitions(
    (
        ("dataset_version", "Frozen source dataset version."),
        ("feature_version", "Deterministic feature artifact version."),
        ("preprocessing_config_hash", "Frozen preprocessing policy hash."),
        ("bls_config_hash", "Frozen blind-BLS policy hash."),
    ),
    ColumnRole.PROVENANCE,
    "provenance",
    "string",
)

TIME_FEATURES = _definitions(
    (
        ("td_global_median", "Median detrended flux."),
        ("td_global_std", "Population standard deviation of detrended flux."),
        ("td_global_mad", "Median absolute deviation of detrended flux."),
        ("td_global_robust_rms", "Gaussian-scaled MAD."),
        ("td_global_iqr", "Interquartile range."),
        ("td_global_skewness", "Standardized third central moment."),
        ("td_global_excess_kurtosis", "Excess standardized fourth moment."),
        ("td_global_p05", "Fifth detrended-flux percentile."),
        ("td_global_p95", "Ninety-fifth detrended-flux percentile."),
        ("td_global_robust_range", "P95 minus P05."),
        ("td_in_transit_count", "Observed cadences inside candidate windows."),
        ("td_out_transit_count", "Observed cadences outside candidate windows."),
        ("td_in_transit_median", "Median in candidate windows."),
        ("td_out_transit_median", "Median outside candidate windows."),
        ("td_observed_depth", "Out-of-window minus in-window median."),
        ("td_depth_significance", "Observed depth over robust out-of-window scatter."),
        ("td_in_out_scatter_ratio", "Robust in/out scatter ratio."),
        ("td_transit_occupancy", "Fraction of valid cadences in candidate windows."),
        ("td_observed_transit_windows", "Number of candidate windows with data."),
        ("td_per_transit_depth_mad", "MAD of per-window depth estimates."),
        ("td_transit_to_transit_scatter", "Standard deviation of per-window depths."),
        ("td_pre_post_flux_difference", "Median pre-window minus post-window flux."),
        ("td_ingress_egress_asymmetry", "Median first-half minus second-half transit flux."),
        ("td_odd_even_depth_difference", "Odd minus even candidate-window depth."),
        (
            "td_odd_even_depth_normalized_difference",
            "Odd/even depth difference normalized by magnitude.",
        ),
    ),
    ColumnRole.FEATURE,
    "time_domain",
)

BLS_FEATURES = _definitions(
    (
        ("bls_candidate_period", "Blind candidate period."),
        ("bls_candidate_duration", "Blind candidate duration."),
        ("bls_candidate_depth", "Blind BLS depth estimate."),
        ("bls_candidate_power", "Blind BLS detection statistic."),
        ("bls_candidate_rank", "Rank among retained independent candidates."),
        ("bls_duty_cycle", "Candidate duration divided by period."),
        ("bls_estimated_transit_count", "Candidate windows intersecting the baseline."),
        ("bls_period_baseline_ratio", "Candidate period divided by observing baseline."),
        ("bls_power_ratio_to_strongest", "Candidate/strongest retained BLS power."),
        ("bls_power_difference_from_strongest", "Strongest minus candidate BLS power."),
        ("bls_local_peak_prominence", "Power above local periodogram median."),
        ("bls_local_peak_width_days", "Half-prominence local peak width in period units."),
        ("bls_has_harmonic_relation", "Detector-derived harmonic-relation indicator."),
        ("bls_detector_harmonic_ratio", "Detector-derived period ratio to a stronger peak."),
        ("bls_effective_max_period_days", "Maximum period searched for this observation."),
        ("bls_observing_baseline_days", "Valid observational time baseline."),
    ),
    ColumnRole.FEATURE,
    "bls",
)

LS_FEATURES = _definitions(
    (
        ("ls_dominant_frequency_per_day", "Strongest Lomb-Scargle frequency."),
        ("ls_dominant_period_days", "Inverse of strongest Lomb-Scargle frequency."),
        ("ls_dominant_power", "Strongest Lomb-Scargle power."),
        ("ls_second_independent_power", "Second independent Lomb-Scargle peak power."),
        ("ls_primary_secondary_power_ratio", "Primary/secondary Lomb-Scargle power."),
        ("ls_spectral_entropy", "Normalized entropy of Lomb-Scargle power."),
        ("ls_power_concentration", "Strongest-power fraction."),
        ("ls_power_at_bls_frequency", "Lomb-Scargle power at candidate frequency."),
        ("ls_power_at_twice_bls_frequency", "Lomb-Scargle power at twice candidate frequency."),
        ("ls_power_at_half_bls_frequency", "Lomb-Scargle power at half candidate frequency."),
        ("ls_bls_to_global_power_ratio", "Candidate-frequency/global-maximum power."),
    ),
    ColumnRole.FEATURE,
    "lomb_scargle",
)

FFT_FEATURES = _definitions(
    (
        ("fft_usable", "One when a segment supports the FFT policy, otherwise zero."),
        ("fft_dominant_frequency_per_day", "Strongest non-DC FFT frequency."),
        ("fft_dominant_power", "Strongest non-DC FFT power."),
        ("fft_second_peak_power", "Second independent FFT-bin power."),
        ("fft_power_concentration", "Strongest non-DC power fraction."),
        ("fft_spectral_entropy", "Normalized entropy of non-DC FFT power."),
        ("fft_low_band_power_fraction", "Power fraction below low-band boundary."),
        ("fft_mid_band_power_fraction", "Power fraction in configured middle band."),
        ("fft_high_band_power_fraction", "Power fraction above middle-band boundary."),
        ("fft_power_at_bls_frequency", "FFT power nearest candidate frequency."),
        ("fft_power_at_twice_bls_frequency", "FFT power nearest twice candidate frequency."),
        ("fft_power_at_half_bls_frequency", "FFT power nearest half candidate frequency."),
        ("fft_bls_to_global_power_ratio", "Candidate-frequency/global-maximum FFT power."),
    ),
    ColumnRole.FEATURE,
    "fft",
)

LABEL_COLUMNS = _definitions(
    (
        ("gold_candidate_label", "Gold candidate label, outside model inputs."),
        ("gold_training_eligible", "Gold-training eligibility, outside model inputs."),
    ),
    ColumnRole.LABEL,
    "labels",
    "string",
)

EVALUATION_COLUMNS = _definitions(
    (
        ("matched_event_id", "Matched catalog/synthetic event, outside model inputs."),
        ("match_type", "Truth match type, outside model inputs."),
    ),
    ColumnRole.EVALUATION,
    "evaluation",
    "string",
)

LABEL_PROVENANCE_COLUMNS = _definitions(
    (
        ("dataset_role", "Dataset role, outside model inputs."),
        ("truth_source", "Truth provenance, outside model inputs."),
    ),
    ColumnRole.PROVENANCE,
    "label_provenance",
    "string",
)

FEATURE_REGISTRY = (
    IDENTITY_COLUMNS
    + PROVENANCE_COLUMNS
    + TIME_FEATURES
    + BLS_FEATURES
    + LS_FEATURES
    + FFT_FEATURES
    + LABEL_COLUMNS
    + EVALUATION_COLUMNS
    + LABEL_PROVENANCE_COLUMNS
)


def validate_registry(registry: tuple[FeatureDefinition, ...] = FEATURE_REGISTRY) -> None:
    """Reject duplicate columns and forbidden names registered as model features."""
    feature_names = [entry.name for entry in registry if entry.role is ColumnRole.FEATURE]
    assert_model_input_boundary(feature_names)
    names = [entry.name for entry in registry]
    if len(names) != len(set(names)):
        raise ValueError("Feature registry contains duplicate column names.")
    leaked = sorted(FORBIDDEN_MODEL_INPUT_COLUMNS.intersection(feature_names))
    if leaked:
        raise ValueError(f"Forbidden fields registered as model features: {leaked}")


def model_feature_names(
    registry: tuple[FeatureDefinition, ...] = FEATURE_REGISTRY,
) -> tuple[str, ...]:
    validate_registry(registry)
    return tuple(entry.name for entry in registry if entry.role is ColumnRole.FEATURE)


def feature_names_by_groups(groups: tuple[str, ...]) -> tuple[str, ...]:
    """Select model features by explicit registry group in canonical order."""
    known = {entry.group for entry in FEATURE_REGISTRY if entry.role is ColumnRole.FEATURE}
    if unknown := set(groups).difference(known):
        raise ValueError(f"Unknown feature registry groups: {sorted(unknown)}")
    return tuple(
        entry.name
        for entry in FEATURE_REGISTRY
        if entry.role is ColumnRole.FEATURE and entry.group in groups
    )


def registry_hash(registry: tuple[FeatureDefinition, ...] = FEATURE_REGISTRY) -> str:
    validate_registry(registry)
    return content_hash([entry.portable_dict() for entry in registry])


validate_registry()
