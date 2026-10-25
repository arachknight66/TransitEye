"""Audited feature registry definitions and leakage validation."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

import numpy as np


class FeatureFamily(StrEnum):
    """Scientific families of candidate features."""

    MORPHOLOGY = "morphology"
    CONSISTENCY = "consistency"
    ECLIPSE = "eclipse"
    VARIABILITY = "variability"
    NOISE_QUALITY = "noise_quality"
    SOURCE_EVIDENCE = "source_evidence"


@dataclass(frozen=True, slots=True)
class FeatureDefinition:
    """Specification of an audited feature."""

    name: str
    family: FeatureFamily
    description: str
    min_value: float
    max_value: float
    default_value: float
    is_source_evidence: bool = False


# Prohibited terms that must NEVER appear as model features or feature dictionary keys.
PROHIBITED_LEAKAGE_TERMS: frozenset[str] = frozenset(
    [
        "tic",
        "tic_id",
        "target_id",
        "toi",
        "epic",
        "kic",
        "label",
        "ground_truth",
        "disposition",
        "class",
        "morphology_class",
        "split",
        "partition",
        "dataset_split",
        "source_group_id",
        "parent_source_group_id",
        "sector",
        "camera",
        "ccd",
        "ra",
        "dec",
        "magnitude",
        "tmag",
        "provenance",
        "filename",
        "source_uri",
        "source_path",
    ]
)

FEATURE_DEFINITIONS: tuple[FeatureDefinition, ...] = (
    # Family 1: Morphology (5)
    FeatureDefinition(
        name="v_shape_ratio",
        family=FeatureFamily.MORPHOLOGY,
        description="Fraction of transit half-duration spent in ingress/egress (1 for V-shape, 0 for pure flat bottom).",
        min_value=0.0,
        max_value=1.0,
        default_value=0.5,
    ),
    FeatureDefinition(
        name="duration_to_period_ratio",
        family=FeatureFamily.MORPHOLOGY,
        description="Ratio of total transit duration to candidate orbital period (T_dur / P).",
        min_value=0.0,
        max_value=1.0,
        default_value=0.02,
    ),
    FeatureDefinition(
        name="boxiness_index",
        family=FeatureFamily.MORPHOLOGY,
        description="Ratio of flat-bottom duration to total duration (1 - ingress_fraction).",
        min_value=0.0,
        max_value=1.0,
        default_value=0.5,
    ),
    FeatureDefinition(
        name="asymmetry_score",
        family=FeatureFamily.MORPHOLOGY,
        description="Normalized difference in ingress versus egress mean flux relative to transit depth.",
        min_value=0.0,
        max_value=10.0,
        default_value=0.0,
    ),
    FeatureDefinition(
        name="batman_vs_trapezoid_delta_bic",
        family=FeatureFamily.MORPHOLOGY,
        description="BIC difference (BIC_trapezoid - BIC_batman). Positive indicates physical limb-darkened transit fits better.",
        min_value=-1000.0,
        max_value=1000.0,
        default_value=0.0,
    ),
    # Family 2: Consistency (5)
    FeatureDefinition(
        name="odd_even_depth_sigma",
        family=FeatureFamily.CONSISTENCY,
        description="Statistical significance of transit depth difference between odd and even cycles.",
        min_value=0.0,
        max_value=100.0,
        default_value=0.0,
    ),
    FeatureDefinition(
        name="per_transit_depth_mad_ratio",
        family=FeatureFamily.CONSISTENCY,
        description="Normalized Median Absolute Deviation (MAD) of individual transit event depths relative to median depth.",
        min_value=0.0,
        max_value=10.0,
        default_value=0.1,
    ),
    FeatureDefinition(
        name="phase_coverage_fraction",
        family=FeatureFamily.CONSISTENCY,
        description="Fraction of orbital phase bins containing at least one valid observation.",
        min_value=0.0,
        max_value=1.0,
        default_value=1.0,
    ),
    FeatureDefinition(
        name="in_transit_duty_cycle",
        family=FeatureFamily.CONSISTENCY,
        description="Fraction of expected in-transit event windows containing valid observation points.",
        min_value=0.0,
        max_value=1.0,
        default_value=1.0,
    ),
    FeatureDefinition(
        name="observed_event_count",
        family=FeatureFamily.CONSISTENCY,
        description="Count of distinct transit events with sufficient in-transit data points.",
        min_value=0.0,
        max_value=1000.0,
        default_value=1.0,
    ),
    # Family 3: Eclipse (4)
    FeatureDefinition(
        name="max_secondary_snr",
        family=FeatureFamily.ECLIPSE,
        description="Maximum signal-to-noise ratio of secondary dip at out-of-transit phases.",
        min_value=0.0,
        max_value=1000.0,
        default_value=0.0,
    ),
    FeatureDefinition(
        name="secondary_to_primary_depth_ratio",
        family=FeatureFamily.ECLIPSE,
        description="Ratio of secondary eclipse depth (near phase 0.5) to primary transit depth.",
        min_value=0.0,
        max_value=10.0,
        default_value=0.0,
    ),
    FeatureDefinition(
        name="secondary_phase_offset_from_half",
        family=FeatureFamily.ECLIPSE,
        description="Absolute deviation of candidate secondary eclipse phase from circular orbit phase 0.5.",
        min_value=0.0,
        max_value=0.5,
        default_value=0.0,
    ),
    FeatureDefinition(
        name="ellipsoidal_amplitude_ratio",
        family=FeatureFamily.ECLIPSE,
        description="Amplitude of out-of-transit cos(4*pi*phase) ellipsoidal variation relative to transit depth.",
        min_value=0.0,
        max_value=5.0,
        default_value=0.0,
    ),
    # Family 4: Variability (4)
    FeatureDefinition(
        name="ls_max_power",
        family=FeatureFamily.VARIABILITY,
        description="Maximum normalized Lomb-Scargle power across the stellar variability periodogram.",
        min_value=0.0,
        max_value=1.0,
        default_value=0.0,
    ),
    FeatureDefinition(
        name="ls_to_bls_period_harmonic_distance",
        family=FeatureFamily.VARIABILITY,
        description="Minimum fractional harmonic distance between Lomb-Scargle best period and candidate transit period.",
        min_value=0.0,
        max_value=1.0,
        default_value=1.0,
    ),
    FeatureDefinition(
        name="out_of_transit_rms_to_depth_ratio",
        family=FeatureFamily.VARIABILITY,
        description="Ratio of detrended out-of-transit RMS scatter to transit depth.",
        min_value=0.0,
        max_value=100.0,
        default_value=1.0,
    ),
    FeatureDefinition(
        name="autocorrelation_at_period",
        family=FeatureFamily.VARIABILITY,
        description="Normalized light curve autocorrelation evaluated at lag equal to candidate period.",
        min_value=-1.0,
        max_value=1.0,
        default_value=0.0,
    ),
    # Family 5: Noise/Quality (4)
    FeatureDefinition(
        name="red_noise_beta_factor",
        family=FeatureFamily.NOISE_QUALITY,
        description="Pont et al. beta factor measuring correlated red noise relative to pure white noise expectation.",
        min_value=0.5,
        max_value=20.0,
        default_value=1.0,
    ),
    FeatureDefinition(
        name="detection_snr",
        family=FeatureFamily.NOISE_QUALITY,
        description="Candidate signal-to-noise ratio accounting for correlated red noise.",
        min_value=0.0,
        max_value=10000.0,
        default_value=5.0,
    ),
    FeatureDefinition(
        name="in_transit_flagged_fraction",
        family=FeatureFamily.NOISE_QUALITY,
        description="Fraction of candidate in-transit raw cadences having non-zero quality flags.",
        min_value=0.0,
        max_value=1.0,
        default_value=0.0,
    ),
    FeatureDefinition(
        name="gap_proximity_fraction",
        family=FeatureFamily.NOISE_QUALITY,
        description="Fraction of candidate in-transit cadences located within 1 hour of an observation gap.",
        min_value=0.0,
        max_value=1.0,
        default_value=0.0,
    ),
    # Family 6: Source Evidence (3)
    FeatureDefinition(
        name="centroid_shift_sigma",
        family=FeatureFamily.SOURCE_EVIDENCE,
        description="Statistical significance of in-transit versus out-of-transit flux centroid shift.",
        min_value=0.0,
        max_value=100.0,
        default_value=0.0,
        is_source_evidence=True,
    ),
    FeatureDefinition(
        name="sap_to_pdcsap_depth_ratio",
        family=FeatureFamily.SOURCE_EVIDENCE,
        description="Ratio of transit depth measured on minimally processed / SAP flux to detrended PDCSAP depth.",
        min_value=0.0,
        max_value=10.0,
        default_value=1.0,
        is_source_evidence=True,
    ),
    FeatureDefinition(
        name="crowdsap_value",
        family=FeatureFamily.SOURCE_EVIDENCE,
        description="Delivered CROWDSAP keyword measuring target flux fraction in the photometric aperture.",
        min_value=0.0,
        max_value=1.0,
        default_value=1.0,
        is_source_evidence=True,
    ),
)

FEATURE_MAP: dict[str, FeatureDefinition] = {f.name: f for f in FEATURE_DEFINITIONS}
FEATURE_NAMES: tuple[str, ...] = tuple(f.name for f in FEATURE_DEFINITIONS)


def get_feature_definition(name: str) -> FeatureDefinition:
    """Retrieve the definition for a given feature name."""
    if name not in FEATURE_MAP:
        raise KeyError(f"Feature '{name}' is not in the audited feature registry.")
    return FEATURE_MAP[name]


def audit_feature_leakage(data: Mapping[str, object]) -> None:
    """Verify that no prohibited label, target, or partition leakage exists in feature data."""
    for key in data:
        lowered = key.lower()
        if lowered in PROHIBITED_LEAKAGE_TERMS:
            raise ValueError(
                f"Prohibited leakage term '{lowered}' detected in feature key '{key}'."
            )
        tokens = set(lowered.split("_"))
        matched = tokens & PROHIBITED_LEAKAGE_TERMS
        if matched:
            raise ValueError(
                f"Prohibited leakage term '{next(iter(matched))}' detected in feature key '{key}'."
            )


def validate_feature_vector(features: Mapping[str, float]) -> dict[str, float]:
    """Validate, sanitize, and impute a feature vector according to registry specs."""
    audit_feature_leakage(features)
    sanitized: dict[str, float] = {}

    for f_def in FEATURE_DEFINITIONS:
        val = features.get(f_def.name)
        if val is None or math.isnan(val) or math.isinf(val):
            sanitized[f_def.name] = f_def.default_value
        else:
            # Clip within physical valid range
            clipped = float(np.clip(val, f_def.min_value, f_def.max_value))
            sanitized[f_def.name] = clipped

    return sanitized
