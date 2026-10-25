"""Audited feature registry and extractors for signal classification and vetting."""

from transiteye.features.extractors import extract_features
from transiteye.features.registry import (
    FEATURE_DEFINITIONS,
    FEATURE_NAMES,
    FeatureDefinition,
    FeatureFamily,
    get_feature_definition,
    validate_feature_vector,
)

__all__ = [
    "FEATURE_DEFINITIONS",
    "FEATURE_NAMES",
    "FeatureDefinition",
    "FeatureFamily",
    "extract_features",
    "get_feature_definition",
    "validate_feature_vector",
]
