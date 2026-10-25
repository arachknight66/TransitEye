"""Morphology classification, confidence calibration, and abstention models."""

from transiteye.classification.calibration import (
    compute_expected_calibration_error,
    compute_multiclass_brier_score,
    fit_platt_calibrator,
    tune_abstention_thresholds,
)
from transiteye.classification.models import (
    MORPHOLOGY_FEATURE_NAMES,
    TARGET_MORPHOLOGY_CLASSES,
    ModelBundle,
    ModelMetadata,
)
from transiteye.classification.train import (
    evaluate_cv_model,
    get_default_morphology_model,
    train_and_select_classifier,
)

__all__ = [
    "MORPHOLOGY_FEATURE_NAMES",
    "TARGET_MORPHOLOGY_CLASSES",
    "ModelBundle",
    "ModelMetadata",
    "compute_expected_calibration_error",
    "compute_multiclass_brier_score",
    "evaluate_cv_model",
    "fit_platt_calibrator",
    "get_default_morphology_model",
    "train_and_select_classifier",
    "tune_abstention_thresholds",
]
