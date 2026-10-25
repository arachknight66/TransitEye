"""Tests for morphology classifier training, calibration, abstention, and serialization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from transiteye.classification.calibration import (
    compute_expected_calibration_error,
    compute_multiclass_brier_score,
    tune_abstention_thresholds,
)
from transiteye.classification.models import (
    MORPHOLOGY_FEATURE_NAMES,
    TARGET_MORPHOLOGY_CLASSES,
    ModelBundle,
)
from transiteye.classification.train import train_and_select_classifier
from transiteye.domain.models import MorphologyClass


@pytest.fixture
def synthetic_training_cohort() -> tuple[
    np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Synthesize grouped features for 4 classes to test training and calibration."""
    rng = np.random.default_rng(42)
    n_features = len(MORPHOLOGY_FEATURE_NAMES)
    classes = list(TARGET_MORPHOLOGY_CLASSES)

    # 40 training sources, 10 per class
    # 8 calibration sources, 2 per class
    # 8 selection sources, 2 per class

    def generate_partition(n_per_class: int, prefix: str):
        x_list: list[np.ndarray] = []
        y_list: list[int] = []
        groups_list: list[str] = []

        for c_idx, _c_name in enumerate(classes):
            for s_idx in range(n_per_class):
                group_id = f"{prefix}_{c_idx}_{s_idx}"
                # Generate 2 observations per source group
                for _obs in range(2):
                    feats = rng.normal(0.0, 1.0, n_features)
                    # Add class-specific signature
                    if c_idx == 0:  # transit_like: low v_shape, high boxiness
                        feats[0] = rng.uniform(0.05, 0.2)  # v_shape_ratio
                        feats[2] = rng.uniform(0.8, 0.95)  # boxiness_index
                        feats[5] = rng.uniform(0.0, 1.0)   # odd_even_sigma
                    elif c_idx == 1:  # eclipse_like: high odd_even_sigma or v_shape
                        feats[0] = rng.uniform(0.8, 1.0)
                        feats[5] = rng.uniform(8.0, 20.0)
                    elif c_idx == 2:  # stellar_variability: high ls_max_power
                        feats[14] = rng.uniform(0.7, 0.95)
                    else:  # other_artifact: high red_noise or flagged
                        feats[18] = rng.uniform(5.0, 15.0)

                    x_list.append(feats)
                    y_list.append(c_idx)
                    groups_list.append(group_id)

        return np.array(x_list), np.array(y_list), groups_list

    x_tr, y_tr, g_tr = generate_partition(10, "train")
    x_cal, y_cal, _ = generate_partition(2, "cal")
    x_sel, y_sel, _ = generate_partition(2, "sel")

    return x_tr, y_tr, g_tr, x_cal, y_cal, x_sel, y_sel


def test_train_and_select_classifier(synthetic_training_cohort) -> None:
    """Verify classifier training, model selection, calibration, and abstention tuning."""
    x_tr, y_tr, g_tr, x_cal, y_cal, x_sel, y_sel = synthetic_training_cohort

    bundle, comparison = train_and_select_classifier(
        x_train=x_tr,
        y_train=y_tr,
        groups_train=g_tr,
        x_cal=x_cal,
        y_cal=y_cal,
        x_sel=x_sel,
        y_sel=y_sel,
        feature_names=MORPHOLOGY_FEATURE_NAMES,
    )

    assert isinstance(bundle, ModelBundle)
    assert bundle.metadata.cv_macro_f1 > 0.6
    assert bundle.metadata.estimator_name in comparison
    assert bundle.calibrator is not None

    # Test calibration metrics
    probs_cal = bundle.calibrator.predict_proba(x_cal)
    brier = compute_multiclass_brier_score(y_cal, probs_cal, len(TARGET_MORPHOLOGY_CLASSES))
    assert 0.0 <= brier <= 1.0
    ece = compute_expected_calibration_error(y_cal, probs_cal)
    assert 0.0 <= ece <= 1.0


def test_model_bundle_serialization_roundtrip(
    synthetic_training_cohort,
    tmp_path: Path,
) -> None:
    """Verify saving and loading model bundle preserves weights and predictions."""
    x_tr, y_tr, g_tr, x_cal, y_cal, x_sel, y_sel = synthetic_training_cohort

    bundle, _ = train_and_select_classifier(
        x_train=x_tr,
        y_train=y_tr,
        groups_train=g_tr,
        x_cal=x_cal,
        y_cal=y_cal,
        x_sel=x_sel,
        y_sel=y_sel,
    )

    save_dir = tmp_path / "model_bundle"
    bundle.save(save_dir)

    assert (save_dir / "model_metadata.json").is_file()
    assert (save_dir / "estimator.joblib").is_file()
    assert (save_dir / "calibrator.joblib").is_file()

    loaded = ModelBundle.load(save_dir)
    assert loaded.metadata.model_id == bundle.metadata.model_id
    assert loaded.metadata.estimator_name == bundle.metadata.estimator_name

    # Predictions before and after loading must be identical
    sample_features = {
        name: float(val)
        for name, val in zip(MORPHOLOGY_FEATURE_NAMES, x_cal[0], strict=True)
    }
    pred_orig = bundle.predict(sample_features)
    pred_loaded = loaded.predict(sample_features)

    assert pred_orig.predicted_class == pred_loaded.predicted_class
    for cls in TARGET_MORPHOLOGY_CLASSES:
        assert pred_orig.probabilities[cls] == pytest.approx(
            pred_loaded.probabilities[cls], abs=1e-6
        )


def test_abstention_and_ood_guardrails(synthetic_training_cohort) -> None:
    """Verify that ambiguous or extreme OOD features trigger abstention."""
    x_tr, y_tr, g_tr, x_cal, y_cal, x_sel, y_sel = synthetic_training_cohort

    bundle, _ = train_and_select_classifier(
        x_train=x_tr,
        y_train=y_tr,
        groups_train=g_tr,
        x_cal=x_cal,
        y_cal=y_cal,
        x_sel=x_sel,
        y_sel=y_sel,
    )

    # 1. Clean transit features -> should predict TRANSIT_LIKE with no abstention
    clean_transit_features = {
        name: float(bundle.metadata.imputation_medians[name])
        for name in MORPHOLOGY_FEATURE_NAMES
    }
    clean_transit_features["v_shape_ratio"] = 0.1
    clean_transit_features["boxiness_index"] = 0.9
    clean_transit_features["odd_even_depth_sigma"] = 0.2
    clean_transit_features["detection_snr"] = 30.0

    pred_clean = bundle.predict(clean_transit_features)
    assert pred_clean.predicted_class == MorphologyClass.TRANSIT_LIKE
    assert pred_clean.abstained is False

    # 2. Extreme Out-of-Distribution feature value -> triggers OOD abstention
    ood_features = dict(clean_transit_features)
    ood_features["red_noise_beta_factor"] = 19.9  # near extreme max
    ood_features["out_of_transit_rms_to_depth_ratio"] = 99.0

    pred_ood = bundle.predict(ood_features)
    # If OOD or margin is triggered, should abstain
    assert pred_ood.abstained is True
    assert pred_ood.predicted_class is None


def test_tune_abstention_thresholds_logic() -> None:
    """Verify tune_abstention_thresholds selects valid boundaries."""
    rng = np.random.default_rng(42)
    y_true = np.array([0, 1, 2, 3] * 5)
    # Probs with high confidence
    y_probs = rng.uniform(0.1, 0.9, (20, 4))
    y_probs /= np.sum(y_probs, axis=1, keepdims=True)

    tau_c, tau_m = tune_abstention_thresholds(y_true, y_probs, min_coverage=0.5)
    assert 0.0 <= tau_c <= 1.0
    assert 0.0 <= tau_m <= 1.0
