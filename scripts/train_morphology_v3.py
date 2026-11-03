"""Training and calibration script for morphology classifier v3 on cohort_v3 (10,500 training samples).

Loads precomputed features from .transiteye-workspace/data/cohort_v3_features.npz,
performs 5-fold StratifiedGroupKFold cross-validation comparison, fits Platt calibrator,
tunes abstention thresholds, and serializes .transiteye-workspace/models/morphology_v3/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from transiteye.classification.calibration import (
    compute_expected_calibration_error,
    compute_multiclass_brier_score,
    fit_platt_calibrator,
    tune_abstention_thresholds,
)
from transiteye.classification.models import (
    ModelMetadata,
)
from transiteye.classification.train import evaluate_cv_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_morphology_v3")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train and calibrate morphology_v3 model on cohort_v3.")
    parser.add_argument(
        "--features-file",
        type=Path,
        default=Path(".transiteye-workspace/data/cohort_v3_features.npz"),
        help="Path to precomputed features .npz file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".transiteye-workspace/models/morphology_v3"),
        help="Output model directory",
    )
    parser.add_argument(
        "--splits-file",
        type=Path,
        default=Path("manifests/cohort_v3_splits.json"),
        help="Path to splits manifest to extract manifest_id",
    )
    args = parser.parse_args()

    if not args.features_file.exists():
        raise FileNotFoundError(f"Features file missing: {args.features_file}")

    logger.info("Loading precomputed feature matrices from %s...", args.features_file)
    data = np.load(args.features_file, allow_pickle=True)
    x_train = data["x_train"]
    y_train = data["y_train"]
    groups_train = list(data["groups_train"])
    x_cal = data["x_cal"]
    y_cal = data["y_cal"]
    x_sel = data["x_sel"]
    y_sel = data["y_sel"]
    feature_names = tuple(str(fn) for fn in data["feature_names"])

    manifest_id = "cohort_v3"
    if args.splits_file.exists():
        with open(args.splits_file, encoding="utf-8") as f:
            manifest_id = json.load(f).get("manifest_id", "cohort_v3")

    logger.info(
        "Loaded: %d train samples (%d unique groups), %d cal samples, %d sel samples, %d features.",
        len(x_train),
        len(set(groups_train)),
        len(x_cal),
        len(x_sel),
        len(feature_names),
    )

    # 1. 5-Fold StratifiedGroupKFold Candidate Model Comparison
    logger.info("Running 5-fold StratifiedGroupKFold cross-validation across architectures...")
    candidates = {
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=14,
            min_samples_split=3,
            n_jobs=-1,
            random_state=42,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=150,
            max_leaf_nodes=31,
            min_samples_leaf=5,
            random_state=42,
        ),
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        ),
    }

    cv_comparison: dict[str, Any] = {}
    best_estimator = None
    best_name = ""
    best_macro_f1 = -1.0
    best_transit_recall = -1.0
    best_metrics: dict[str, dict[str, float]] = {}

    for name, model in candidates.items():
        macro_f1, transit_rec, metrics = evaluate_cv_model(
            model, x_train, y_train, groups=groups_train, n_splits=5
        )
        logger.info(
            "Candidate [%s]: Grouped CV Macro-F1 = %.4f, Transit Recall = %.4f",
            name,
            macro_f1,
            transit_rec,
        )
        cv_comparison[name] = {
            "macro_f1": macro_f1,
            "transit_recall": transit_rec,
            "metrics": metrics,
        }
        if (macro_f1 > best_macro_f1) or (
            abs(macro_f1 - best_macro_f1) < 1e-4 and transit_rec > best_transit_recall
        ):
            best_estimator = model
            best_name = name
            best_macro_f1 = macro_f1
            best_transit_recall = transit_rec
            best_metrics = metrics

    logger.info("Selected champion: %s (Macro-F1=%.4f)", best_name, best_macro_f1)

    # 2. Refit Selected Champion on all training data
    logger.info("Fitting champion [%s] on all %d training samples...", best_name, len(x_train))
    best_estimator.fit(x_train, y_train)

    # 3. Platt Sigmoid Calibration on held-out calibration partition
    logger.info("Fitting Platt sigmoid calibrator on %d held-out calibration targets...", len(x_cal))
    calibrator = fit_platt_calibrator(best_estimator, x_cal, y_cal)

    # Calibration diagnostics
    probs_cal = calibrator.predict_proba(x_cal)
    brier_score = compute_multiclass_brier_score(y_cal, probs_cal, 4)
    ece = compute_expected_calibration_error(y_cal, probs_cal)
    logger.info("Calibration diagnostics on CALIBRATION partition: Brier Score = %.6f, ECE = %.4f", brier_score, ece)

    # 4. Tune Abstention Thresholds on held-out selection partition
    logger.info("Tuning abstention thresholds on %d held-out selection targets...", len(x_sel))
    probs_sel = calibrator.predict_proba(x_sel)
    tau_conf, tau_margin = tune_abstention_thresholds(y_sel, probs_sel)
    logger.info("Tuned thresholds: tau_conf = %.3f, tau_margin = %.3f", tau_conf, tau_margin)

    # 5. Compute feature bounding boxes and medians
    medians = {fn: float(np.median(x_train[:, i])) for i, fn in enumerate(feature_names)}
    min_bounds = {fn: float(np.min(x_train[:, i])) for i, fn in enumerate(feature_names)}
    max_bounds = {fn: float(np.max(x_train[:, i])) for i, fn in enumerate(feature_names)}

    # Serialize estimator and calibrator
    args.output_dir.mkdir(parents=True, exist_ok=True)
    est_path = args.output_dir / "estimator.joblib"
    cal_path = args.output_dir / "calibrator.joblib"
    meta_path = args.output_dir / "model_metadata.json"

    joblib.dump(best_estimator, est_path)
    joblib.dump(calibrator, cal_path)

    est_bytes = est_path.read_bytes()
    est_sha256 = hashlib.sha256(est_bytes).hexdigest()

    # Model metadata
    metadata = ModelMetadata(
        model_id=f"morphology_{best_name}_v3",
        estimator_name=best_name,
        feature_names=feature_names,
        imputation_medians=medians,
        train_manifest_id=manifest_id,
        code_version="0.1.0",
        training_sample_count=len(x_train),
        synthetic_sample_count=len(x_train) - (len(x_train) // 3),
        cv_macro_f1=best_macro_f1,
        cv_metrics_per_class=best_metrics,
        tau_conf=tau_conf,
        tau_margin=tau_margin,
        feature_min_bounds=min_bounds,
        feature_max_bounds=max_bounds,
        extra={
            "estimator_sha256": est_sha256,
            "cv_comparison": cv_comparison,
            "brier_score": brier_score,
            "ece": ece,
        },
    )

    with open(meta_path, "w", encoding="utf-8") as f:
        meta_dict = {
            "model_id": metadata.model_id,
            "estimator_name": metadata.estimator_name,
            "estimator_sha256": est_sha256,
            "feature_names": list(metadata.feature_names),
            "imputation_medians": metadata.imputation_medians,
            "train_manifest_id": metadata.train_manifest_id,
            "code_version": metadata.code_version,
            "training_sample_count": metadata.training_sample_count,
            "synthetic_sample_count": metadata.synthetic_sample_count,
            "cv_macro_f1": metadata.cv_macro_f1,
            "cv_metrics_per_class": metadata.cv_metrics_per_class,
            "tau_conf": metadata.tau_conf,
            "tau_margin": metadata.tau_margin,
            "feature_min_bounds": metadata.feature_min_bounds,
            "feature_max_bounds": metadata.feature_max_bounds,
            "has_calibrator": True,
            "schema_version": "1",
            "extra": metadata.extra,
        }
        json.dump(meta_dict, f, indent=2, sort_keys=True)

    logger.info("Successfully serialized morphology_v3 bundle to %s.", args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
