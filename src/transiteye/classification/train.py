"""Model training, grouped cross-validation, selection, and bundle serialization."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from transiteye.characterization.fitting import refine_candidate
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
from transiteye.datasets.labels import read_labels
from transiteye.datasets.splits import DatasetSplit, read_split_manifest
from transiteye.detection.peaks import find_candidates
from transiteye.domain.models import LightCurve
from transiteye.features.extractors import extract_features, stellar_temperature
from transiteye.io.fits import read_tess_light_curve
from transiteye.preprocessing.detrend import two_pass_detrend
from transiteye.preprocessing.representation import prepare_light_curve

logger = logging.getLogger(__name__)


def evaluate_cv_model(
    model: Any,
    x_data: NDArray[np.float64],
    y_data: NDArray[np.int64],
    groups: list[str],
    n_splits: int = 5,
) -> tuple[float, float, dict[str, dict[str, float]]]:
    """Perform StratifiedGroupKFold cross-validation on training data."""
    unique_groups = len(set(groups))
    n_splits = min(n_splits, unique_groups, int(np.min(np.bincount(y_data))))
    if n_splits < 2:
        # Fall back to 2 splits or simple holdout
        n_splits = 2

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

    y_true_all: list[int] = []
    y_pred_all: list[int] = []

    for train_idx, val_idx in sgkf.split(x_data, y_data, groups=groups):
        x_tr, x_val = x_data[train_idx], x_data[val_idx]
        y_tr, y_val = y_data[train_idx], y_data[val_idx]

        model.fit(x_tr, y_tr)
        preds = model.predict(x_val)

        y_true_all.extend(y_val.tolist())
        y_pred_all.extend(preds.tolist())

    y_true_arr = np.array(y_true_all)
    y_pred_arr = np.array(y_pred_all)

    macro_f1 = float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    # Transit-like recall (class 0)
    transit_recall = float(
        recall_score(
            y_true_arr,
            y_pred_arr,
            labels=[0],
            average="macro",
            zero_division=0,
        )
    )

    class_names = [cls.value for cls in TARGET_MORPHOLOGY_CLASSES]
    report = classification_report(
        y_true_arr,
        y_pred_arr,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    class_metrics: dict[str, dict[str, float]] = {}
    for name in class_names:
        if name in report:
            class_metrics[name] = {
                "precision": float(report[name]["precision"]),
                "recall": float(report[name]["recall"]),
                "f1_score": float(report[name]["f1-score"]),
            }

    return macro_f1, transit_recall, class_metrics


def train_and_select_classifier(
    x_train: NDArray[np.float64],
    y_train: NDArray[np.int64],
    groups_train: list[str],
    x_cal: NDArray[np.float64] | None = None,
    y_cal: NDArray[np.int64] | None = None,
    x_sel: NDArray[np.float64] | None = None,
    y_sel: NDArray[np.int64] | None = None,
    feature_names: tuple[str, ...] = MORPHOLOGY_FEATURE_NAMES,
    train_manifest_id: str = "pilot_v1",
    synthetic_sample_count: int = 0,
) -> tuple[ModelBundle, dict[str, Any]]:
    """Compare candidate classifiers via grouped CV, select best, calibrate, and tune abstention."""
    candidate_estimators: dict[str, Any] = {
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            class_weight="balanced",
            random_state=42,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_depth=6,
            class_weight="balanced",
            random_state=42,
        ),
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    }

    best_name = "random_forest"
    best_macro_f1 = -1.0
    best_transit_recall = -1.0
    best_metrics: dict[str, dict[str, float]] = {}
    cv_comparison: dict[str, Any] = {}

    for name, est in candidate_estimators.items():
        macro_f1, tr_recall, metrics = evaluate_cv_model(
            est, x_train, y_train, groups_train
        )
        cv_comparison[name] = {
            "macro_f1": macro_f1,
            "transit_recall": tr_recall,
            "metrics": metrics,
        }

        # Select by Macro-F1, then transit-like recall
        if macro_f1 > best_macro_f1 or (
            np.isclose(macro_f1, best_macro_f1, atol=1e-3) and tr_recall > best_transit_recall
        ):
            best_macro_f1 = macro_f1
            best_transit_recall = tr_recall
            best_name = name
            best_metrics = metrics

    selected_estimator = candidate_estimators[best_name]
    # Retrain selected estimator on all training data
    selected_estimator.fit(x_train, y_train)

    # Imputation medians and feature bounds
    medians: dict[str, float] = {}
    min_bounds: dict[str, float] = {}
    max_bounds: dict[str, float] = {}
    for i, fname in enumerate(feature_names):
        col = x_train[:, i]
        medians[fname] = float(np.median(col))
        min_bounds[fname] = float(np.min(col))
        max_bounds[fname] = float(np.max(col))

    # Calibration on calibration partition (if provided)
    calibrator = None
    brier_score = None
    ece = None
    if x_cal is not None and y_cal is not None and len(x_cal) >= len(TARGET_MORPHOLOGY_CLASSES):
        calibrator = fit_platt_calibrator(selected_estimator, x_cal, y_cal)
        probs_cal = calibrator.predict_proba(x_cal)
        brier_score = compute_multiclass_brier_score(
            y_cal, probs_cal, len(TARGET_MORPHOLOGY_CLASSES)
        )
        ece = compute_expected_calibration_error(y_cal, probs_cal)

    # Selection partition abstention tuning
    tau_conf = 0.5
    tau_margin = 0.15
    if x_sel is not None and y_sel is not None and len(x_sel) > 0:
        model_for_probs = calibrator if calibrator is not None else selected_estimator
        probs_sel = model_for_probs.predict_proba(x_sel)
        tau_conf, tau_margin = tune_abstention_thresholds(y_sel, probs_sel)

    metadata = ModelMetadata(
        model_id=f"morphology_{best_name}_v1",
        estimator_name=best_name,
        feature_names=feature_names,
        imputation_medians=medians,
        train_manifest_id=train_manifest_id,
        code_version="0.1.0",
        training_sample_count=len(x_train),
        synthetic_sample_count=synthetic_sample_count,
        cv_macro_f1=best_macro_f1,
        cv_metrics_per_class=best_metrics,
        tau_conf=tau_conf,
        tau_margin=tau_margin,
        feature_min_bounds=min_bounds,
        feature_max_bounds=max_bounds,
        extra={
            "cv_comparison": cv_comparison,
            "brier_score": brier_score,
            "ece": ece,
        },
    )

    bundle = ModelBundle(
        metadata=metadata,
        estimator=selected_estimator,
        calibrator=calibrator,
    )

    return bundle, cv_comparison


def _extract_lc_feature_vector(
    lc: LightCurve,
    feature_names: tuple[str, ...],
    snr_threshold: float = 1.5,
) -> NDArray[np.float64] | None:
    """Run real preprocessing, detrending, search, refinement, and feature extraction on a LightCurve."""
    try:
        prepared = prepare_light_curve(lc)
        dual_p1 = two_pass_detrend(prepared, baseline_window_days=0.75, device="auto")
        det = find_candidates(
            dual_p1,
            max_candidates=1,
            snr_threshold=snr_threshold,
            device="auto",
        )
        if not det.candidates:
            det = find_candidates(
                dual_p1,
                max_candidates=1,
                snr_threshold=0.5,
                device="auto",
            )
        if not det.candidates:
            return None

        cand = det.candidates[0]
        candidate_tuples = ((cand.epoch_days or 0.0, cand.period_days or 1.0, cand.duration_hours or 2.0),)
        dual_final = two_pass_detrend(prepared, candidate_events=candidate_tuples, device="auto")
        crowdsap_str = lc.metadata.get("crowdsap")
        crowdsap = float(crowdsap_str) if crowdsap_str and crowdsap_str != "None" else None

        refined = refine_candidate(
            cand,
            dual_final.prepared.time_days,
            dual_final.detrended_flux,
            dual_final.prepared.flux_error_normalized,
            crowdsap=crowdsap,
            batman_snr_threshold=7.0,
            teff=stellar_temperature(lc),
        )
        feats_dict = extract_features(dual_final, refined, det.variability, lc)
        return np.array([float(feats_dict.get(fn, 0.0)) for fn in feature_names], dtype=np.float64)
    except Exception as exc:
        logger.warning("Feature extraction failed for target %s: %s", lc.provenance.target_id, exc)
        return None


def _find_target_fits(pilot_dir: Path, target_id: str) -> Path | None:
    """Find the local FITS file in pilot_dir matching target_id."""
    padded = target_id.zfill(16)
    matches = sorted(p for p in pilot_dir.glob(f"*-{padded}-*_lc.fits") if p.stat().st_size > 100000)
    if not matches:
        matches = sorted(p for p in pilot_dir.glob(f"*0{target_id}-*_lc.fits") if p.stat().st_size > 100000)
    return matches[0] if matches else None


def _augment_feature_vector_with_physical_priors(
    vec: NDArray[np.float64],
    c_idx: int,
    feature_names: tuple[str, ...],
    seed: int,
) -> NDArray[np.float64]:
    """Reject the historical label-conditioned feature transform."""
    raise RuntimeError("Label-conditioned feature transforms are invalid and disabled.")


def train_from_pilot_cohort(
    pilot_dir: Path = Path(".transiteye-workspace/data/pilot"),
    labels_path: Path = Path("manifests/pilot_labels.v1.json"),
    splits_path: Path = Path("manifests/pilot_splits.v1.json"),
    output_dir: Path | None = Path(".transiteye-workspace/models/morphology_v1"),
    feature_names: tuple[str, ...] = MORPHOLOGY_FEATURE_NAMES,
    include_synthetic_augmentation: bool = False,
) -> tuple[ModelBundle, dict[str, Any]]:
    """Train and calibrate the morphology classifier on real TESS FITS light curves from the pilot cohort."""
    if include_synthetic_augmentation:
        raise ValueError("Pilot augmentation is disabled pending independent validation.")
    labels = read_labels(labels_path)
    splits = read_split_manifest(splits_path)
    split_map = {a.source_group_id: a.split for a in splits.assignments}
    classes_str = [c.value for c in TARGET_MORPHOLOGY_CLASSES]

    x_train_real: list[NDArray[np.float64]] = []
    y_train_real: list[int] = []
    g_train_real: list[str] = []

    x_train_syn: list[NDArray[np.float64]] = []
    y_train_syn: list[int] = []
    g_train_syn: list[str] = []

    x_cal_list: list[NDArray[np.float64]] = []
    y_cal_list: list[int] = []

    x_sel_list: list[NDArray[np.float64]] = []
    y_sel_list: list[int] = []

    for lbl in labels:
        split = split_map.get(lbl.source_group_id)
        if split not in (DatasetSplit.TRAIN, DatasetSplit.CALIBRATION, DatasetSplit.SELECTION):
            continue

        raw_cls = lbl.label.value if hasattr(lbl.label, "value") else str(lbl.label)
        if raw_cls not in classes_str:
            continue
        c_idx = classes_str.index(raw_cls)

        fits_path = _find_target_fits(pilot_dir, lbl.target_id)
        if fits_path is None:
            continue

        lc = read_tess_light_curve(fits_path)
        vec = _extract_lc_feature_vector(lc, feature_names)
        if vec is None:
            continue

        if split == DatasetSplit.TRAIN:
            x_train_real.append(vec)
            y_train_real.append(c_idx)
            g_train_real.append(lbl.source_group_id)

        elif split == DatasetSplit.CALIBRATION:
            x_cal_list.append(vec)
            y_cal_list.append(c_idx)
        elif split == DatasetSplit.SELECTION:
            x_sel_list.append(vec)
            y_sel_list.append(c_idx)

    x_tr_real_arr = np.asarray(x_train_real, dtype=np.float64)
    y_tr_real_arr = np.asarray(y_train_real, dtype=np.int64)

    if include_synthetic_augmentation and x_train_syn:
        x_tr_all = np.vstack([x_tr_real_arr, np.asarray(x_train_syn, dtype=np.float64)])
        y_tr_all = np.concatenate([y_tr_real_arr, np.asarray(y_train_syn, dtype=np.int64)])
        g_tr_all = g_train_real + g_train_syn
        syn_count = len(x_train_syn)
    else:
        x_tr_all = x_tr_real_arr
        y_tr_all = y_tr_real_arr
        g_tr_all = g_train_real
        syn_count = 0

    x_cal_arr = np.asarray(x_cal_list, dtype=np.float64) if x_cal_list else None
    y_cal_arr = np.asarray(y_cal_list, dtype=np.int64) if y_cal_list else None
    x_sel_arr = np.asarray(x_sel_list, dtype=np.float64) if x_sel_list else None
    y_sel_arr = np.asarray(y_sel_list, dtype=np.int64) if y_sel_list else None

    bundle, comparison = train_and_select_classifier(
        x_train=x_tr_all,
        y_train=y_tr_all,
        groups_train=g_tr_all,
        x_cal=x_cal_arr,
        y_cal=y_cal_arr,
        x_sel=x_sel_arr,
        y_sel=y_sel_arr,
        feature_names=feature_names,
        train_manifest_id=splits.manifest_id,
        synthetic_sample_count=syn_count,
    )

    if output_dir is not None:
        bundle.save(output_dir)

    return bundle, comparison


def get_default_morphology_model(models_dir: Path | None = None) -> ModelBundle:
    """Reject implicit model selection and synthetic fallback training."""
    raise RuntimeError(
        "Implicit classifier selection is disabled. Pass an explicitly selected ModelBundle "
        "to the analysis service; no model is trained automatically."
    )
