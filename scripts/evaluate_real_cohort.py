"""One-shot, label-blind evaluation of the real catalog cohort classifier."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import tomllib
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from transiteye.characterization.fitting import refine_candidate
from transiteye.classification.models import (
    MORPHOLOGY_FEATURE_NAMES,
    TARGET_MORPHOLOGY_CLASSES,
    ModelBundle,
)
from transiteye.datasets.labels import read_labels
from transiteye.datasets.splits import (
    DatasetSplit,
    read_split_manifest,
    validate_labels_against_split,
)
from transiteye.detection.peaks import find_candidates
from transiteye.evaluation.protocol import compute_classification_metrics
from transiteye.features.extractors import extract_features
from transiteye.io.fits import read_tess_light_curve
from transiteye.io.manifest import read_manifest
from transiteye.preprocessing.detrend import two_pass_detrend
from transiteye.preprocessing.representation import prepare_light_curve

LOGGER = logging.getLogger("transiteye.evaluate_real_cohort")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _features_from_light_curve(lc: Any) -> tuple[np.ndarray | None, str]:
    """Mirror the fixed training feature path without accepting or inspecting a label."""
    prepared = prepare_light_curve(lc)
    dual_pass1 = two_pass_detrend(prepared, baseline_window_days=0.75, device="auto")
    detection = find_candidates(
        dual_pass1, max_candidates=1, snr_threshold=1.5, device="auto"
    )
    if not detection.candidates:
        detection = find_candidates(
            dual_pass1, max_candidates=1, snr_threshold=0.5, device="auto"
        )
    if not detection.candidates:
        return None, "no_candidate_at_fixed_search_thresholds"

    candidate = detection.candidates[0]
    event = (
        candidate.epoch_days or 0.0,
        candidate.period_days or 1.0,
        candidate.duration_hours or 2.0,
    )
    dual_final = two_pass_detrend(
        prepared, candidate_events=(event,), device="auto"
    )
    crowdsap_str = lc.metadata.get("crowdsap")
    crowdsap = (
        float(crowdsap_str)
        if crowdsap_str and crowdsap_str != "None"
        else None
    )
    refined = refine_candidate(
        candidate,
        dual_final.prepared.time_days,
        dual_final.detrended_flux,
        dual_final.prepared.flux_error_normalized,
        crowdsap=crowdsap,
        batman_snr_threshold=7.0,
    )
    feature_map = extract_features(dual_final, refined, detection.variability, lc)
    vector = np.asarray(
        [float(feature_map.get(name, 0.0)) for name in MORPHOLOGY_FEATURE_NAMES],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(vector)):
        return None, "non_finite_features"
    return vector, "candidate_features_extracted"


def _metrics(indices: list[int], records: list[dict[str, Any]], class_names: list[str]) -> dict[str, float]:
    if not indices:
        return {"accuracy": 0.0, "macro_f1": 0.0, "transit_like_recall": 0.0}
    selected = [records[index] for index in indices]
    metric = compute_classification_metrics(
        [row["true_class"] for row in selected],
        [row["predicted_class"] for row in selected],
        np.asarray([row["probabilities"] for row in selected], dtype=np.float64),
        class_names,
        [row["abstained"] for row in selected],
    )
    return {
        "accuracy": metric.accuracy,
        "macro_f1": metric.macro_f1,
        "transit_like_recall": metric.per_class_recall["transit_like"],
    }


def _grouped_bootstrap(
    records: list[dict[str, Any]],
    indices: list[int],
    class_names: list[str],
    replicates: int,
    seed: int,
) -> dict[str, list[float]]:
    """Percentile intervals resampling source groups with replacement."""
    rng = np.random.default_rng(seed)
    groups = np.asarray([records[index]["source_group_id"] for index in indices])
    unique_groups = np.unique(groups)
    draws: dict[str, list[float]] = {
        "accuracy": [], "macro_f1": [], "transit_like_recall": []
    }
    group_to_indices = {
        group: [indices[pos] for pos in np.flatnonzero(groups == group)]
        for group in unique_groups
    }
    for _ in range(replicates):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        sample_indices = [
            index for group in sampled for index in group_to_indices[str(group)]
        ]
        replicate_metrics = _metrics(sample_indices, records, class_names)
        for name, value in replicate_metrics.items():
            draws[name].append(value)
    return {
        name: [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]
        for name, values in draws.items()
    }


def evaluate(config_path: Path, output_dir: Path) -> dict[str, Any]:
    raise RuntimeError(
        "Historical real-cohort evaluation is disabled: its test was already opened and its "
        "proxy labels and feature path do not validate the current pipeline."
    )
    config_path = config_path.resolve()
    repo_root = config_path.parent.parent
    config_checksum = _sha256(config_path)
    checksum_sidecar = config_path.with_suffix(config_path.suffix + ".sha256")
    if not checksum_sidecar.is_file() or checksum_sidecar.read_text(encoding="utf-8").split()[0] != config_checksum:
        raise ValueError("Frozen protocol checksum sidecar is missing or does not match.")
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    protocol = config["protocol"]
    if _sha256(Path(__file__).resolve()) != protocol["sha256"]["runner"]:
        raise ValueError("Evaluation runner checksum differs from the frozen protocol.")
    class_names = [item.value for item in TARGET_MORPHOLOGY_CLASSES]
    labels_path = repo_root / protocol["labels_manifest"]
    splits_path = repo_root / protocol["splits_manifest"]
    products_path = repo_root / protocol["products_manifest"]
    data_dir = repo_root / protocol["data_dir"]
    model_dir = repo_root / protocol["model_dir"]

    locked_hashes = protocol["sha256"]
    for key, path in {
        "labels_manifest": labels_path,
        "splits_manifest": splits_path,
        "products_manifest": products_path,
        "model_metadata": model_dir / "model_metadata.json",
        "model_estimator": model_dir / "estimator.joblib",
        "model_calibrator": model_dir / "calibrator.joblib",
    }.items():
        actual = _sha256(path)
        if actual != locked_hashes[key]:
            raise ValueError(f"Locked input checksum mismatch for {key}: {actual}")

    splits = read_split_manifest(splits_path)
    labels = read_labels(labels_path)
    validate_labels_against_split(labels, splits)
    split_by_group = {entry.source_group_id: entry.split for entry in splits.assignments}
    test_labels = sorted(
        (label for label in labels if split_by_group[label.source_group_id] == DatasetSplit.TEST),
        key=lambda label: label.target_id,
    )
    if len(test_labels) != int(protocol["expected_test_sources"]):
        raise ValueError(f"Expected {protocol['expected_test_sources']} locked test labels; got {len(test_labels)}")

    product_manifest = read_manifest(products_path)
    products_by_target: dict[str, Any] = {}
    for product in product_manifest.entries:
        current = products_by_target.get(product.target_id)
        if current is None or (int(product.sector or -1), product.local_filename or "") > (
            int(current.sector or -1), current.local_filename or ""
        ):
            products_by_target[product.target_id] = product

    bundle = ModelBundle.load(model_dir)
    if bundle.metadata.model_id != protocol["model_id"]:
        raise ValueError("Loaded model ID differs from the frozen protocol.")
    if bundle.metadata.tau_conf != float(protocol["tau_confidence"]):
        raise ValueError("Model confidence threshold differs from the frozen protocol.")
    if bundle.metadata.tau_margin != float(protocol["tau_margin"]):
        raise ValueError("Model margin threshold differs from the frozen protocol.")

    # All model decisions are made before a test label is read for scoring.
    records: list[dict[str, Any]] = []
    for index, label in enumerate(test_labels, start=1):
        product = products_by_target.get(label.target_id)
        if product is None:
            raise FileNotFoundError(f"No locked product for test TIC {label.target_id}")
        fits_path = data_dir / product.resolved_filename()
        actual_checksum = _sha256(fits_path)
        if actual_checksum != product.expected_sha256:
            raise ValueError(f"FITS checksum mismatch for TIC {label.target_id}")

        status = "feature_extraction_error"
        error: str | None = None
        probabilities = [0.25] * len(class_names)
        predicted_index = 0
        abstained = True
        try:
            light_curve = read_tess_light_curve(fits_path)
            vector, status = _features_from_light_curve(light_curve)
            if vector is not None:
                feature_map = dict(zip(MORPHOLOGY_FEATURE_NAMES, vector, strict=True))
                prediction = bundle.predict(feature_map)
                probabilities = [
                    float(prediction.probabilities[enum])
                    for enum in TARGET_MORPHOLOGY_CLASSES
                ]
                predicted_index = int(np.argmax(probabilities))
                abstained = bool(prediction.abstained)
        except Exception as exc:  # Record a failed target as an abstention in system metrics.
            status = "feature_extraction_error"
            error = repr(exc)
            LOGGER.exception("Evaluation failed for locked test TIC %s", label.target_id)

        records.append({
            "target_id": label.target_id,
            "source_group_id": label.source_group_id,
            "product_id": product.product_id,
            "sector": product.sector,
            "product_sha256": actual_checksum,
            "status": status,
            "true_class": label.label.value,
            "predicted_class": class_names[predicted_index],
            "probabilities": probabilities,
            "abstained": abstained,
            "error": error,
        })
        LOGGER.info("Evaluated locked source %d/%d (TIC %s): %s", index, len(test_labels), label.target_id, status)

    successful_indices = [
        i for i, row in enumerate(records) if row["status"] == "candidate_features_extracted"
    ]
    all_indices = list(range(len(records)))
    successful_metrics = compute_classification_metrics(
        [records[i]["true_class"] for i in successful_indices],
        [records[i]["predicted_class"] for i in successful_indices],
        np.asarray([records[i]["probabilities"] for i in successful_indices], dtype=np.float64),
        class_names,
        [records[i]["abstained"] for i in successful_indices],
    )
    system_metrics = compute_classification_metrics(
        [row["true_class"] for row in records],
        [row["predicted_class"] for row in records],
        np.asarray([row["probabilities"] for row in records], dtype=np.float64),
        class_names,
        [row["abstained"] for row in records],
    )
    system_metrics_report = {
        name: getattr(system_metrics, name)
        for name in system_metrics.__dataclass_fields__
        if name not in {"expected_calibration_error", "brier_score"}
    }
    class_counts = Counter(row["true_class"] for row in records)
    candidate_counts = Counter(row["true_class"] for row in records if row["status"] == "candidate_features_extracted")
    failures = [row for row in records if row["error"]]
    bootstrap = {
        "candidate_conditional": _grouped_bootstrap(
            records, successful_indices, class_names,
            int(protocol["bootstrap_replicates"]), int(protocol["bootstrap_seed"]),
        ),
        "end_to_end": _grouped_bootstrap(
            records, all_indices, class_names,
            int(protocol["bootstrap_replicates"]), int(protocol["bootstrap_seed"]),
        ),
    }
    report: dict[str, Any] = {
        "protocol_id": protocol["protocol_id"],
        "protocol_config_sha256": config_checksum,
        "evaluated_at_utc": datetime.now(UTC).isoformat(),
        "model_id": bundle.metadata.model_id,
        "model_estimator": bundle.metadata.estimator_name,
        "split_manifest_id": splits.manifest_id,
        "product_manifest_id": product_manifest.manifest_id,
        "input_checksums": locked_hashes,
        "test_split_opened": True,
        "test_source_count": len(records),
        "class_counts": dict(class_counts),
        "candidate_feature_success_count": len(successful_indices),
        "candidate_feature_success_by_class": dict(candidate_counts),
        "no_candidate_count": sum(row["status"] == "no_candidate_at_fixed_search_thresholds" for row in records),
        "failed_target_count": len(failures),
        "classification_conditional_on_candidate": successful_metrics.__dict__ if hasattr(successful_metrics, "__dict__") else {
            name: getattr(successful_metrics, name) for name in successful_metrics.__dataclass_fields__
        },
        "end_to_end_with_misses_and_failures_abstaining": {
            **system_metrics_report,
            "calibration_metrics_omitted": True,
        },
        "bootstrap_95_percentile_intervals": bootstrap,
        "fixed_decision_thresholds": {
            "tau_confidence": bundle.metadata.tau_conf,
            "tau_margin": bundle.metadata.tau_margin,
            "candidate_search_snr_threshold_sequence": [1.5, 0.5],
        },
        "label_leakage_controls": {
            "labels_used_for_feature_extraction_or_prediction": False,
            "test_labels_used_only_after_predictions_for_scoring": True,
            "selection_or_threshold_tuning_on_test": False,
        },
        "limitations": [
            "Catalog-derived morphology labels are proxies and are not independent human light-curve adjudications.",
            "Classification conditional on a detected candidate is distinct from end-to-end system performance.",
            "A transit-like morphology classification does not confirm an exoplanet.",
        ],
        "failures": [{key: row[key] for key in ("target_id", "status", "error")} for row in failures],
        "per_source": records,
    }
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "evaluation_report.json"
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite one-shot evaluation: {report_path}")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/evaluation_real_catalog.v1.toml"))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(".transiteye-workspace/evaluations/real_catalog_cohort_v1"),
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = evaluate(args.config, args.output_dir)
    print(json.dumps({key: value for key, value in result.items() if key != "per_source"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
