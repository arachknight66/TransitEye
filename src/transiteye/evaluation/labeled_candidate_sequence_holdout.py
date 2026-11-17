"""One-time locked-holdout evaluation for the labeled candidate-sequence models."""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import multiprocessing
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)

from transiteye.classification.sequence_model import CandidateSequenceCNN
from transiteye.evaluation.labeled_candidate_sequence_training import _extract_source_group
from transiteye.evaluation.labeled_sequence_cohort import CLASSES, sha256_file

LOGGER = logging.getLogger(__name__)


def _scores(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    predictions = probabilities.argmax(axis=1)
    one_hot = np.eye(len(CLASSES), dtype=np.float64)[y_true]
    return {
        "sample_count": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(
            y_true, predictions, labels=np.arange(len(CLASSES))
        ).tolist(),
        "class_order": list(CLASSES),
        "classification_report": classification_report(
            y_true,
            predictions,
            labels=np.arange(len(CLASSES)),
            target_names=list(CLASSES),
            output_dict=True,
            zero_division=0,
        ),
        "log_loss": float(log_loss(y_true, probabilities, labels=np.arange(len(CLASSES)))),
        "multiclass_brier_score": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
    }


def evaluate_locked_holdout(
    manifest_path: Path,
    model_dir: Path,
    *,
    workers: int = 4,
    device: str = "cuda",
) -> dict[str, Any]:
    """Evaluate frozen artifacts once on the source-grouped holdout without fitting or tuning."""
    if not 1 <= workers <= 8:
        raise ValueError("workers must be between 1 and 8")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; explicitly select CPU to continue")

    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("protocol_id") != "labeled_catalog_event_candidate_sequence_binary_v1":
        raise ValueError("Unexpected cohort protocol identity")
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    fit_report = json.loads((model_dir / "training_report.json").read_text(encoding="utf-8"))
    if fit_report.get("cohort_manifest_sha256") != manifest_sha256:
        raise ValueError("Model report and holdout manifest checksums differ")
    if fit_report.get("fresh_holdout_opened") is not False:
        raise ValueError("Training report indicates the fresh holdout was already opened")
    if fit_report.get("training_source") != "fresh_initialization_no_pretrained_weights":
        raise ValueError("Unexpected model training provenance")

    output_path = model_dir / "holdout_evaluation.json"
    if output_path.exists():
        raise FileExistsError(
            f"Locked holdout report already exists at {output_path}; refusing a second evaluation"
        )
    state_path = model_dir / "sequence_model_state.pt"
    forest_path = model_dir / "same_candidate_random_forest.joblib"
    state_sha256 = sha256_file(state_path)
    forest_sha256 = sha256_file(forest_path)
    cache_identity = hashlib.sha256(json.dumps({
        "manifest_sha256": manifest_sha256,
        "fit_report_sha256": sha256_file(model_dir / "training_report.json"),
        "cnn_state_sha256": state_sha256,
        "baseline_sha256": forest_sha256,
        "extractor_source": inspect.getsource(_extract_source_group),
    }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    marker_path = model_dir / "holdout_evaluation_started.json"
    marker = {
        "cache_identity": cache_identity,
        "manifest_sha256": manifest_sha256,
        "cnn_state_sha256": state_sha256,
        "baseline_sha256": forest_sha256,
        "status": "in_progress",
        "holdout_opened_for_final_evaluation": True,
    }
    if marker_path.exists():
        prior_marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if prior_marker.get("cache_identity") != cache_identity:
            raise ValueError("A different model or protocol already opened this holdout")
    else:
        marker_path.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    product_groups: dict[str, list[dict[str, Any]]] = {}
    for row in manifest["products"]:
        if row["split"] == "fresh_holdout":
            product_groups.setdefault(row["source_group_id"], []).append(row)
    tasks: list[dict[str, Any]] = []
    for group, rows in sorted(product_groups.items()):
        labels = {row["label"] for row in rows}
        if len(labels) != 1:
            raise ValueError(f"Holdout TIC has conflicting labels: {group}")
        ordered = sorted(rows, key=lambda row: (-len(row["event_refs"]), row["product_id"]))
        tasks.append({
            "source_group_id": group,
            "label": next(iter(labels)),
            "split": "fresh_holdout",
            "products": ordered,
        })
    expected_groups = manifest.get("partition_counts", {}).get("fresh_holdout")
    if expected_groups is not None and expected_groups.get("source_groups") != len(tasks):
        raise ValueError("Holdout group count does not match the frozen manifest")
    if not tasks:
        raise ValueError("Frozen manifest contains no fresh holdout groups")

    LOGGER.info("Extracting frozen holdout candidates for %d TIC groups", len(tasks))
    results: dict[str, dict[str, Any]] = {}
    checkpoint_dir = model_dir / "holdout_candidate_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    pending: list[dict[str, Any]] = []
    for task in tasks:
        token = hashlib.sha256(task["source_group_id"].encode()).hexdigest()[:24]
        cache_path = checkpoint_dir / f"{token}.json"
        if cache_path.is_file():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if cached.get("cache_identity") == cache_identity:
                    results[task["source_group_id"]] = cached["result"]
                    continue
            except (OSError, json.JSONDecodeError, KeyError):
                pass
        pending.append(task)

    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        futures = {pool.submit(_extract_source_group, task): task for task in pending}
        for completed, future in enumerate(as_completed(futures), start=1):
            task = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "source_group_id": task["source_group_id"],
                    "label": task["label"],
                    "split": "fresh_holdout",
                    "status": "worker_failure",
                    "error": f"{type(exc).__name__}: {exc}",
                    "attempts": [],
                }
            results[task["source_group_id"]] = result
            token = hashlib.sha256(task["source_group_id"].encode()).hexdigest()[:24]
            cache_path = checkpoint_dir / f"{token}.json"
            temporary = cache_path.with_suffix(".tmp")
            temporary.write_text(json.dumps({
                "cache_identity": cache_identity,
                "result": result,
            }, allow_nan=False, sort_keys=True) + "\n", encoding="utf-8")
            temporary.replace(cache_path)
            if completed == 1 or completed % 25 == 0 or completed == len(pending):
                LOGGER.info("Holdout candidate matching %d/%d pending groups", completed, len(pending))

    ordered_results = [results[task["source_group_id"]] for task in tasks]
    matches = [row for row in ordered_results if row["status"] == "matched"]
    status_counts = Counter(row["status"] for row in ordered_results)
    if not matches:
        raise RuntimeError("No holdout candidates matched; see holdout extraction audit")

    results_path = model_dir / "holdout_candidate_matching_results.parquet"
    sequences_path = model_dir / "holdout_candidate_sequences.npz"
    pq.write_table(
        pa.Table.from_pylist([
            {key: value for key, value in row.items() if key != "sequence"}
            for row in ordered_results
        ]),
        results_path,
        compression="zstd",
    )
    np.savez_compressed(
        sequences_path,
        source_group_id=np.asarray([row["source_group_id"] for row in matches], dtype=str),
        label=np.asarray([row["label"] for row in matches], dtype=str),
        sequence=np.stack([row["sequence"] for row in matches]).astype(np.float32),
    )

    class_index = {label: index for index, label in enumerate(CLASSES)}
    y_true = np.asarray([class_index[row["label"]] for row in matches], dtype=np.int64)
    sequences = np.stack([row["sequence"] for row in matches]).astype(np.float32)
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    model = CandidateSequenceCNN(class_count=len(CLASSES))
    model.load_state_dict(state)
    model.to(device).eval()
    with torch.inference_mode():
        logits = model(torch.from_numpy(sequences).to(device)).cpu().numpy()
    temperature = float(fit_report["cnn"]["calibration_temperature"])
    scaled_logits = logits / temperature
    shifted = scaled_logits - scaled_logits.max(axis=1, keepdims=True)
    cnn_probabilities = np.exp(shifted)
    cnn_probabilities /= cnn_probabilities.sum(axis=1, keepdims=True)

    forest = joblib.load(forest_path)
    forest_probabilities = forest.predict_proba(sequences.reshape(len(sequences), -1))
    report = {
        "evaluation_protocol_id": "labeled_catalog_event_candidate_sequence_binary_v1_locked_holdout",
        "evaluation_mode": "one_time_final_inference_only_no_fitting_calibration_or_selection",
        "cohort_manifest_sha256": manifest_sha256,
        "fit_report_sha256": sha256_file(model_dir / "training_report.json"),
        "cnn_state_sha256": state_sha256,
        "baseline_artifact_sha256": forest_sha256,
        "candidate_matching_results_sha256": sha256_file(results_path),
        "candidate_sequence_cache_sha256": sha256_file(sequences_path),
        "holdout_source_group_count": len(tasks),
        "candidate_matching_status_counts": dict(status_counts),
        "matched_candidate_count": len(matches),
        "matched_class_counts": dict(Counter(row["label"] for row in matches)),
        "class_order": list(CLASSES),
        "calibration_temperature_from_development": temperature,
        "cnn": _scores(y_true, cnn_probabilities),
        "same_candidate_random_forest": _scores(y_true, forest_probabilities),
        "holdout_opened_for_final_evaluation": True,
        "holdout_used_for_fit_or_model_selection": False,
        "limitations": [
            "labels remain catalog-derived proxies, not independent morphology adjudication",
            "metrics are conditional on blind detection and catalog-event matching",
            "unmatched and failed candidates are reported in coverage counts, not treated as negatives",
            "the model covers transit_like and eclipse_like only",
        ],
    }
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    marker["status"] = "complete"
    marker["report_sha256"] = sha256_file(output_path)
    marker_path.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
