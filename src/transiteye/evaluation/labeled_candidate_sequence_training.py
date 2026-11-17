"""Train an isolated binary CNN after blind candidate-to-catalog event matching.

Only the frozen development partitions are opened. Catalog ephemerides select
matching detections and align sequence windows; they are never model inputs.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import math
import multiprocessing
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from scipy.optimize import minimize_scalar
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from transiteye.classification.sequence_model import CandidateSequenceCNN, build_candidate_sequence
from transiteye.detection.peaks import find_candidates
from transiteye.evaluation.labeled_sequence_cohort import CLASSES, SEED, sha256_file
from transiteye.io.fits import read_tess_light_curve
from transiteye.preprocessing.detrend import two_pass_detrend
from transiteye.preprocessing.representation import prepare_light_curve

LOGGER = logging.getLogger(__name__)
BINS = 256
PARTITIONS = ("train", "calibration", "selection")


def _digest_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _phase_distance(candidate_epoch: float, event_epoch: float, period: float) -> float:
    return abs((candidate_epoch - event_epoch + 0.5 * period) % period - 0.5 * period)


def _candidate_matches_event(candidate: Any, event: dict[str, Any], label: str) -> bool:
    """Apply fixed period-alias and phase gates to a blind BLS candidate."""
    candidate_period = candidate.period_days
    candidate_epoch = candidate.epoch_days
    candidate_duration = candidate.duration_hours
    if (
        candidate_period is None
        or candidate_epoch is None
        or candidate_duration is None
        or candidate_period <= 0
        or candidate_duration <= 0
    ):
        return False
    event_period = float(event["period_days"])
    event_epoch = float(event["epoch_days_bjd_minus_2457000"])
    if event_period <= 0:
        return False
    alias_ratios = (0.5, 1.0, 2.0)
    if not any(abs(candidate_period / (event_period * ratio) - 1.0) <= 0.02 for ratio in alias_ratios):
        return False
    offsets = (0.0, 0.5 * event_period) if label == "eclipse_like" else (0.0,)
    phase_error = min(_phase_distance(candidate_epoch, event_epoch + offset, event_period) for offset in offsets)
    reference_duration = event.get("duration_hours")
    tolerance_days = max(
        0.5 / 24.0,
        0.5 * candidate_duration / 24.0,
        (1.25 * float(reference_duration) / 24.0) if reference_duration is not None else 0.0,
    )
    return phase_error <= tolerance_days


def _extract_source_group(task: dict[str, Any]) -> dict[str, Any]:
    """Blindly search products for a source and return its first event-matched sequence."""
    group_id = task["source_group_id"]
    label = task["label"]
    attempts: list[dict[str, Any]] = []
    for product in task["products"]:
        path = Path(product["relative_path"])
        attempt: dict[str, Any] = {"product_id": product["product_id"], "status": "error"}
        try:
            if sha256_file(path) != product["expected_sha256"]:
                raise ValueError("product SHA-256 mismatch")
            light_curve = read_tess_light_curve(path)
            if str(light_curve.provenance.target_id) != str(product["target_id"]):
                raise ValueError("FITS TIC does not match frozen product record")
            if int(light_curve.provenance.sector or -1) != int(product["sector"]):
                raise ValueError("FITS sector does not match frozen product record")
            prepared = prepare_light_curve(light_curve)
            detrended = two_pass_detrend(prepared, baseline_window_days=0.75, device="cpu")
            detection = find_candidates(
                detrended,
                max_candidates=3,
                snr_threshold=6.0,
                device="cpu",
                min_period=0.3,
                max_period=None,
                min_duration_hours=0.5,
                max_duration_hours=12.0,
                num_durations=15,
                frequency_factor=5.0,
            )
            usable_candidates = [
                item for item in detection.candidates
                if item.observed_event_count is not None and item.observed_event_count >= 2
            ]
            matching = [
                (candidate, event)
                for candidate in usable_candidates
                for event in product["event_refs"]
                if _candidate_matches_event(candidate, event, label)
            ]
            if not matching:
                attempt["status"] = "no_repeated_candidate_matches_event"
                attempt["detected_candidate_count"] = len(detection.candidates)
                attempt["repeated_candidate_count"] = len(usable_candidates)
                attempts.append(attempt)
                continue
            candidate, event = max(
                matching,
                key=lambda pair: float(pair[0].detection_snr or 0.0),
            )
            sequence = build_candidate_sequence(
                prepared.time_days,
                prepared.flux_normalized,
                detrended.detrended_flux,
                float(candidate.epoch_days),
                float(candidate.period_days),
                float(candidate.duration_hours),
                bins=BINS,
            )
            if not np.isfinite(sequence).all():
                raise ValueError("candidate sequence contains non-finite values")
            attempt.update({
                "status": "candidate_matched",
                "detected_candidate_count": len(detection.candidates),
                "repeated_candidate_count": len(usable_candidates),
                "matched_catalog": event["catalog"],
                "matched_record_id": event["record_id"],
                "candidate_period_days": float(candidate.period_days),
                "candidate_duration_hours": float(candidate.duration_hours),
                "candidate_detection_snr": float(candidate.detection_snr or 0.0),
                "candidate_observed_event_count": int(candidate.observed_event_count),
                "candidate_flags": list(candidate.flags),
            })
            attempts.append(attempt)
            return {
                "source_group_id": group_id,
                "label": label,
                "status": "matched",
                "product_id": product["product_id"],
                "split": product["split"],
                "sequence": sequence.astype(np.float32).tolist(),
                "candidate": {key: attempt[key] for key in (
                    "candidate_period_days", "candidate_duration_hours", "candidate_detection_snr",
                    "candidate_observed_event_count", "candidate_flags", "matched_catalog",
                    "matched_record_id",
                )},
                "attempts": attempts,
            }
        except Exception as exc:
            attempt["status"] = "input_or_detection_failure"
            attempt["error"] = f"{type(exc).__name__}: {exc}"
            attempts.append(attempt)
    return {
        "source_group_id": group_id,
        "label": label,
        "split": task["split"],
        "status": "no_candidate_match",
        "attempts": attempts,
    }


def _temperature(logits: Tensor, labels: Tensor) -> float:
    logits = logits.detach().cpu()
    labels = labels.detach().cpu()

    def objective(log_t: float) -> float:
        return float(nn.functional.cross_entropy(logits / math.exp(log_t), labels).item())

    result = minimize_scalar(objective, bounds=(-3.0, 3.0), method="bounded")
    if not result.success or not math.isfinite(float(result.fun)):
        raise RuntimeError("Temperature scaling failed on the calibration partition")
    return float(math.exp(result.x))


def _grouped_scores(labels: list[int], logits: np.ndarray) -> dict[str, Any]:
    predictions = logits.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=list(range(len(CLASSES)))).tolist(),
        "class_order": list(CLASSES),
        "sample_count": len(labels),
    }


def _fit_models(
    records: list[dict[str, Any]],
    output_dir: Path,
    *,
    maximum_epochs: int = 100,
    batch_size: int = 32,
    device: str = "cuda",
) -> dict[str, Any]:
    class_index = {label: index for index, label in enumerate(CLASSES)}
    by_split = {name: [row for row in records if row["split"] == name] for name in PARTITIONS}
    counts = {name: Counter(row["label"] for row in rows) for name, rows in by_split.items()}
    if any(not rows for rows in by_split.values()):
        raise RuntimeError(f"A development partition has no matched candidates: {counts}")
    if any(any(counts[name].get(label, 0) == 0 for label in CLASSES) for name in PARTITIONS):
        raise RuntimeError(f"Both labels must have matched candidates in all development partitions: {counts}")

    x = {
        name: np.stack([np.asarray(row["sequence"], dtype=np.float32) for row in rows])
        for name, rows in by_split.items()
    }
    y = {
        name: np.asarray([class_index[row["label"]] for row in rows], dtype=np.int64)
        for name, rows in by_split.items()
    }
    groups = {name: [row["source_group_id"] for row in rows] for name, rows in by_split.items()}
    if any(len(group_list) != len(set(group_list)) for group_list in groups.values()):
        raise RuntimeError("Candidate set must contain at most one observation per TIC")
    if any(set(groups[left]) & set(groups[right]) for i, left in enumerate(PARTITIONS) for right in PARTITIONS[i + 1:]):
        raise RuntimeError("TIC source group leaked between development partitions")

    # Frozen same-candidate tabular baseline: random forest over the same 3x256 sequence bins.
    x_rf = {name: values.reshape(len(values), -1) for name, values in x.items()}
    baseline = RandomForestClassifier(
        n_estimators=300,
        max_depth=14,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=SEED,
        n_jobs=1,
    )
    baseline.fit(x_rf["train"], y["train"])
    baseline_selection = _grouped_scores(y["selection"].tolist(), baseline.predict_proba(x_rf["selection"]))
    baseline_path = output_dir / "same_candidate_random_forest.joblib"
    joblib.dump(baseline, baseline_path)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    model = CandidateSequenceCNN(class_count=len(CLASSES)).to(device)
    y_train = torch.tensor(y["train"], dtype=torch.long)
    x_train = torch.from_numpy(x["train"])
    x_cal = torch.from_numpy(x["calibration"])
    y_cal = torch.tensor(y["calibration"], dtype=torch.long)
    x_sel = torch.from_numpy(x["selection"]).to(device)
    train_support = torch.bincount(y_train, minlength=len(CLASSES)).float()
    class_weights = torch.sqrt(train_support.sum() / train_support.clamp_min(1.0))
    class_weights /= class_weights.mean()
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True, generator=generator)

    best_score = -1.0
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, maximum_epochs + 1):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x.to(device)), batch_y.to(device))
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            logits = model(x_sel).cpu().numpy()
        score = float(f1_score(y["selection"], logits.argmax(axis=1), average="macro", zero_division=0))
        history.append({"epoch": float(epoch), "selection_macro_f1": score})
        LOGGER.info("epoch %d/%d selection macro-F1=%.4f", epoch, maximum_epochs, score)
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= 15:
                break
    if best_state is None:
        raise RuntimeError("CNN produced no selectable checkpoint")
    model.load_state_dict(best_state)
    model.to(device).eval()
    with torch.no_grad():
        calibration_logits = model(x_cal.to(device)).cpu()
        selection_logits = model(x_sel).cpu()
    temperature = _temperature(calibration_logits, y_cal)
    calibrated_selection_logits = selection_logits.numpy() / temperature

    state_path = output_dir / "sequence_model_state.pt"
    torch.save(best_state, state_path)
    report = {
        "model_id": "labeled_candidate_sequence_cnn_binary_v1",
        "training_source": "fresh_initialization_no_pretrained_weights",
        "class_order": list(CLASSES),
        "partition_candidate_counts": {name: len(rows) for name, rows in by_split.items()},
        "partition_class_counts": {name: dict(counts[name]) for name in PARTITIONS},
        "baseline": {
            "model": "RandomForestClassifier on the same 768 sequence-bin features",
            "selection": baseline_selection,
            "artifact_sha256": sha256_file(baseline_path),
        },
        "cnn": {
            "architecture": "CandidateSequenceCNN(class_count=2)",
            "best_epoch_selected_on_selection": best_epoch,
            "selection_macro_f1_at_selected_epoch": best_score,
            "selection_after_temperature_scaling": _grouped_scores(y["selection"].tolist(), calibrated_selection_logits),
            "calibration_temperature": temperature,
            "artifact_sha256": sha256_file(state_path),
        },
        "training_history": history,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "device": device,
        "seed": SEED,
        "test_partitions_opened": False,
        "fresh_holdout_opened": False,
        "independent_performance_claim": False,
        "limitations": [
            "two-class research model only; stellar_variability and other_artifact are untrained",
            "catalog labels and candidate-event matches are proxies, not independent morphology adjudication",
            "selection metrics are for model selection and are not independent performance estimates",
            "unlabeled products and unmatched candidates were excluded, never treated as negatives",
            "candidate detection failures remain in the full extraction accounting",
            "the fresh holdout is sealed and was not used for fitting or selection",
        ],
    }
    return report


def train_labeled_candidate_sequence(
    manifest_path: Path,
    output_dir: Path,
    *,
    workers: int = 4,
    maximum_epochs: int = 100,
    batch_size: int = 32,
    device: str = "cuda",
) -> dict[str, Any]:
    """Extract matched development candidates, baseline, then train the binary CNN."""
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available; explicitly select CPU to continue")
    if not 1 <= workers <= 8 or maximum_epochs < 1 or batch_size < 1:
        raise ValueError("workers must be 1..8; epoch count and batch size must be positive")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest.get("protocol_id") != "labeled_catalog_event_candidate_sequence_binary_v1":
        raise ValueError("Unexpected cohort protocol identity")
    if manifest.get("unlabeled_and_unmatched_are_negatives") is not False:
        raise ValueError("Protocol must keep unlabeled products and unmatched candidates out of negatives")
    products = manifest["products"]
    product_ids = [row["product_id"] for row in products]
    if len(product_ids) != len(set(product_ids)):
        raise ValueError("Frozen cohort contains duplicate product IDs")
    if any(not row.get("event_refs") for row in products):
        raise ValueError("Frozen cohort has product labels without event ephemerides")

    by_group: dict[str, list[dict[str, Any]]] = {}
    for product in products:
        by_group.setdefault(product["source_group_id"], []).append(product)
    tasks: list[dict[str, Any]] = []
    for group, rows in sorted(by_group.items()):
        labels = {row["label"] for row in rows}
        splits = {row["split"] for row in rows}
        if len(labels) != 1 or len(splits) != 1:
            raise ValueError(f"TIC source group has inconsistent labels or partitions: {group}")
        split = next(iter(splits))
        if split == "fresh_holdout":
            continue
        if split not in PARTITIONS:
            raise ValueError(f"Unsupported development partition {split}")
        # Prefer products with more independent event references; stable product ID breaks ties.
        ordered = sorted(rows, key=lambda row: (-len(row["event_refs"]), row["product_id"]))
        tasks.append({
            "source_group_id": group,
            "label": next(iter(labels)),
            "split": split,
            "products": ordered,
        })
    if len(tasks) == 0:
        raise ValueError("No train/calibration/selection source groups remain")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "candidate_checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    extraction_identity = hashlib.sha256(
        (inspect.getsource(_extract_source_group)
         + inspect.getsource(_candidate_matches_event)
         + inspect.getsource(build_candidate_sequence)
         + inspect.getsource(find_candidates)
         + str(BINS)).encode()
    ).hexdigest()
    cache_identity = _digest_json({
        "cohort_manifest_sha256": manifest_hash,
        "extraction_identity_sha256": extraction_identity,
        "sequence_model_source_sha256": sha256_file(Path("src/transiteye/classification/sequence_model.py")),
        "config_sha256": sha256_file(Path("configs/labeled_candidate_sequence.v1.toml")),
    })
    checkpoint_results: dict[str, dict[str, Any]] = {}
    pending: list[dict[str, Any]] = []
    for task in tasks:
        token = hashlib.sha256(task["source_group_id"].encode()).hexdigest()[:24]
        result_path = checkpoint_dir / f"{token}.json"
        if result_path.is_file():
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
                if result.get("cache_identity") == cache_identity:
                    checkpoint_results[task["source_group_id"]] = result["result"]
                    continue
            except (OSError, json.JSONDecodeError, KeyError):
                pass
        pending.append(task)

    LOGGER.info("Extracting %d development source groups (%d cached)", len(pending), len(checkpoint_results))
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
                    "split": task["split"],
                    "status": "worker_failure",
                    "error": f"{type(exc).__name__}: {exc}",
                    "attempts": [],
                }
            token = hashlib.sha256(task["source_group_id"].encode()).hexdigest()[:24]
            result_path = checkpoint_dir / f"{token}.json"
            payload = {"cache_identity": cache_identity, "result": result}
            temporary = result_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(payload, allow_nan=False, sort_keys=True) + "\n", encoding="utf-8")
            temporary.replace(result_path)
            checkpoint_results[task["source_group_id"]] = result
            if completed == 1 or completed % 25 == 0 or completed == len(pending):
                LOGGER.info("Candidate matching %d/%d source groups", completed, len(pending))

    ordered_results = [checkpoint_results[task["source_group_id"]] for task in tasks]
    status_counts = Counter(row["status"] for row in ordered_results)
    usable = [row for row in ordered_results if row["status"] == "matched"]
    result_table = pa.Table.from_pylist([
        {key: value for key, value in row.items() if key != "sequence"}
        for row in ordered_results
    ])
    pq.write_table(result_table, output_dir / "candidate_matching_results.parquet", compression="zstd")
    np.savez_compressed(
        output_dir / "matched_candidate_sequences.npz",
        source_group_id=np.asarray([row["source_group_id"] for row in usable], dtype=str),
        label=np.asarray([row["label"] for row in usable], dtype=str),
        split=np.asarray([row["split"] for row in usable], dtype=str),
        sequence=np.stack([row["sequence"] for row in usable]).astype(np.float32)
        if usable else np.empty((0, 3, BINS), dtype=np.float32),
    )
    matching_summary = {
        "cohort_manifest_sha256": manifest_hash,
        "candidate_match_cache_identity": cache_identity,
        "development_source_group_count": len(tasks),
        "matched_candidate_group_count": len(usable),
        "status_counts": dict(status_counts),
        "matched_by_partition_and_class": {
            partition: dict(Counter(row["label"] for row in usable if row["split"] == partition))
            for partition in PARTITIONS
        },
        "unopened_fresh_holdout_source_count": len({
            row["source_group_id"] for row in products if row["split"] == "fresh_holdout"
        }),
        "fresh_holdout_opened": False,
    }
    (output_dir / "candidate_matching_summary.json").write_text(
        json.dumps(matching_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not usable:
        raise RuntimeError("No development candidates matched catalog events; see candidate_matching_results.parquet")

    report = _fit_models(usable, output_dir, maximum_epochs=maximum_epochs, batch_size=batch_size, device=device)
    report.update({
        "protocol_id": manifest["protocol_id"],
        "cohort_manifest_sha256": manifest_hash,
        "cohort_manifest_id": manifest_hash,
        "candidate_matching_cache_identity": cache_identity,
        "candidate_matching_status_counts": dict(status_counts),
        "development_source_group_count": len(tasks),
        "matched_candidate_group_count": len(usable),
        "unopened_fresh_holdout_source_count": matching_summary["unopened_fresh_holdout_source_count"],
        "candidate_matching_results_sha256": sha256_file(output_dir / "candidate_matching_results.parquet"),
        "matched_sequence_cache_sha256": sha256_file(output_dir / "matched_candidate_sequences.npz"),
        "trainer_source_sha256": sha256_file(Path(__file__)),
        "cohort_builder_source_sha256": sha256_file(Path("src/transiteye/evaluation/labeled_sequence_cohort.py")),
        "config_sha256": sha256_file(Path("configs/labeled_candidate_sequence.v1.toml")),
        "lockfile_sha256": sha256_file(Path("uv.lock")),
    })
    (output_dir / "training_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
