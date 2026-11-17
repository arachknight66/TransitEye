"""Build sequences and train the experimental development-only PyTorch classifier.

The script never opens test-partition FITS files. Catalog labels are source-level
proxies, so the output is exploratory and is not an independently evaluated model.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import logging
import multiprocessing
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from sklearn.metrics import accuracy_score, f1_score
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from transiteye.classification.models import TARGET_MORPHOLOGY_CLASSES
from transiteye.classification.sequence_model import (
    CandidateSequenceCNN,
    build_candidate_sequence,
)
from transiteye.datasets.labels import read_labels
from transiteye.datasets.splits import (
    DatasetSplit,
    read_split_manifest,
    validate_labels_against_split,
)
from transiteye.detection.peaks import find_candidates
from transiteye.io.fits import read_tess_light_curve
from transiteye.io.manifest import read_manifest
from transiteye.preprocessing.detrend import two_pass_detrend
from transiteye.preprocessing.representation import prepare_light_curve

LOGGER = logging.getLogger("transiteye.train_sequence_model")
SEED = 42
SEQUENCE_BINS = 256
SEQUENCE_CLASSES = tuple(item.value for item in TARGET_MORPHOLOGY_CLASSES)


def _source_sequence(
    task: tuple[str, str, str | None, float],
) -> tuple[str, np.ndarray | None, str | None]:
    """Extract one candidate-centered sequence with a CPU-only scientific path."""
    target_id, path_text, expected_sha256, _label_index = task
    path = Path(path_text)
    try:
        if expected_sha256 and hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
            return target_id, None, "product checksum mismatch"
        light_curve = read_tess_light_curve(path)
        prepared = prepare_light_curve(light_curve)
        detrended = two_pass_detrend(prepared, baseline_window_days=0.75, device="cpu")
        detection = find_candidates(
            detrended, max_candidates=1, snr_threshold=1.5, device="cpu"
        )
        if not detection.candidates:
            detection = find_candidates(
                detrended, max_candidates=1, snr_threshold=0.5, device="cpu"
            )
        if not detection.candidates:
            return target_id, None, "no candidate above fallback detector threshold"
        candidate = detection.candidates[0]
        if (
            candidate.period_days is None
            or candidate.epoch_days is None
            or candidate.duration_hours is None
        ):
            return target_id, None, "candidate missing period, epoch, or duration"
        sequence = build_candidate_sequence(
            prepared.time_days,
            prepared.flux_normalized,
            detrended.detrended_flux,
            candidate.epoch_days,
            candidate.period_days,
            candidate.duration_hours,
            bins=SEQUENCE_BINS,
        )
        return target_id, sequence, None
    except Exception as exc:
        return target_id, None, f"sequence extraction error: {type(exc).__name__}: {exc}"


def _temperature_scale(logits: Tensor, labels: Tensor) -> float:
    """Fit one positive scalar temperature on the calibration partition only."""
    logits = logits.detach().cpu()
    labels = labels.detach().cpu()

    def objective(log_temperature: float) -> float:
        temperature = float(np.exp(log_temperature))
        return float(nn.functional.cross_entropy(logits / temperature, labels).item())

    result = minimize_scalar(objective, bounds=(-3.0, 3.0), method="bounded")
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError("Temperature calibration did not converge.")
    return float(np.exp(result.x))


def _source_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def train_sequence_model(
    data_dir: Path,
    products_path: Path,
    labels_path: Path,
    splits_path: Path,
    output_dir: Path,
    workers: int = 4,
    maximum_epochs: int = 100,
    batch_size: int = 32,
    device: str = "cuda",
    baseline_report_path: Path = Path(
        ".transiteye-workspace/models/morphology_fresh_baseline_v1/training_report.json"
    ),
) -> dict[str, Any]:
    """Extract development sequences, fit from scratch, calibrate, and save an audit report."""
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; select --device cpu explicitly.")
    if workers < 1 or maximum_epochs < 1 or batch_size < 1:
        raise ValueError("workers, maximum_epochs, and batch_size must be positive.")
    if not baseline_report_path.is_file():
        raise RuntimeError("A completed fresh baseline report is required before neural training.")
    baseline_report = json.loads(baseline_report_path.read_text(encoding="utf-8"))
    if baseline_report.get("test_split_opened") is not False:
        raise RuntimeError("The baseline must confirm that the test partition remained unopened.")
    if baseline_report.get("training_manifest_id") is None:
        raise RuntimeError("The baseline report is missing its split-manifest identity.")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    labels = read_labels(labels_path)
    splits = read_split_manifest(splits_path)
    validate_labels_against_split(labels, splits)
    split_map = {item.source_group_id: item.split for item in splits.assignments}
    class_index = {name: index for index, name in enumerate(SEQUENCE_CLASSES)}
    products = read_manifest(products_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    product_by_target = {
        product.target_id: product for product in products.entries
    }

    tasks: list[tuple[str, str, str | None, float]] = []
    task_records: list[tuple[Any, DatasetSplit]] = []
    failures: list[dict[str, str]] = []
    total_development = 0
    for label in labels:
        split = split_map[label.source_group_id]
        if split is DatasetSplit.TEST:
            continue
        total_development += 1
        if split not in (DatasetSplit.TRAIN, DatasetSplit.CALIBRATION, DatasetSplit.SELECTION):
            failures.append({"target_id": label.target_id, "reason": f"unsupported split {split.value}"})
            continue
        product = product_by_target.get(label.target_id)
        if product is None:
            failures.append({"target_id": label.target_id, "reason": "no product manifest entry"})
            continue
        path = data_dir / product.resolved_filename()
        if not path.is_file():
            failures.append({"target_id": label.target_id, "reason": "FITS file missing"})
            continue
        target_class = label.label.value
        if target_class not in class_index:
            failures.append({"target_id": label.target_id, "reason": f"unsupported label {target_class}"})
            continue
        tasks.append((label.target_id, str(path), product.expected_sha256, float(class_index[target_class])))
        task_records.append((label, split))

    arrays: dict[DatasetSplit, list[np.ndarray]] = {
        part: [] for part in (DatasetSplit.TRAIN, DatasetSplit.CALIBRATION, DatasetSplit.SELECTION)
    }
    targets: dict[DatasetSplit, list[int]] = {part: [] for part in arrays}
    groups: dict[DatasetSplit, list[str]] = {part: [] for part in arrays}
    cache_path = output_dir / "development_sequences.npz"
    cache_metadata_path = output_dir / "development_sequences.json"
    extractor_identity = hashlib.sha256(
        (
            inspect.getsource(_source_sequence)
            + inspect.getsource(build_candidate_sequence)
            + str(SEQUENCE_BINS)
        ).encode("utf-8")
    ).hexdigest()
    cache_identity = hashlib.sha256(
        json.dumps(
            {
                "product_manifest_id": products.manifest_id,
                "split_manifest_id": splits.manifest_id,
                "extractor_sha256": extractor_identity,
                "sequence_protocol_sha256": _source_hash(
                    Path("configs/sequence_model_development.v3.toml")
                ),
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    cache_metadata: dict[str, Any] | None = None
    if cache_path.is_file() and cache_metadata_path.is_file():
        try:
            existing_metadata = json.loads(cache_metadata_path.read_text(encoding="utf-8"))
            if (
                existing_metadata.get("cache_identity") == cache_identity
                and existing_metadata.get("cache_sha256") == _source_hash(cache_path)
            ):
                with np.load(cache_path, allow_pickle=False) as cached:
                    for name, split in (
                        ("train", DatasetSplit.TRAIN),
                        ("calibration", DatasetSplit.CALIBRATION),
                        ("selection", DatasetSplit.SELECTION),
                    ):
                        arrays[split] = list(cached[f"{name}_x"].copy())
                        targets[split] = cached[f"{name}_y"].astype(np.int64).tolist()
                        groups[split] = cached[f"{name}_groups"].astype(str).tolist()
                failures = existing_metadata.get("failures", [])
                cache_metadata = existing_metadata
                LOGGER.info("Loaded verified development sequence cache with %d rows", sum(map(len, arrays.values())))
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            LOGGER.warning("Sequence cache is invalid; rebuilding from FITS inputs.")

    if cache_metadata is None:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            for index, (target_id, sequence, error) in enumerate(pool.map(_source_sequence, tasks), start=1):
                if index == 1 or index % 10 == 0 or index == len(tasks):
                    LOGGER.info("Extracted sequences %d/%d (TIC %s)", index, len(tasks), target_id)
                if error is not None or sequence is None:
                    failures.append({"target_id": target_id, "reason": error or "sequence unavailable"})
                    continue
                label, split = task_records[index - 1]
                arrays[split].append(sequence)
                targets[split].append(class_index[label.label.value])
                groups[split].append(label.source_group_id)

        cache_payload: dict[str, np.ndarray] = {}
        for name, split in (
            ("train", DatasetSplit.TRAIN),
            ("calibration", DatasetSplit.CALIBRATION),
            ("selection", DatasetSplit.SELECTION),
        ):
            cache_payload[f"{name}_x"] = np.stack(arrays[split]).astype(np.float32)
            cache_payload[f"{name}_y"] = np.asarray(targets[split], dtype=np.int64)
            cache_payload[f"{name}_groups"] = np.asarray(groups[split], dtype=str)
        temporary_cache = cache_path.with_name(f"{cache_path.name}.tmp.npz")
        np.savez_compressed(temporary_cache, **cache_payload)
        temporary_cache.replace(cache_path)
        cache_metadata = {
            "cache_identity": cache_identity,
            "cache_sha256": _source_hash(cache_path),
            "failures": failures,
            "sequence_count": sum(map(len, arrays.values())),
        }
        cache_metadata_path.write_text(
            json.dumps(cache_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    counts = {
        split.value: Counter(SEQUENCE_CLASSES[index] for index in targets[split])
        for split in arrays
    }
    if any(not arrays[split] for split in arrays):
        raise RuntimeError(f"A development partition has no usable sequences: {counts}")
    if any(counts[split.value].get(name, 0) == 0 for split in arrays for name in SEQUENCE_CLASSES):
        raise RuntimeError(f"Every development partition must retain all classes: {counts}")
    if len(groups[DatasetSplit.TRAIN]) != len(set(groups[DatasetSplit.TRAIN])):
        raise RuntimeError("Training contains duplicate source groups.")
    group_sets = {split: set(values) for split, values in groups.items()}
    if any(
        group_sets[left].intersection(group_sets[right])
        for left, right in (
            (DatasetSplit.TRAIN, DatasetSplit.CALIBRATION),
            (DatasetSplit.TRAIN, DatasetSplit.SELECTION),
            (DatasetSplit.CALIBRATION, DatasetSplit.SELECTION),
        )
    ):
        raise RuntimeError("A source group appears in multiple model partitions.")

    x_train = torch.from_numpy(np.stack(arrays[DatasetSplit.TRAIN])).to(torch.float32)
    y_train = torch.tensor(targets[DatasetSplit.TRAIN], dtype=torch.long)
    x_cal = torch.from_numpy(np.stack(arrays[DatasetSplit.CALIBRATION])).to(torch.float32)
    y_cal = torch.tensor(targets[DatasetSplit.CALIBRATION], dtype=torch.long)
    x_sel = torch.from_numpy(np.stack(arrays[DatasetSplit.SELECTION])).to(torch.float32)
    y_sel = torch.tensor(targets[DatasetSplit.SELECTION], dtype=torch.long)

    model = CandidateSequenceCNN(class_count=len(SEQUENCE_CLASSES)).to(device)
    train_support = torch.bincount(y_train, minlength=len(SEQUENCE_CLASSES)).float()
    class_weights = torch.sqrt(train_support.sum() / train_support.clamp_min(1.0))
    class_weights /= class_weights.mean()
    loss_function = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True, generator=generator
    )
    x_sel_device = x_sel.to(device)
    best_score = -1.0
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    patience = 15
    stale_epochs = 0

    for epoch in range(1, maximum_epochs + 1):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x.to(device))
            loss = loss_function(logits, batch_y.to(device))
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            selected_logits = model(x_sel_device).cpu()
        selected_predictions = selected_logits.argmax(dim=1).numpy()
        score = float(f1_score(y_sel.numpy(), selected_predictions, average="macro", zero_division=0))
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is None:
        raise RuntimeError("Training produced no selectable checkpoint.")
    model.load_state_dict(best_state)
    model.to(device).eval()
    with torch.no_grad():
        calibration_logits = model(x_cal.to(device)).cpu()
        selection_logits = model(x_sel_device).cpu()
    temperature = _temperature_scale(calibration_logits, y_cal)
    selection_probabilities = torch.softmax(selection_logits / temperature, dim=1).numpy()
    selection_predictions = selection_probabilities.argmax(axis=1)

    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "sequence_model_state.pt"
    torch.save(best_state, state_path)
    report: dict[str, Any] = {
        "model_id": "candidate_sequence_cnn_v3",
        "training_source": "fresh_initialization_no_pretrained_weights",
        "dataset_manifest_id": products.manifest_id,
        "split_manifest_id": splits.manifest_id,
        "product_manifest_sha256": _source_hash(products_path),
        "split_manifest_sha256": _source_hash(splits_path),
        "label_manifest_sha256": _source_hash(labels_path),
        "config_sha256": _source_hash(Path("configs/sequence_model_development.v3.toml")),
        "sequence_cache_sha256": _source_hash(cache_path),
        "source_sha256": _source_hash(Path("src/transiteye/classification/sequence_model.py")),
        "state_sha256": _source_hash(state_path),
        "trainer_sha256": _source_hash(Path("scripts/train_sequence_model.py")),
        "lockfile_sha256": _source_hash(Path("uv.lock")),
        "baseline_training_report_sha256": _source_hash(baseline_report_path),
        "baseline_config_sha256": _source_hash(Path("configs/model_baseline_development.v1.toml")),
        "baseline_model_id": baseline_report.get("model_id"),
        "baseline_cv_comparison": baseline_report.get("cv_comparison"),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "numpy_version": np.__version__,
        "device": str(device),
        "seed": SEED,
        "input_shape": [3, SEQUENCE_BINS],
        "class_order": list(SEQUENCE_CLASSES),
        "source_counts": {split: dict(value) for split, value in counts.items()},
        "development_source_count": total_development,
        "sequence_count": sum(len(values) for values in arrays.values()),
        "fit_failures": failures,
        "fit_failure_count": len(failures),
        "best_epoch_selected_on_selection_partition": best_epoch,
        "selection_macro_f1_at_selected_epoch": best_score,
        "selection_accuracy_after_temperature_scaling": float(accuracy_score(targets[DatasetSplit.SELECTION], selection_predictions)),
        "selection_macro_f1_after_temperature_scaling": float(f1_score(targets[DatasetSplit.SELECTION], selection_predictions, average="macro", zero_division=0)),
        "calibration_temperature": temperature,
        "test_partitions_opened": False,
        "label_scope": "catalog-derived source-level proxies; candidate-local morphology is unverified",
        "independent_performance_claim": False,
    }
    (output_dir / "training_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(".transiteye-workspace/data/real_catalog_cohort_v3_dev"))
    parser.add_argument("--products", type=Path, default=Path(".transiteye-workspace/data/real_catalog_cohort_v3_dev_products.v1.json"))
    parser.add_argument("--labels", type=Path, default=Path("manifests/real_catalog_cohort.v3.labels.json"))
    parser.add_argument("--splits", type=Path, default=Path("manifests/real_catalog_cohort.v3.splits.json"))
    parser.add_argument("--output-dir", type=Path, default=Path(".transiteye-workspace/models/candidate_sequence_cnn_v3"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--maximum-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--baseline-report",
        type=Path,
        default=Path(".transiteye-workspace/models/morphology_fresh_baseline_v1/training_report.json"),
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    report = train_sequence_model(
        args.data_dir, args.products, args.labels, args.splits, args.output_dir,
        args.workers, args.maximum_epochs, args.batch_size, args.device,
        args.baseline_report,
    )
    print(json.dumps({key: value for key, value in report.items() if key != "fit_failures"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
