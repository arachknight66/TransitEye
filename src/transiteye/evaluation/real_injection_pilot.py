"""Development-only injections into frozen TRAIN light curves, with unchanged controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import tomllib
from dataclasses import replace
from importlib.metadata import version
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from transiteye.datasets.splits import DatasetSplit
from transiteye.domain.models import AnalysisSettings, LightCurve, as_jsonable
from transiteye.evaluation.injection_pilot import (
    _write_json,
    injection_flux,
    matches_truth,
    summarize,
)
from transiteye.execution.service import analyze_light_curve
from transiteye.io.fits import read_tess_light_curve
from transiteye.provenance import code_fingerprint
from transiteye.reporting.bundles import verify_bundle_integrity

logger = logging.getLogger(__name__)


def _assumed_settings(config: dict[str, Any]) -> AnalysisSettings:
    """Use the recorded unverified fallback, not an asserted measured exposure."""
    if float(config["exposure_minutes_assumed"]) != 2.0:
        raise ValueError("real-substrate pilot currently supports only the unverified two-minute fallback")
    return AnalysisSettings(**config["analysis"])


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_sources(config_path: Path) -> dict[str, Any]:
    """Select by TRAIN split and product availability, without reading labels or flux."""
    config = tomllib.loads(config_path.read_text())
    splits, products = Path(config["split_manifest"]), Path(config["product_manifest"])
    assignments = json.loads(splits.read_text())["assignments"]
    entries = json.loads(products.read_text())["entries"]
    root = Path(config["data_directory"])
    train = sorted((item["source_group_id"] for item in assignments if item["split"] == "train"),
                   key=lambda group: int(group.removeprefix("tic-")))[:config["source_count"]]
    sources = []
    for group in train:
        target = group.removeprefix("tic-")
        available = [item for item in entries if item["target_id"] == target
                     and (root / item["local_filename"]).is_file()]
        chosen = max(available, key=lambda item: (item["sector"], item["product_id"])) if available else None
        sources.append({"source_group_id": group, "split": "train", "target_id": target,
                        "product": chosen, "status": "selected" if chosen else "missing_download"})
    return {"config_sha256": _sha256(config_path), "split_sha256": _sha256(splits),
            "product_manifest_sha256": _sha256(products), "selection_rule": config["source_selection"],
            "sources": sources}


def make_injected_curve(
    substrate: LightCurve, source_group: str, split: DatasetSplit, trial_id: str, seed: int, config: dict[str, Any],
    period: float | None, depth: float | None,
) -> tuple[LightCurve, float | None]:
    """Preserve input cadences, masks, errors, units and parent provenance for a trial."""
    if split is not DatasetSplit.TRAIN:
        raise ValueError("real-substrate injection is restricted to TRAIN sources")
    rng = np.random.default_rng(seed)
    finite_time = substrate.time_days[np.isfinite(substrate.time_days)]
    if len(finite_time) == 0:
        raise ValueError("substrate has no finite times")
    epoch = None if period is None else float(np.min(finite_time) + rng.uniform(0, period))
    flux = substrate.flux.copy()
    if period is not None and depth is not None and epoch is not None:
        dip = injection_flux(substrate.time_days, period, epoch, config["duration_hours"], depth,
                             config["ingress_fraction"], config["exposure_minutes_assumed"] / 1440,
                             config["injection_integration_points"])
        flux *= 1.0 - dip
    digest = hashlib.sha256(substrate.time_days.tobytes() + flux.tobytes()).hexdigest()
    parent = substrate.provenance
    provenance = replace(parent, source_uri=f"synthetic://real-substrate-pilot/{trial_id}", source_path=None,
                         checksum_sha256=digest, product_type="synthetic_tess_light_curve",
                         target_id=trial_id, metadata={**parent.metadata,
                             "parent_target_id": parent.target_id or "unknown",
                             "parent_source_path": parent.source_path or "unknown",
                             "parent_source_uri": parent.source_uri,
                             "parent_input_sha256": parent.checksum_sha256 or "unknown",
                             "parent_source_group_id": source_group, "parent_split": "train",
                             "trial_seed": str(seed)})
    return replace(substrate, flux=flux, provenance=provenance), epoch


def _row(config: dict[str, Any], source: dict[str, Any], substrate: LightCurve, trial_id: str,
         period: float | None, depth: float | None, settings: AnalysisSettings,
         bundle: Path) -> dict[str, Any]:
    seed = int.from_bytes(hashlib.sha256(f"{config['seed']}:{trial_id}".encode()).digest()[:8], "big")
    curve, epoch = make_injected_curve(substrate, source["source_group_id"], DatasetSplit.TRAIN,
                                       trial_id, seed, config, period, depth)
    row: dict[str, Any] = {
        "trial_id": trial_id, "seed": seed, "source_group_id": source["source_group_id"],
        "parent_input_sha256": substrate.provenance.checksum_sha256,
        "input_sha256": curve.provenance.checksum_sha256,
        "kind": "null" if period is None else "injection", "truth_period_days": period,
        "truth_epoch_days": epoch, "truth_depth_ppm": depth,
        "truth_duration_hours": None if period is None else config["duration_hours"],
        "status": "error", "error": None, "detected": False, "fit_failed": False,
        "physical_fit_failed": False, "fit_error": None, "matched": False, "harmonic": False,
    }
    for name in ("period_days", "depth_ppm", "duration_hours"):
        row[f"error_{name}"] = None
        for level in ("68", "95"):
            row[f"covered_{name}_{level}"] = None
    try:
        result, _ = analyze_light_curve(curve, analysis_settings=settings, output_dir=bundle)
        row["status"], row["error"] = result.status.value, result.error_message
        if result.candidates:
            candidate = result.candidates[0]
            row["detected"] = True
            row["fit_failed"] = "fit_failed" in candidate.flags
            row["physical_fit_failed"] = "physical_fit_failed" in candidate.flags
            row["fit_error"] = result.candidate_errors.get(candidate.candidate_id)
            if period is not None and epoch is not None and candidate.period_days and candidate.epoch_days is not None:
                row["harmonic"] = any(abs(candidate.period_days / (period * factor) - 1) <= config["match_period_fraction"]
                                      for factor in (0.5, 2.0))
                row["matched"] = not row["fit_failed"] and matches_truth(
                    candidate.period_days, candidate.epoch_days, period, epoch,
                    config["match_period_fraction"], config["match_epoch_hours"])
                if row["matched"]:
                    truth = {"period_days": period, "depth_ppm": depth, "duration_hours": config["duration_hours"]}
                    measured = {"period_days": candidate.period_days, "depth_ppm": candidate.depth_ppm,
                                "duration_hours": candidate.duration_hours}
                    intervals = result.parameter_intervals.get(candidate.candidate_id, {})
                    for name, value in truth.items():
                        if value is not None and measured[name] is not None:
                            row[f"error_{name}"] = float(measured[name] - value)
                            for level in ("68", "95"):
                                interval = intervals.get(name, {}).get(level)
                                if interval is not None and interval.lower is not None and interval.upper is not None:
                                    row[f"covered_{name}_{level}"] = bool(interval.lower <= value <= interval.upper)
    except Exception as exc:
        logger.exception("Real-substrate pilot trial %s failed", trial_id)
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def run(config_path: Path, output_dir: Path) -> dict[str, Any]:
    """Run the frozen selection with resumable per-trial checkpoints."""
    config = tomllib.loads(config_path.read_text())
    if config["schema_version"] != "1":
        raise ValueError("unsupported pilot configuration")
    selection = select_sources(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    selection_path = output_dir / "selection.json"
    if selection_path.exists():
        if json.loads(selection_path.read_text()) != selection:
            raise ValueError("frozen source selection or input manifests changed")
    else:
        _write_json(selection_path, selection)
    settings = _assumed_settings(config)
    identity = {"config_sha256": _sha256(config_path), "selection_sha256": _sha256(selection_path),
                "backend_sha256": code_fingerprint(), "thread_environment": {
                    key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text())
        if any(previous.get(key) != value for key, value in identity.items()):
            raise ValueError("existing pilot directory has different inputs, code or thread environment")
    else:
        _write_json(metadata_path, {**identity, "schema_version": "1", "config": config,
                    "settings": as_jsonable(settings), "python_version": platform.python_version(),
                    "transiteye_version": version("transiteye"), "numpy_version": np.__version__,
                    "scipy_version": version("scipy"), "batman_version": version("batman-package"),
                    "limitations": "Four TRAIN sources, ideal trapezoids, assumed exposure; controls may contain real signals."})
    input_audit = []
    for source in selection["sources"]:
        entry: dict[str, Any] = {"source_group_id": source["source_group_id"], "error": None}
        try:
            if source["product"] is None:
                raise ValueError("no downloaded product")
            source_path = Path(config["data_directory"]) / source["product"]["local_filename"]
            if _sha256(source_path) != source["product"]["expected_sha256"]:
                raise ValueError("input checksum mismatch")
        except (OSError, ValueError) as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
        input_audit.append(entry)
    _write_json(output_dir / "input_audit.json", {"sources": input_audit,
                "failures": sum(entry["error"] is not None for entry in input_audit)})
    if any(entry["error"] is not None for entry in input_audit):
        raise ValueError("one or more selected sources failed preflight; see input_audit.json")
    rows = []
    for source in selection["sources"]:
        if source["product"] is None:
            raise ValueError(f"selected TRAIN source {source['source_group_id']} has no downloaded product")
        source_path = Path(config["data_directory"]) / source["product"]["local_filename"]
        if _sha256(source_path) != source["product"]["expected_sha256"]:
            raise ValueError(f"input checksum mismatch for {source['source_group_id']}")
        substrate = read_tess_light_curve(source_path)
        variants = [(f"{source['target_id']}_p{pi}_d{di}", float(period), float(depth))
                    for (pi, period), (di, depth) in product(enumerate(config["periods_days"]),
                                                              enumerate(config["depths_ppm"]))]
        variants.append((f"{source['target_id']}_control", None, None))
        for trial_id, period, depth in variants:
            trial_dir = output_dir / "trials" / trial_id
            trial_dir.mkdir(parents=True, exist_ok=True)
            checkpoint, bundle = trial_dir / "trial.json", trial_dir / "bundle"
            row = None
            if checkpoint.exists():
                previous = json.loads(checkpoint.read_text())
                if previous["status"] == "error" or verify_bundle_integrity(bundle):
                    row = previous
            if row is None:
                row = _row(config, source, substrate, trial_id, period, depth, settings, bundle)
                _write_json(checkpoint, row)
            rows.append(row)
            logger.info("Real-substrate pilot %d/%d: %s (%s)", len(rows), len(selection["sources"]) * len(variants),
                        trial_id, row["status"])
    table = pa.Table.from_pylist([{key: value for key, value in row.items() if key != "seed"} for row in rows])
    table = table.append_column("seed", pa.array([row["seed"] for row in rows], type=pa.uint64()))
    pq.write_table(table, output_dir / "trials.parquet")
    aggregate = summarize(rows)
    aggregate["control_trials"] = aggregate.pop("null_trials")
    aggregate["control_candidates"] = aggregate.pop("null_alerts")
    aggregate["control_usable"] = aggregate.pop("null_usable")
    summary = {**identity, **aggregate}
    _write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(run(args.config, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
