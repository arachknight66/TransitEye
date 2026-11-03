"""Run a frozen, checksum-verified development benchmark outside the repository."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
import tomllib
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from transiteye.config import AcquisitionConfig, AppConfig, WorkspaceConfig
from transiteye.domain.models import AnalysisSettings
from transiteye.execution.benchmark import BenchmarkReport, run_sector_benchmark
from transiteye.io.manifest import read_manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(config_path: Path, manifest_path: Path, output_dir: Path) -> BenchmarkReport:
    """Preflight every frozen input, then run the resumable CPU development workload."""
    config = tomllib.loads(config_path.read_text())
    if config["schema_version"] != "1" or config["device"] != "cpu":
        raise ValueError("unsupported benchmark configuration")
    manifest = read_manifest(manifest_path)
    split_path, product_path = Path(config["split_manifest"]), Path(config["product_manifest"])
    expected = {"benchmark_config_sha256": _sha256(config_path), "split_manifest_sha256": _sha256(split_path),
                "product_manifest_sha256": _sha256(product_path)}
    if any(manifest.query.get(key) != value for key, value in expected.items()):
        raise ValueError("frozen benchmark configuration or source manifest has changed")
    assignments = json.loads(split_path.read_text())["assignments"]
    train = {row["source_group_id"] for row in assignments if row["split"] == "train"}
    if len(manifest.entries) != config["target_count"] or len({entry.target_id for entry in manifest.entries}) != len(manifest.entries):
        raise ValueError("benchmark manifest does not contain the required distinct sources")
    if any(f"tic-{entry.target_id}" not in train for entry in manifest.entries):
        raise ValueError("benchmark manifest contains a non-TRAIN source")
    root = Path(config["data_directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "development_100_v1_report.json").exists():
        raise ValueError("benchmark output already has a final report; choose a new output directory")
    preflight_started = time.perf_counter()
    audit: list[dict[str, Any]] = []
    for entry in manifest.entries:
        path = root / entry.resolved_filename()
        row = {"target_id": entry.target_id, "product_id": entry.product_id,
               "local_filename": entry.resolved_filename(), "expected_sha256": entry.expected_sha256,
               "actual_sha256": None, "error": None}
        try:
            if not entry.expected_sha256:
                raise ValueError("missing expected checksum")
            row["actual_sha256"] = _sha256(path)
            if row["actual_sha256"] != entry.expected_sha256:
                raise ValueError("input checksum mismatch")
        except (OSError, ValueError) as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        audit.append(row)
    preflight_seconds = time.perf_counter() - preflight_started
    (output_dir / "preflight.json").write_text(json.dumps({"inputs": audit,
        "failures": sum(row["error"] is not None for row in audit), "elapsed_seconds": preflight_seconds},
        indent=2, sort_keys=True) + "\n")
    if any(row["error"] is not None for row in audit):
        raise ValueError("benchmark preflight failed; see preflight.json for every input")
    settings = AnalysisSettings(**config["analysis"], device=config["device"])
    workspace = output_dir / "workspace"
    app_config = AppConfig(version="1", workspace=WorkspaceConfig(
        root=workspace, cache_directory=workspace / "cache", results_directory=workspace / "results"),
        acquisition=AcquisitionConfig())
    job_config = {**asdict(settings), "device": config["device"], "flux": "PDCSAP_FLUX",
                  "model_dir": None, "exposure_seconds": None,
                  "target_timeout_seconds": config["target_timeout_seconds"],
                  "expected_checksums": {entry.product_id: entry.expected_sha256 for entry in manifest.entries}}
    targets = [(entry.product_id, (root / entry.resolved_filename()).resolve().as_uri()) for entry in manifest.entries]
    context = {"config_sha256": _sha256(config_path), "manifest_sha256": _sha256(manifest_path),
               "manifest_id": manifest.manifest_id, "split_manifest_sha256": expected["split_manifest_sha256"],
               "product_manifest_sha256": expected["product_manifest_sha256"],
               "preflight_seconds": preflight_seconds, "model_id": None,
               "python_version": platform.python_version(), "platform": platform.platform(),
               "transiteye_version": version("transiteye"), "numpy_version": np.__version__,
               "scipy_version": version("scipy"), "batman_version": version("batman-package"),
               "thread_environment": {key: os.environ.get(key) for key in
                                      ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
               "limitations": "Development TRAIN sample, single candidate, no classifier; exposure is assumed."}
    return run_sector_benchmark(targets, output_dir, device=config["device"], workers=config["workers"],
        benchmark_id="development_100_v1", job_config=job_config, app_config=app_config,
        pause_after_finished=config["pause_after_finished"], rss_sample_seconds=config["rss_sample_seconds"],
        report_context=context)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(asdict(run(args.config, args.manifest, args.output_dir)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
