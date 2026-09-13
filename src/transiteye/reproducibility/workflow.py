"""B060 thin offline orchestration over existing TransitEye modules."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from transiteye.config import load_config
from transiteye.detection.bls import run_bls
from transiteye.detection.peaks import extract_peaks
from transiteye.modeling.artifacts import load_frozen_model
from transiteye.modeling.inference import predict_candidates
from transiteye.preprocessing.pipeline import LightCurveRecord, preprocess
from transiteye.serialization import canonical_json

from .manifest import (
    DEMO_DATASET,
    DEMO_FEATURES,
    MODEL,
    current_manifest_directory,
    verify_reproducibility_manifest,
)


def verify_mode(repository: Path) -> dict[str, Any]:
    return verify_reproducibility_manifest(repository, current_manifest_directory(repository))


def report_mode(repository: Path) -> Path:
    from .reporting import generate_final_results

    manifest = current_manifest_directory(repository)
    before = verify_reproducibility_manifest(repository, manifest)
    if before["status"] != "verified":
        raise ValueError("Frozen artifacts failed verification before report generation.")
    result = generate_final_results(repository, manifest)
    after = verify_reproducibility_manifest(repository, manifest)
    if after != before:
        raise ValueError("Report generation changed a registered frozen artifact.")
    return result


def demo_smoke_mode(repository: Path) -> dict[str, Any]:
    """Exercise preprocessing, blind BLS, peak extraction, and frozen inference offline."""
    verification = verify_mode(repository)
    if verification["status"] != "verified":
        raise ValueError("Frozen artifacts failed verification before demo smoke.")
    time = np.arange(0.0, 10.0, 0.02)
    phase = ((time - 0.3 + 1.25) % 2.5) - 1.25
    flux = np.ones_like(time)
    flux[np.abs(phase) < 0.06] -= 0.015
    record = LightCurveRecord(
        pd.DataFrame(
            {
                "cadence_row": np.arange(len(time)),
                "time": time,
                "pdcsap_flux": flux,
                "pdcsap_flux_err": np.full(len(time), 0.001),
                "sap_flux": flux,
                "sap_flux_err": np.full(len(time), 0.001),
                "quality": np.zeros(len(time), dtype=int),
            }
        ),
        {"fixture": "deterministic_box_transit"},
    )
    processed = preprocess(record, gap_days=0.5, trend_window=51, positive_spike_mad=8.0)
    config = load_config(repository / "configs/bls/mvp.yaml")
    if config.bls is None:
        raise ValueError("BLS configuration is unavailable.")
    bls = run_bls(
        processed.data,
        settings=config.bls,
        observation_group_id="demo-smoke-group",
        preprocessing_hash="demo-smoke-preprocess-v1",
    )
    candidates = extract_peaks(bls, config.bls)
    model = load_frozen_model(repository / "data/models" / MODEL)
    features = (
        pd.read_parquet(
            repository / "data/features" / DEMO_DATASET / DEMO_FEATURES / "features.parquet"
        )
        .sort_values("candidate_id", kind="stable")
        .head(2)
    )
    predictions = predict_candidates(features, model)
    result = {
        "status": "passed",
        "network_required": False,
        "fixture": "tiny_deterministic_box_transit",
        "preprocessing_valid_cadences": int(processed.data["valid"].sum()),
        "bls_candidate_count": int(len(candidates)),
        "bls_best_period_days": bls.best_period,
        "expected_period_days": 2.5,
        "period_recovered_within_5_percent": abs(bls.best_period / 2.5 - 1.0) <= 0.05,
        "frozen_inference_rows": int(len(predictions)),
        "official_model": MODEL,
        "official_threshold": model.threshold,
        "modules_reused": [
            "transiteye.preprocessing.preprocess",
            "transiteye.detection.run_bls",
            "transiteye.detection.extract_peaks",
            "transiteye.modeling.predict_candidates",
        ],
    }
    output = repository / "results/demo_smoke.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(canonical_json(result) + "\n", encoding="utf-8")
    return result


def full_replay_mode(repository: Path) -> None:
    """Invoke accepted local stage scripts; deliberately excludes network acquisition."""
    scripts = (
        "build_demo_dataset.py",
        "build_features.py",
        "run_demo_modeling.py",
        "run_robustness_evaluation.py",
        "run_final_evaluation.py",
    )
    for script in scripts:
        subprocess.run(
            [sys.executable, str(repository / "scripts" / script)],
            cwd=repository,
            check=True,
        )
