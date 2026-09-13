from __future__ import annotations

import inspect
import json
from pathlib import Path

from transiteye.features import build_feature_matrix
from transiteye.modeling.inference import predict_candidates
from transiteye.reproducibility.release import (
    RELEASE_PATHS,
    build_release_manifest,
    validate_release_manifest,
)

ROOT = Path(".")
ACCEPTANCE_KEYS = {
    "architecture_coherent",
    "claims_valid",
    "dependencies_valid",
    "documentation_valid",
    "frozen_artifacts_valid",
    "offline_modes_valid",
    "quality_checks_valid",
    "repository_audited",
    "secrets_absent",
}


def test_release_manifest_is_deterministic_portable_and_complete() -> None:
    saved = json.loads((ROOT / "results/release_manifest.json").read_text(encoding="utf-8"))
    assert set(saved["acceptance"]) == ACCEPTANCE_KEYS
    assert all(saved["acceptance"].values())
    assert saved == build_release_manifest(ROOT, saved["acceptance"])
    assert validate_release_manifest(ROOT, saved) == []
    assert saved["package_version"] == "0.1.0"
    assert saved["release_status"] == "complete"
    assert saved["scientific_readiness"] == "blocked"
    assert saved["release_version_id"].startswith("release-")
    assert all(not Path(path).is_absolute() for path in RELEASE_PATHS.values())


def test_release_manifest_preserves_authoritative_identities() -> None:
    saved = json.loads((ROOT / "results/release_manifest.json").read_text(encoding="utf-8"))
    inputs = saved["identity_inputs"]
    status = json.loads((ROOT / "results/project_status.json").read_text(encoding="utf-8"))
    index = json.loads((ROOT / "results/index.json").read_text(encoding="utf-8"))
    assert inputs["official_model"] == status["official_model"]
    assert inputs["official_threshold"] == status["official_threshold"]
    assert inputs["final_evaluation"] == status["final_evaluation"]
    assert inputs["repro_version"] == status["repro_version"]
    assert inputs["scientific_dataset"] == index["scientific_dataset"]
    assert inputs["demo_dataset"] == index["demo_dataset"]
    assert inputs["scientific_features"] == index["scientific_features"]
    assert inputs["demo_features"] == index["demo_features"]


def test_clean_distribution_contract_and_shared_downstream_interfaces() -> None:
    required = (
        "README.md",
        "uv.lock",
        "scripts/reproduce_project.py",
        "results/index.json",
        "results/project_status.json",
        "data/reproducibility/repro-f43b9410e759001971ae/project_manifest.json",
        "data/models/model-4a58f313bc50b9c3dc41/model.joblib",
    )
    assert all((ROOT / path).exists() for path in required)
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "Run commands from the repository root" in readme
    assert "source-only Git checkout" in readme
    assert list(inspect.signature(build_feature_matrix).parameters) == [
        "root",
        "dataset_version",
        "config_path",
    ]
    assert list(inspect.signature(predict_candidates).parameters) == [
        "feature_matrix",
        "model",
    ]
