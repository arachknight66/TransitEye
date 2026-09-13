"""Portable identity and immutable storage for the B052--B057 package."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from transiteye.acquisition.checksums import sha256_file
from transiteye.serialization import canonical_json, content_hash


class FinalEvaluationFreezeError(ValueError):
    """Raised when an upstream or final evaluation checksum is invalid."""


@dataclass(frozen=True)
class FinalEvaluationIdentity:
    model_version: str
    robustness_evaluation_version: str
    demo_feature_version: str
    scientific_feature_version: str
    demo_dataset_version: str
    scientific_dataset_version: str
    calibration_protocol: str
    interpretability_protocol: str
    scorecard_policy: str
    readiness_policy: str

    def portable_dict(self) -> dict[str, str]:
        return asdict(self)


def make_final_evaluation_version(identity: FinalEvaluationIdentity) -> str:
    return f"final-evaluation-{content_hash(identity.portable_dict())}"


def verify_declared_checksums(directory: Path, metadata_name: str) -> dict[str, str]:
    """Verify either table_checksums or artifact_checksums declared by metadata."""
    metadata_path = directory / metadata_name
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    declared = metadata.get("table_checksums", metadata.get("artifact_checksums", {}))
    checksums = {str(metadata_path): sha256_file(metadata_path)}
    for filename, expected in declared.items():
        path = directory / filename
        actual = sha256_file(path)
        if actual != expected:
            raise FinalEvaluationFreezeError(f"Upstream checksum mismatch: {path}")
        checksums[str(path)] = actual
    return checksums


def freeze_final_evaluation(
    *,
    identity: FinalEvaluationIdentity,
    tables: dict[str, pd.DataFrame],
    payloads: dict[str, Any],
    upstream_checksums: dict[str, str],
    root: Path,
) -> Path:
    """Freeze the final diagnostic package, or compare an exact deterministic replay."""
    version = make_final_evaluation_version(identity)
    directory = root / version
    canonical_tables = {name: frame.reset_index(drop=True) for name, frame in tables.items()}
    checksums_payload = {
        "upstream_artifacts": dict(sorted(upstream_checksums.items())),
        "validation": "sha256_verified_before_analysis",
    }
    all_payloads = {**payloads, "checksums.json": checksums_payload}
    metadata_path = directory / "evaluation_metadata.json"
    if directory.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["final_evaluation_version"] != version:
            raise FinalEvaluationFreezeError("Final evaluation identity mismatch.")
        for name, expected in metadata["artifact_checksums"].items():
            if sha256_file(directory / name) != expected:
                raise FinalEvaluationFreezeError(f"Final artifact checksum mismatch: {name}")
        for name, frame in canonical_tables.items():
            pd.testing.assert_frame_equal(
                pd.read_parquet(directory / name), frame, check_dtype=False
            )
        for name, payload in all_payloads.items():
            if (directory / name).read_text(encoding="utf-8") != canonical_json(payload) + "\n":
                raise FinalEvaluationFreezeError(f"Final evaluation replay differs: {name}")
        return directory

    directory.mkdir(parents=True, exist_ok=False)
    for name, frame in canonical_tables.items():
        frame.to_parquet(directory / name, engine="pyarrow", index=False)
    for name, payload in all_payloads.items():
        (directory / name).write_text(canonical_json(payload) + "\n", encoding="utf-8")
    artifacts = sorted([*canonical_tables, *all_payloads])
    metadata = {
        "final_evaluation_version": version,
        "identity_inputs": identity.portable_dict(),
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "official_model_modified": False,
        "fixed_threshold": 0.325,
        "scientific_outputs_are_unlabeled": True,
        "demo_pipeline_status": "complete",
        "scientific_readiness": "blocked",
        "scientific_claim_boundary": {
            "supported": [
                "controlled synthetic-injection validation",
                "development-demo classifier performance",
                "grouped TIC robustness",
                "injected-event recovery",
                "demo-to-scientific covariate shift diagnostics",
                "model-generated candidate scores",
            ],
            "not_supported": [
                "real TESS exoplanet detection accuracy",
                "confirmed planet discovery",
                "validated real candidate classification",
                "population-level TESS generalization",
                "calibrated scientific planet probability",
            ],
        },
        "artifact_checksums": {name: sha256_file(directory / name) for name in artifacts},
    }
    metadata_path.write_text(canonical_json(metadata) + "\n", encoding="utf-8")
    return directory
