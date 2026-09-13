"""Portable deterministic identity and immutable B046--B051 artifacts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from transiteye.acquisition.checksums import sha256_file
from transiteye.serialization import canonical_json, content_hash


class EvaluationFreezeError(ValueError):
    """Raised when an immutable evaluation artifact conflicts or is corrupt."""


@dataclass(frozen=True)
class EvaluationIdentity:
    model_version: str
    demo_dataset_version: str
    demo_feature_version: str
    scientific_dataset_version: str
    scientific_feature_version: str
    official_split_id: str
    evaluation_config_hash: str
    injection_grid_identity: str

    def portable_dict(self) -> dict[str, str]:
        return asdict(self)


def make_evaluation_version(identity: EvaluationIdentity) -> str:
    return f"evaluation-{content_hash(identity.portable_dict())}"


def freeze_evaluation(
    *,
    identity: EvaluationIdentity,
    tables: dict[str, pd.DataFrame],
    payloads: dict[str, Any],
    root: str | Path,
) -> Path:
    """Freeze deterministic tables/payloads, or verify exact replay."""
    version = make_evaluation_version(identity)
    directory = Path(root) / version
    metadata_path = directory / "evaluation_metadata.json"
    canonical_tables = {
        filename: frame.reset_index(drop=True) for filename, frame in tables.items()
    }
    if directory.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["evaluation_version"] != version:
            raise EvaluationFreezeError("Frozen evaluation identity mismatch.")
        for filename, expected in metadata["artifact_checksums"].items():
            if sha256_file(directory / filename) != expected:
                raise EvaluationFreezeError(f"Frozen evaluation checksum mismatch: {filename}")
        for filename, frame in canonical_tables.items():
            pd.testing.assert_frame_equal(
                pd.read_parquet(directory / filename), frame, check_dtype=False
            )
        for filename, payload in payloads.items():
            if (directory / filename).read_text(encoding="utf-8") != canonical_json(payload) + "\n":
                raise EvaluationFreezeError(f"Frozen evaluation replay differs: {filename}")
        return directory

    directory.mkdir(parents=True, exist_ok=False)
    for filename, frame in canonical_tables.items():
        frame.to_parquet(directory / filename, engine="pyarrow", index=False)
    for filename, payload in payloads.items():
        (directory / filename).write_text(canonical_json(payload) + "\n", encoding="utf-8")
    artifacts = sorted([*tables, *payloads])
    metadata = {
        "evaluation_version": version,
        "identity_inputs": identity.portable_dict(),
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "statistical_role": "development_demo_robustness",
        "scientific_outputs_are_unlabeled": True,
        "official_model_modified": False,
        "artifact_checksums": {
            filename: sha256_file(directory / filename) for filename in artifacts
        },
    }
    metadata_path.write_text(canonical_json(metadata) + "\n", encoding="utf-8")
    return directory
