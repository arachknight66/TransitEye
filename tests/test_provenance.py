from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from transiteye.config import config_hash, load_config
from transiteye.provenance import (
    GitMetadata,
    RunContext,
    collect_git_metadata,
    dependency_versions,
    derive_seed,
)


def test_seed_derivation_is_deterministic() -> None:
    assert derive_seed(42, "dataset_split") == derive_seed(42, "dataset_split")


def test_seed_derivation_separates_components() -> None:
    assert derive_seed(42, "dataset_split") != derive_seed(42, "random_forest")


def test_seed_derivation_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        derive_seed(-1, "dataset_split")
    with pytest.raises(ValueError):
        derive_seed(42, " ")


def test_git_collection_fails_gracefully_outside_repository(tmp_path: Path) -> None:
    assert collect_git_metadata(tmp_path) == GitMetadata(commit_sha=None, is_dirty=None)


def test_run_context_round_trip_and_config_integration() -> None:
    config = load_config("configs/base.yaml")
    context = RunContext.create(
        config_hash=config_hash(config),
        master_seed=config.reproducibility.master_seed,
        component_names=["elm_initialization", "dataset_split"],
        cwd=Path("."),
        timestamp_utc=datetime(2026, 9, 9, tzinfo=UTC),
    )

    restored = RunContext.from_dict(context.to_dict())

    assert restored == context
    assert context.config_hash == config_hash(config)
    assert context.derived_seeds["dataset_split"] == derive_seed(42, "dataset_split")
    assert context.to_json() == restored.to_json()
    assert context.run_fingerprint.startswith("run-")
    assert "pydantic" in dependency_versions()


def test_run_context_rejects_naive_timestamp() -> None:
    with pytest.raises(ValueError):
        RunContext.create(config_hash="a" * 20, master_seed=42, timestamp_utc=datetime(2026, 9, 9))
