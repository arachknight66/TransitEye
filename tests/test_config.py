from __future__ import annotations

from pathlib import Path

import pytest

from transiteye.config import (
    ConfigurationError,
    config_hash,
    load_config,
    resolve_config,
    stable_config_json,
)


def test_load_valid_config() -> None:
    config = load_config(Path("configs/base.yaml"))

    assert config.project.name == "TransitEye"
    assert resolve_config(config)["reproducibility"]["master_seed"] == 42


def test_load_rejects_malformed_yaml(tmp_path: Path) -> None:
    path = tmp_path / "broken.yaml"
    path.write_text("project: [unterminated\n", encoding="utf-8")

    with pytest.raises(ConfigurationError):
        load_config(path)


def test_load_rejects_non_mapping_root(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- not\n- a mapping\n", encoding="utf-8")

    with pytest.raises(ConfigurationError):
        load_config(path)


def test_load_rejects_missing_required_fields(tmp_path: Path) -> None:
    path = tmp_path / "missing.yaml"
    path.write_text("config_version: 1\nproject:\n  name: TransitEye\n", encoding="utf-8")

    with pytest.raises(ConfigurationError):
        load_config(path)


def test_config_hash_is_deterministic_and_order_independent() -> None:
    first = {
        "config_version": 1,
        "project": {"name": "TransitEye"},
        "reproducibility": {"master_seed": 42},
    }
    second = {
        "reproducibility": {"master_seed": 42},
        "project": {"name": "TransitEye"},
        "config_version": 1,
    }

    assert config_hash(first) == config_hash(first)
    assert config_hash(first) == config_hash(second)


def test_config_hash_changes_for_changed_configuration() -> None:
    baseline = {
        "config_version": 1,
        "project": {"name": "TransitEye"},
        "reproducibility": {"master_seed": 42},
    }
    changed = {
        **baseline,
        "reproducibility": {"master_seed": 43},
    }

    assert config_hash(baseline) != config_hash(changed)
    assert stable_config_json(baseline).startswith('{"config_version":1')


def test_load_rejects_unknown_configuration_fields(tmp_path: Path) -> None:
    path = tmp_path / "unknown.yaml"
    path.write_text(
        "config_version: 1\nproject:\n  name: TransitEye\nunknown: true\n"
        "reproducibility:\n  master_seed: 42\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError):
        load_config(path)
