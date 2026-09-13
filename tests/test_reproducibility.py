from __future__ import annotations

import json
import socket
from pathlib import Path

import pandas as pd

from transiteye.acquisition.checksums import sha256_file
from transiteye.reproducibility.manifest import (
    FINAL_EVALUATION,
    MODEL,
    _environment,
    build_artifact_inventory,
    build_reproducibility_manifest,
    current_manifest_directory,
    make_repro_version,
    verify_inventory,
    verify_reproducibility_manifest,
)
from transiteye.reproducibility.reporting import FIGURE_STEMS, TABLE_NAMES
from transiteye.reproducibility.workflow import (
    demo_smoke_mode,
    report_mode,
    verify_mode,
)


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_manifest_is_portable_deterministic_and_preserves_lineage() -> None:
    directory = current_manifest_directory(Path("."))
    assert build_reproducibility_manifest(Path(".")) == directory
    rebuilt_inventory = build_artifact_inventory(Path("."))
    assert len(rebuilt_inventory) == 613
    inventory_payload = _json(directory / "artifact_inventory.json")
    inventory = inventory_payload["artifacts"]
    assert isinstance(inventory, list)
    project = _json(directory / "project_manifest.json")
    assert project["repro_version"] == make_repro_version(inventory)
    assert all(not Path(record["path"]).is_absolute() for record in inventory)
    assert all("/home/" not in record["path"] for record in inventory)
    graph = {node["artifact_id"]: node for node in project["lineage_graph"]}
    assert graph[MODEL]["parent_ids"] == ["features-6278a328b8f624b2759b"]
    assert MODEL in graph[FINAL_EVALUATION]["parent_ids"]
    original = make_repro_version(inventory)
    assert original == make_repro_version(inventory)
    assert "platform" not in project["identity_inputs"]
    assert "hostname" not in project["identity_inputs"]
    environment = _environment(Path("."))
    assert environment["identity_role"] == "non_identity_provenance"
    assert environment["project_package_version"] == "0.1.0"


def test_checksum_verification_detects_missing_and_corruption(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"accepted")
    inventory = [{"path": "artifact.bin", "sha256": sha256_file(artifact)}]
    assert verify_inventory(tmp_path, inventory) == {
        "verified": 1,
        "missing": [],
        "checksum_mismatch": [],
    }
    artifact.write_bytes(b"mutated")
    assert verify_inventory(tmp_path, inventory)["checksum_mismatch"] == ["artifact.bin"]
    artifact.unlink()
    assert verify_inventory(tmp_path, inventory)["missing"] == ["artifact.bin"]


def test_real_manifest_verifies_all_registered_artifacts() -> None:
    directory = current_manifest_directory(Path("."))
    result = verify_reproducibility_manifest(Path("."), directory)
    assert result["status"] == "verified"
    assert result["verified"] == 613
    assert result["missing"] == []
    assert result["checksum_mismatch"] == []
    assert result["metadata_mismatch"] == []


def test_report_outputs_are_complete_deterministic_and_claim_bounded() -> None:
    repository = Path(".")
    model_path = repository / "data/models" / MODEL / "model.joblib"
    model_before = sha256_file(model_path)
    report_mode(repository)
    paths = [repository / "results/tables" / name for name in TABLE_NAMES]
    paths += [
        repository / "results/figures" / f"{stem}.{suffix}"
        for stem in FIGURE_STEMS
        for suffix in ("png", "pdf")
    ]
    first = {str(path): sha256_file(path) for path in paths}
    report_mode(repository)
    assert first == {str(path): sha256_file(path) for path in paths}
    assert model_before == sha256_file(model_path)
    readiness = pd.read_csv(repository / "results/tables/table_7_scientific_readiness.csv")
    assert set(readiness["status"]) == {"PASS", "FAIL"}
    assert not {"accuracy", "precision", "recall", "f1"}.intersection(readiness.columns)
    data_summary = pd.read_csv(repository / "results/tables/table_1_data_pipeline_summary.csv")
    scientific = data_summary.loc[data_summary["metric"] == "scientific_unlabeled_candidates"].iloc[
        0
    ]
    assert scientific["value"] == 75
    assert scientific["context"] == "no gold labels"
    figure_sources = _json(repository / "results/figures/figure_sources.json")
    assert set(figure_sources) == set(FIGURE_STEMS)


def test_offline_modes_and_result_contract(monkeypatch: object) -> None:
    def blocked_socket(*args: object, **kwargs: object) -> None:
        raise AssertionError("Network access attempted")

    monkeypatch.setattr(socket, "socket", blocked_socket)  # type: ignore[attr-defined]
    repository = Path(".")
    assert verify_mode(repository)["status"] == "verified"
    assert report_mode(repository) == repository / "results"
    smoke = demo_smoke_mode(repository)
    assert smoke["status"] == "passed"
    assert smoke["network_required"] is False
    assert smoke["period_recovered_within_5_percent"] is True
    assert smoke["modules_reused"] == [
        "transiteye.preprocessing.preprocess",
        "transiteye.detection.run_bls",
        "transiteye.detection.extract_peaks",
        "transiteye.modeling.predict_candidates",
    ]
    index = _json(repository / "results/index.json")
    assert len(index["tables"]) == 7
    assert len(index["figures"]) == 20
    status = _json(repository / "results/project_status.json")
    assert status["software_pipeline_status"] == "complete"
    assert status["demo_pipeline_status"] == "complete"
    assert status["scientific_readiness"] == "blocked"
    assert status["official_model"] == MODEL
    assert status["official_threshold"] == 0.325
