"""B058 canonical full-project artifact inventory and verification."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import tomllib
from collections import Counter
from pathlib import Path
from typing import Any, cast

import pandas as pd

from transiteye import __version__
from transiteye.acquisition.checksums import sha256_file
from transiteye.provenance import collect_git_metadata
from transiteye.serialization import canonical_json, content_hash

CATALOG_ID = "toi-20260909T175435Z-b65e9b5d5a5c17f0904e"
COHORT_ID = "pilot-toi-20260909T175435Z-b65e9b5d5a5c17f0904e-8c1d8d1a1e5581bfe536"
MAST_ID = "mast-products-cc80ce4b85ed6ab29d84"
MAST_DIRECTORY = "mast-products-b011-resumable"
MAST_CHECKSUM = "8fc3e8615aaf25539f46e08d1c0a403d27dcbfb3a43dec93819fbfcc0d06a45a"
EXPANSION_ID = "expansion-be59541b9388b685e1f9"
SCIENTIFIC_DATASET = "dataset-4b84e8acaa6f6b334c2f"
DEMO_MANIFEST = "demo-manifest-9459aec7c2e26e406301"
DEMO_DATASET = "dataset-demo-a74b6aad7c21faf15b0d"
SCIENTIFIC_FEATURES = "features-4e9f6d54b84780a2be42"
DEMO_FEATURES = "features-6278a328b8f624b2759b"
MODEL = "model-4a58f313bc50b9c3dc41"
ROBUSTNESS = "evaluation-e1d75a22b6e9f3e57ca8"
FINAL_EVALUATION = "final-evaluation-a4d1c707d285fa72a87f"
MANIFEST_POLICY = "project-reproducibility-manifest-v1"


class ReproducibilityError(ValueError):
    """Raised for incomplete or inconsistent frozen project provenance."""


def _read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _relative(repository: Path, path: Path) -> str:
    return path.relative_to(repository).as_posix()


def _add_record(
    records: list[dict[str, Any]],
    repository: Path,
    path: Path,
    *,
    identifier: str,
    category: str,
    artifact_type: str,
    parents: list[str],
    policy: dict[str, Any],
    expected: str | None = None,
) -> None:
    if not path.is_file():
        raise ReproducibilityError(f"Frozen artifact is missing: {_relative(repository, path)}")
    checksum = sha256_file(path)
    if expected is not None and checksum != expected:
        raise ReproducibilityError(
            f"Frozen artifact checksum mismatch: {_relative(repository, path)}"
        )
    records.append(
        {
            "identifier": identifier,
            "category": category,
            "artifact_type": artifact_type,
            "path": _relative(repository, path),
            "sha256": checksum,
            "parent_ids": sorted(set(parents)),
            "policy_identity": policy,
            "immutable": True,
            "frozen": True,
        }
    )


def _add_declared_directory(
    records: list[dict[str, Any]],
    repository: Path,
    directory: Path,
    metadata_name: str,
    *,
    identifier: str,
    category: str,
    artifact_type: str,
    parents: list[str],
) -> dict[str, Any]:
    metadata_path = directory / metadata_name
    metadata = _read_json(metadata_path)
    policy = cast(dict[str, Any], metadata.get("identity_inputs", {}))
    _add_record(
        records,
        repository,
        metadata_path,
        identifier=identifier,
        category=category,
        artifact_type=f"{artifact_type}_metadata",
        parents=parents,
        policy=policy,
    )
    declared = metadata.get("table_checksums", metadata.get("artifact_checksums", {}))
    for filename, expected in sorted(declared.items()):
        _add_record(
            records,
            repository,
            directory / filename,
            identifier=identifier,
            category=category,
            artifact_type=artifact_type,
            parents=parents,
            policy=policy,
            expected=str(expected),
        )
    return metadata


def build_artifact_inventory(repository: Path) -> list[dict[str, Any]]:
    """Build the inventory from explicit metadata and lineage receipts."""
    records: list[dict[str, Any]] = []
    catalog_dir = repository / "data/catalogs" / CATALOG_ID
    catalog = _read_json(catalog_dir / "metadata.json")
    _add_record(
        records,
        repository,
        catalog_dir / "metadata.json",
        identifier=CATALOG_ID,
        category="CATALOG",
        artifact_type="catalog_metadata",
        parents=[],
        policy={"normalization_schema_version": catalog["normalization_schema_version"]},
    )
    _add_record(
        records,
        repository,
        catalog_dir / "normalized.parquet",
        identifier=CATALOG_ID,
        category="CATALOG",
        artifact_type="normalized_catalog",
        parents=[],
        policy={"normalization_schema_version": catalog["normalization_schema_version"]},
        expected=str(catalog["normalized_table_sha256"]),
    )

    cohort_dir = repository / "data/manifests" / COHORT_ID
    cohort = _read_json(cohort_dir / "metadata.json")
    _add_record(
        records,
        repository,
        cohort_dir / "metadata.json",
        identifier=COHORT_ID,
        category="CATALOG",
        artifact_type="cohort_metadata",
        parents=[CATALOG_ID],
        policy={"cohort_config_hash": cohort["cohort_config_hash"]},
    )
    for field, filename in (
        ("targets_sha256", "targets.parquet"),
        ("gold_events_sha256", "gold_events.parquet"),
        ("secondary_events_sha256", "secondary_events.parquet"),
        ("exclusions_sha256", "exclusions.parquet"),
        ("selected_target_events_sha256", "selected_target_events.parquet"),
    ):
        _add_record(
            records,
            repository,
            cohort_dir / filename,
            identifier=COHORT_ID,
            category="CATALOG",
            artifact_type="cohort_table",
            parents=[CATALOG_ID],
            policy={
                "cohort_config_hash": cohort["cohort_config_hash"],
                "declared_content_sha256": cohort[field],
            },
        )

    mast_dir = repository / "data/manifests" / MAST_DIRECTORY
    mast = _read_json(mast_dir / "metadata.json")
    _add_record(
        records,
        repository,
        mast_dir / "metadata.json",
        identifier=MAST_ID,
        category="ACQUISITION",
        artifact_type="mast_snapshot_metadata",
        parents=[COHORT_ID],
        policy={"snapshot_checksum": mast["products_sha256"]},
    )
    _add_record(
        records,
        repository,
        mast_dir / "normalized_products.parquet",
        identifier=MAST_ID,
        category="ACQUISITION",
        artifact_type="mast_product_snapshot",
        parents=[COHORT_ID],
        policy={"snapshot_checksum": mast["products_sha256"]},
        expected=MAST_CHECKSUM,
    )
    _add_record(
        records,
        repository,
        mast_dir / "coverage.parquet",
        identifier=MAST_ID,
        category="ACQUISITION",
        artifact_type="mast_coverage",
        parents=[COHORT_ID],
        policy={"snapshot_checksum": mast["products_sha256"]},
    )

    expansion_dir = repository / "data/manifests" / EXPANSION_ID
    expansion = _read_json(expansion_dir / "metadata.json")
    expansion_policy = {"expansion_policy_hash": expansion["expansion_policy_hash"]}
    for path in sorted(expansion_dir.iterdir()):
        if path.is_file():
            _add_record(
                records,
                repository,
                path,
                identifier=EXPANSION_ID,
                category="ACQUISITION",
                artifact_type="expansion_manifest",
                parents=[MAST_ID],
                policy=expansion_policy,
            )

    downloads = pd.read_parquet(expansion_dir / "download_receipts.parquet")
    preprocessing = pd.read_parquet(expansion_dir / "preprocessing_receipts.parquet")
    detection = pd.read_parquet(expansion_dir / "detection_receipts.parquet")
    for row in downloads.loc[downloads["status"] == "downloaded"].itertuples():
        raw_id = f"raw-{row.sha256}"
        _add_record(
            records,
            repository,
            repository / "data/raw" / str(row.relative_raw_path),
            identifier=raw_id,
            category="RAW",
            artifact_type="tess_fits",
            parents=[EXPANSION_ID],
            policy={"source": "MAST", "data_uri": str(row.data_uri)},
            expected=str(row.sha256),
        )
    for row in preprocessing.loc[preprocessing["status"] == "processed"].itertuples():
        raw_checksum = str(row.source_raw_checksum)
        processed_id = f"processed-{row.processed_checksum}"
        processed_path = repository / "data/processed/mvp" / raw_checksum[:20] / "cadences.parquet"
        _add_record(
            records,
            repository,
            processed_path,
            identifier=processed_id,
            category="PREPROCESSING",
            artifact_type="processed_light_curve",
            parents=[f"raw-{raw_checksum}"],
            policy={"preprocessing_config_hash": str(row.preprocessing_config_hash)},
            expected=str(row.processed_checksum),
        )
        metadata_path = processed_path.with_name("metadata.json")
        _add_record(
            records,
            repository,
            metadata_path,
            identifier=processed_id,
            category="PREPROCESSING",
            artifact_type="processed_metadata",
            parents=[f"raw-{raw_checksum}"],
            policy={"preprocessing_config_hash": str(row.preprocessing_config_hash)},
        )
    for row in detection.loc[detection["status"] == "success"].itertuples():
        raw_checksum = str(row.source_raw_checksum)
        processed_checksum = str(
            preprocessing.loc[
                preprocessing["source_raw_checksum"] == raw_checksum, "processed_checksum"
            ].iloc[0]
        )
        for detected_filename, checksum, kind in (
            (row.periodogram_filename, row.periodogram_sha256, "bls_periodogram"),
            (row.candidate_filename, row.candidate_sha256, "bls_candidates"),
            (row.match_filename, row.match_sha256, "catalog_matches"),
        ):
            _add_record(
                records,
                repository,
                repository / "data/candidates" / str(row.bls_config_hash) / str(detected_filename),
                identifier=f"detection-{row.observation_group_id}",
                category="DETECTION",
                artifact_type=kind,
                parents=[f"processed-{processed_checksum}"],
                policy={"bls_config_hash": str(row.bls_config_hash)},
                expected=str(checksum),
            )

    demo_manifest_dir = repository / "data/manifests" / DEMO_MANIFEST
    demo_meta = _read_json(demo_manifest_dir / "metadata.json")
    demo_policy = {"demo_policy_hash": demo_meta["demo_policy_hash"]}
    for path in sorted(demo_manifest_dir.iterdir()):
        if path.is_file():
            _add_record(
                records,
                repository,
                path,
                identifier=DEMO_MANIFEST,
                category="DEMO",
                artifact_type="demo_manifest",
                parents=[EXPANSION_ID],
                policy=demo_policy,
            )
    demo_lineage_path = repository / "data/synthetic" / DEMO_MANIFEST / "artifact_lineage.parquet"
    _add_record(
        records,
        repository,
        demo_lineage_path,
        identifier=DEMO_MANIFEST,
        category="DEMO",
        artifact_type="demo_artifact_lineage",
        parents=[EXPANSION_ID],
        policy=demo_policy,
    )
    demo_lineage = pd.read_parquet(demo_lineage_path)
    for row in demo_lineage.itertuples():
        variant_id = str(row.variant_id)
        variant_dir = repository / "data/synthetic" / DEMO_MANIFEST / "variants" / variant_id
        _add_record(
            records,
            repository,
            variant_dir / "source.parquet",
            identifier=variant_id,
            category="DEMO",
            artifact_type="demo_source_curve",
            parents=[DEMO_MANIFEST, f"raw-{row.source_raw_checksum}"],
            policy=demo_policy,
            expected=str(row.source_sha256),
        )
        _add_record(
            records,
            repository,
            variant_dir / "processed.parquet",
            identifier=variant_id,
            category="DEMO",
            artifact_type="demo_processed_curve",
            parents=[DEMO_MANIFEST],
            policy={**demo_policy, "preprocessing_config_hash": str(row.preprocessing_config_hash)},
            expected=str(row.demo_processed_checksum),
        )
        _add_record(
            records,
            repository,
            variant_dir / "metadata.json",
            identifier=variant_id,
            category="DEMO",
            artifact_type="demo_variant_metadata",
            parents=[DEMO_MANIFEST],
            policy=demo_policy,
        )
        detection_dir = repository / "data/synthetic" / DEMO_MANIFEST / "detection"
        for demo_filename, checksum, kind in (
            (row.periodogram_filename, row.periodogram_sha256, "demo_bls_periodogram"),
            (row.candidate_filename, row.candidate_sha256, "demo_bls_candidates"),
            (row.match_filename, row.match_sha256, "demo_event_matches"),
        ):
            _add_record(
                records,
                repository,
                detection_dir / str(demo_filename),
                identifier=f"demo-detection-{variant_id}",
                category="DETECTION",
                artifact_type=kind,
                parents=[variant_id],
                policy={"bls_config_hash": str(row.bls_config_hash)},
                expected=str(checksum),
            )

    _add_declared_directory(
        records,
        repository,
        repository / "data/datasets" / SCIENTIFIC_DATASET,
        "dataset_metadata.json",
        identifier=SCIENTIFIC_DATASET,
        category="DATASET",
        artifact_type="scientific_dataset",
        parents=[EXPANSION_ID],
    )
    _add_declared_directory(
        records,
        repository,
        repository / "data/datasets" / DEMO_DATASET,
        "dataset_metadata.json",
        identifier=DEMO_DATASET,
        category="DATASET",
        artifact_type="demo_dataset",
        parents=[DEMO_MANIFEST],
    )
    for dataset, feature_id in (
        (SCIENTIFIC_DATASET, SCIENTIFIC_FEATURES),
        (DEMO_DATASET, DEMO_FEATURES),
    ):
        feature_dir = repository / "data/features" / dataset / feature_id
        feature_meta = _add_declared_directory(
            records,
            repository,
            feature_dir,
            "feature_metadata.json",
            identifier=feature_id,
            category="FEATURES",
            artifact_type="feature_matrix",
            parents=[dataset],
        )
        for filename, checksum_field in (
            ("feature_registry.json", "registry_sha256"),
            ("feature_validation.json", "validation_sha256"),
        ):
            _add_record(
                records,
                repository,
                feature_dir / filename,
                identifier=feature_id,
                category="FEATURES",
                artifact_type=filename.removesuffix(".json"),
                parents=[dataset],
                policy=cast(dict[str, Any], feature_meta["identity_inputs"]),
                expected=str(feature_meta[checksum_field]),
            )
    _add_declared_directory(
        records,
        repository,
        repository / "data/models" / MODEL,
        "model_metadata.json",
        identifier=MODEL,
        category="MODELING",
        artifact_type="frozen_model",
        parents=[DEMO_FEATURES],
    )
    _add_declared_directory(
        records,
        repository,
        repository / "data/evaluation" / ROBUSTNESS,
        "evaluation_metadata.json",
        identifier=ROBUSTNESS,
        category="EVALUATION",
        artifact_type="robustness_evaluation",
        parents=[MODEL, DEMO_FEATURES, SCIENTIFIC_FEATURES],
    )
    _add_declared_directory(
        records,
        repository,
        repository / "data/evaluation" / FINAL_EVALUATION,
        "evaluation_metadata.json",
        identifier=FINAL_EVALUATION,
        category="EVALUATION",
        artifact_type="final_evaluation",
        parents=[ROBUSTNESS, MODEL],
    )
    return sorted(
        records, key=lambda record: (record["category"], record["path"], record["artifact_type"])
    )


def _environment(repository: Path) -> dict[str, Any]:
    direct_names: list[str] = []
    with (repository / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)["project"]
    for requirement in project["dependencies"]:
        direct_names.append(str(requirement).split(">", 1)[0].split("<", 1)[0].split("=", 1)[0])
    versions = {}
    for name in sorted(direct_names):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "missing"
    git = collect_git_metadata(repository)
    return {
        "identity_role": "non_identity_provenance",
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "project_package_version": __version__,
        "uv_lock_sha256": sha256_file(repository / "uv.lock"),
        "pyproject_sha256": sha256_file(repository / "pyproject.toml"),
        "direct_dependency_versions": versions,
        "runtime_library_versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "matplotlib", "scikit-learn", "astropy", "pyarrow")
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "git": {"commit_sha": git.commit_sha, "is_dirty": git.is_dirty},
    }


def _project_graph() -> list[dict[str, Any]]:
    return [
        {"artifact_id": CATALOG_ID, "artifact_type": "catalog_snapshot", "parent_ids": []},
        {"artifact_id": COHORT_ID, "artifact_type": "cohort", "parent_ids": [CATALOG_ID]},
        {
            "artifact_id": MAST_ID,
            "artifact_type": "mast_product_snapshot",
            "parent_ids": [COHORT_ID],
        },
        {
            "artifact_id": EXPANSION_ID,
            "artifact_type": "expansion_manifest",
            "parent_ids": [MAST_ID],
        },
        {
            "artifact_id": "scientific-raw-products",
            "artifact_type": "raw_products",
            "parent_ids": [EXPANSION_ID],
        },
        {
            "artifact_id": "scientific-processed-products",
            "artifact_type": "processed_products",
            "parent_ids": ["scientific-raw-products"],
        },
        {
            "artifact_id": "scientific-bls-outputs",
            "artifact_type": "blind_bls_outputs",
            "parent_ids": ["scientific-processed-products"],
        },
        {
            "artifact_id": SCIENTIFIC_DATASET,
            "artifact_type": "scientific_dataset",
            "parent_ids": ["scientific-bls-outputs", CATALOG_ID],
        },
        {
            "artifact_id": DEMO_MANIFEST,
            "artifact_type": "demo_manifest",
            "parent_ids": ["scientific-processed-products"],
        },
        {
            "artifact_id": DEMO_DATASET,
            "artifact_type": "demo_dataset",
            "parent_ids": [DEMO_MANIFEST],
        },
        {
            "artifact_id": SCIENTIFIC_FEATURES,
            "artifact_type": "feature_matrix",
            "parent_ids": [SCIENTIFIC_DATASET],
        },
        {
            "artifact_id": DEMO_FEATURES,
            "artifact_type": "feature_matrix",
            "parent_ids": [DEMO_DATASET],
        },
        {"artifact_id": MODEL, "artifact_type": "model", "parent_ids": [DEMO_FEATURES]},
        {
            "artifact_id": ROBUSTNESS,
            "artifact_type": "robustness_evaluation",
            "parent_ids": [MODEL, DEMO_FEATURES, SCIENTIFIC_FEATURES],
        },
        {
            "artifact_id": FINAL_EVALUATION,
            "artifact_type": "final_evaluation",
            "parent_ids": [ROBUSTNESS, MODEL],
        },
    ]


def make_repro_version(inventory: list[dict[str, Any]]) -> str:
    """Hash only portable artifact and reproduction-policy identity."""
    identity = {
        "policy": MANIFEST_POLICY,
        "artifact_fingerprints": [
            {key: row[key] for key in ("identifier", "artifact_type", "path", "sha256")}
            for row in inventory
        ],
        "reproduction_policy": "uv-offline-verify-report-smoke-v1",
    }
    return f"repro-{content_hash(identity)}"


def build_reproducibility_manifest(root: str | Path) -> Path:
    repository = Path(root)
    inventory = build_artifact_inventory(repository)
    identity = {
        "policy": MANIFEST_POLICY,
        "artifact_fingerprints": [
            {key: row[key] for key in ("identifier", "artifact_type", "path", "sha256")}
            for row in inventory
        ],
        "reproduction_policy": "uv-offline-verify-report-smoke-v1",
    }
    repro_version = make_repro_version(inventory)
    directory = repository / "data/reproducibility" / repro_version
    if directory.exists():
        summary = verify_reproducibility_manifest(repository, directory)
        if summary["status"] != "verified":
            raise ReproducibilityError("Existing reproducibility manifest failed verification.")
        return directory
    directory.mkdir(parents=True, exist_ok=False)
    project_manifest = {
        "repro_version": repro_version,
        "identity_inputs": identity,
        "lineage_graph": _project_graph(),
        "deterministic_metadata": [
            "artifact identities",
            "relative paths",
            "checksums",
            "policy identities",
            "parent identities",
        ],
        "non_identity_metadata": [
            "environment platform",
            "git status",
            "execution time",
            "hostname",
            "username",
            "absolute project path",
        ],
        "official_threshold": 0.325,
        "demo_pipeline_status": "complete",
        "scientific_readiness": "blocked",
    }
    checksums = {row["path"]: row["sha256"] for row in inventory}
    categories = Counter(str(row["category"]) for row in inventory)
    payloads = {
        "project_manifest.json": project_manifest,
        "artifact_inventory.json": {
            "repro_version": repro_version,
            "artifact_count": len(inventory),
            "artifacts": inventory,
        },
        "checksums.json": {
            "repro_version": repro_version,
            "algorithm": "SHA-256",
            "artifacts": dict(sorted(checksums.items())),
        },
        "environment.json": _environment(repository),
        "pipeline_summary.json": {
            "repro_version": repro_version,
            "artifact_categories": dict(sorted(categories.items())),
            "software_pipeline_status": "complete",
            "demo_pipeline_status": "complete",
            "scientific_readiness": "blocked",
        },
        "reproduction_plan.json": {
            "command": "uv run python scripts/reproduce_project.py",
            "offline_modes": ["--verify", "--report", "--demo-smoke"],
            "full_replay": {
                "available": True,
                "flag": "--full-replay",
                "warning": "expensive local replay; acquisition is not rerun",
            },
        },
    }
    for filename, payload in payloads.items():
        (directory / filename).write_text(canonical_json(payload) + "\n", encoding="utf-8")
    return directory


def verify_reproducibility_manifest(
    root: str | Path, manifest_directory: str | Path
) -> dict[str, Any]:
    repository = Path(root)
    directory = Path(manifest_directory)
    inventory = _read_json(directory / "artifact_inventory.json")["artifacts"]
    summary = verify_inventory(repository, inventory)
    missing = summary["missing"]
    mismatched = summary["checksum_mismatch"]
    assert isinstance(missing, list) and isinstance(mismatched, list)
    metadata_mismatch: list[str] = []
    model = _read_json(repository / "data/models" / MODEL / "model_metadata.json")
    scientific_dataset = _read_json(
        repository / "data/datasets" / SCIENTIFIC_DATASET / "dataset_metadata.json"
    )
    feature_metadata = [
        _read_json(repository / "data/features" / dataset / feature / "feature_metadata.json")
        for dataset, feature in (
            (DEMO_DATASET, DEMO_FEATURES),
            (SCIENTIFIC_DATASET, SCIENTIFIC_FEATURES),
        )
    ]
    robustness = _read_json(
        repository / "data/evaluation" / ROBUSTNESS / "evaluation_metadata.json"
    )
    final = _read_json(
        repository / "data/evaluation" / FINAL_EVALUATION / "evaluation_metadata.json"
    )
    if (
        model.get("model_version") != MODEL
        or model.get("identity_inputs", {}).get("selected_threshold") != 0.325
    ):
        metadata_mismatch.append(MODEL)
    scientific_identity = scientific_dataset.get("identity_inputs", {})
    if (
        scientific_dataset.get("dataset_version") != SCIENTIFIC_DATASET
        or scientific_identity.get("acquisition_snapshot_id") != MAST_ID
        or scientific_identity.get("acquisition_snapshot_hash") != MAST_CHECKSUM
        or scientific_identity.get("catalog_snapshot_id") != CATALOG_ID
    ):
        metadata_mismatch.append(SCIENTIFIC_DATASET)
    for metadata, expected_dataset, feature_id in zip(
        feature_metadata,
        (DEMO_DATASET, SCIENTIFIC_DATASET),
        (DEMO_FEATURES, SCIENTIFIC_FEATURES),
        strict=True,
    ):
        if (
            metadata.get("feature_version") != feature_id
            or metadata.get("identity_inputs", {}).get("dataset_version") != expected_dataset
        ):
            metadata_mismatch.append(feature_id)
    robustness_identity = robustness.get("identity_inputs", {})
    if (
        robustness.get("evaluation_version") != ROBUSTNESS
        or robustness_identity.get("model_version") != MODEL
        or robustness_identity.get("demo_feature_version") != DEMO_FEATURES
        or robustness_identity.get("scientific_feature_version") != SCIENTIFIC_FEATURES
    ):
        metadata_mismatch.append(ROBUSTNESS)
    if (
        final.get("identity_inputs", {}).get("model_version") != MODEL
        or final.get("identity_inputs", {}).get("robustness_evaluation_version") != ROBUSTNESS
    ):
        metadata_mismatch.append(FINAL_EVALUATION)
    status = "verified" if not (missing or mismatched or metadata_mismatch) else "failed"
    return {
        "status": status,
        "verified": summary["verified"],
        "missing": missing,
        "checksum_mismatch": mismatched,
        "metadata_mismatch": metadata_mismatch,
    }


def verify_inventory(repository: Path, inventory: list[dict[str, Any]]) -> dict[str, Any]:
    """Verify file existence and SHA-256 for any portable inventory."""
    missing: list[str] = []
    mismatched: list[str] = []
    for record in inventory:
        path = repository / str(record["path"])
        if not path.is_file():
            missing.append(str(record["path"]))
        elif sha256_file(path) != record["sha256"]:
            mismatched.append(str(record["path"]))
    return {
        "verified": len(inventory) - len(missing) - len(mismatched),
        "missing": missing,
        "checksum_mismatch": mismatched,
    }


def current_manifest_directory(repository: Path) -> Path:
    directories = sorted((repository / "data/reproducibility").glob("repro-*"))
    if len(directories) != 1:
        raise ReproducibilityError("Expected exactly one frozen reproducibility manifest.")
    return directories[0]
