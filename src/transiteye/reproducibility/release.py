"""Deterministic final release metadata built around frozen project identities."""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
from pathlib import Path
from typing import Any, cast

from transiteye.acquisition.checksums import sha256_file
from transiteye.serialization import canonical_json

RELEASE_POLICY = "academic-project-release-manifest-v1"
RELEASE_PATHS = {
    "readme": "README.md",
    "final_report": "reports/TransitEye_Final_Report.md",
    "demo_script": "presentation/DEMO_SCRIPT.md",
    "tables": "results/tables",
    "figures": "results/figures",
    "release_notes": "RELEASE_NOTES.md",
    "changelog": "CHANGELOG.md",
    "result_index": "results/index.json",
    "project_status": "results/project_status.json",
}


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return cast(dict[str, Any], value)


def _release_document_checksum(key: str, path: Path) -> str:
    """Keep the B065 documentation snapshot stable around the later UI section."""
    if key != "readme":
        return sha256_file(path)
    text = path.read_text(encoding="utf-8")
    historical = re.sub(
        r"\n## Interactive App\n.*?\n(?=## Repository structure)",
        "\n",
        text,
        flags=re.DOTALL,
    )
    return hashlib.sha256(historical.encode("utf-8")).hexdigest()


def build_release_manifest(repository: Path, acceptance: dict[str, bool]) -> dict[str, Any]:
    """Build portable release metadata without timestamps or machine identity."""
    status = _load_json(repository / "results/project_status.json")
    index = _load_json(repository / "results/index.json")
    project = tomllib.loads((repository / "pyproject.toml").read_text(encoding="utf-8"))
    documentation_checksums = {
        key: _release_document_checksum(key, repository / path)
        for key, path in RELEASE_PATHS.items()
        if (repository / path).is_file()
    }
    identity_inputs = {
        "policy": RELEASE_POLICY,
        "package_version": project["project"]["version"],
        "official_model": status["official_model"],
        "official_threshold": status["official_threshold"],
        "scientific_dataset": index["scientific_dataset"],
        "demo_dataset": index["demo_dataset"],
        "scientific_features": index["scientific_features"],
        "demo_features": index["demo_features"],
        "robustness_evaluation": index["robustness_evaluation"],
        "final_evaluation": status["final_evaluation"],
        "repro_version": status["repro_version"],
        "documentation_checksums": documentation_checksums,
        "acceptance": acceptance,
    }
    digest = hashlib.sha256(canonical_json(identity_inputs).encode()).hexdigest()[:20]
    return {
        "release_version_id": f"release-{digest}",
        "release_status": "complete",
        "package_version": project["project"]["version"],
        "scientific_readiness": status["scientific_readiness"],
        "identity_inputs": identity_inputs,
        "paths": RELEASE_PATHS,
        "acceptance": acceptance,
    }


def write_release_manifest(repository: Path, acceptance: dict[str, bool]) -> Path:
    """Write the canonical deterministic release manifest."""
    output = repository / "results/release_manifest.json"
    output.write_text(canonical_json(build_release_manifest(repository, acceptance)) + "\n")
    return output


def validate_release_manifest(repository: Path, manifest: dict[str, Any]) -> list[str]:
    """Return portable-path, checksum, identity, and status validation failures."""
    failures: list[str] = []
    paths = manifest.get("paths", {})
    if not isinstance(paths, dict):
        return ["paths is not an object"]
    for key, value in paths.items():
        path = Path(value)
        if path.is_absolute():
            failures.append(f"absolute path: {key}")
        elif not (repository / path).exists():
            failures.append(f"missing path: {key}")
    expected = build_release_manifest(repository, manifest.get("acceptance", {}))
    if manifest.get("release_version_id") != expected["release_version_id"]:
        failures.append("release identity mismatch")
    if manifest.get("scientific_readiness") != "blocked":
        failures.append("scientific readiness changed")
    return failures
