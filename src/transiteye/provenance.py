"""Minimal provenance and deterministic seed primitives."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from transiteye.identifiers import make_run_fingerprint
from transiteye.serialization import canonical_json


@dataclass(frozen=True)
class GitMetadata:
    """Best-effort Git revision information."""

    commit_sha: str | None
    is_dirty: bool | None


def collect_git_metadata(cwd: str | Path | None = None) -> GitMetadata:
    """Collect Git metadata, returning unavailable values outside a repository."""
    working_directory = str(cwd) if cwd is not None else None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            cwd=working_directory,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            cwd=working_directory,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return GitMetadata(commit_sha=None, is_dirty=None)
    return GitMetadata(commit_sha=commit or None, is_dirty=bool(status.strip()))


def derive_seed(master_seed: int, component_name: str) -> int:
    """Derive a stable 32-bit seed from a master seed and component name."""
    if isinstance(master_seed, bool) or master_seed < 0:
        raise ValueError("Master seed must be a non-negative integer.")
    if not component_name.strip():
        raise ValueError("Component name must be non-empty.")
    payload = f"{master_seed}:{component_name.strip()}".encode()
    derived = int.from_bytes(hashlib.sha256(payload).digest()[:8], byteorder="big")
    seed = derived % ((2**32) - 1)
    return seed if seed != 0 else 1


def derive_seeds(master_seed: int, component_names: Iterable[str]) -> dict[str, int]:
    """Derive named seeds in deterministic key order."""
    names = sorted(set(component_names))
    return {name: derive_seed(master_seed, name) for name in names}


def dependency_versions() -> dict[str, str]:
    """Return installed package versions for provenance without local paths."""
    versions: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        try:
            name = distribution.metadata["Name"]
        except KeyError:
            continue
        versions[name.lower()] = distribution.version
    return dict(sorted(versions.items()))


def platform_information() -> dict[str, str]:
    """Return portable platform metadata."""
    return {
        "implementation": platform.python_implementation(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "system": platform.system(),
    }


def _utc_timestamp() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


@dataclass(frozen=True)
class RunContext:
    """Serializable provenance for one future experiment execution."""

    timestamp_utc: str
    git_commit_sha: str | None
    git_dirty: bool | None
    python_version: str
    platform: dict[str, str]
    dependencies: dict[str, str]
    master_seed: int
    derived_seeds: dict[str, int]
    config_hash: str
    run_fingerprint: str

    @classmethod
    def create(
        cls,
        *,
        config_hash: str,
        master_seed: int,
        component_names: Iterable[str] = (),
        cwd: str | Path | None = None,
        timestamp_utc: datetime | None = None,
    ) -> RunContext:
        """Construct a context from the current environment and Git state."""
        timestamp = timestamp_utc or _utc_timestamp()
        if timestamp.tzinfo is None:
            raise ValueError("Provenance timestamps must be timezone-aware.")
        git = collect_git_metadata(cwd)
        return cls(
            timestamp_utc=timestamp.astimezone(UTC).isoformat().replace("+00:00", "Z"),
            git_commit_sha=git.commit_sha,
            git_dirty=git.is_dirty,
            python_version=sys.version.split()[0],
            platform=platform_information(),
            dependencies=dependency_versions(),
            master_seed=master_seed,
            derived_seeds=derive_seeds(master_seed, component_names),
            config_hash=config_hash,
            run_fingerprint=make_run_fingerprint(
                config_hash=config_hash,
                source_revision=git.commit_sha,
                master_seed=master_seed,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible provenance fields in deterministic key order."""
        return cast(dict[str, Any], json.loads(canonical_json(asdict(self))))

    def to_json(self) -> str:
        """Return canonical provenance JSON."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RunContext:
        """Reconstruct a serialized run context."""
        return cls(**value)
