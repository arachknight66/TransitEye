"""Application configuration loaded from TOML without optional dependencies."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class WorkspaceConfig:
    """Locations for non-versioned data and generated artifacts."""

    root: Path
    cache_directory: Path
    results_directory: Path


@dataclass(frozen=True, slots=True)
class AcquisitionConfig:
    """Network and retry limits for archive product acquisition."""

    max_attempts: int = 3
    retry_delay_seconds: float = 1.0
    download_workers: int = 2

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least one.")
        if self.retry_delay_seconds < 0:
            raise ValueError("retry_delay_seconds cannot be negative.")
        if self.download_workers < 1:
            raise ValueError("download_workers must be at least one.")


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Top-level configuration for all Phase 1 services."""

    version: str
    workspace: WorkspaceConfig
    acquisition: AcquisitionConfig


def default_config() -> AppConfig:
    """Return local defaults that keep large artifacts outside the repository."""

    root = Path.cwd() / ".transiteye-workspace"
    return AppConfig(
        version="1",
        workspace=WorkspaceConfig(
            root=root,
            cache_directory=root / "cache",
            results_directory=root / "results",
        ),
        acquisition=AcquisitionConfig(),
    )


def load_config(path: Path | None = None) -> AppConfig:
    """Load a TOML configuration file, using explicit defaults for absent fields."""

    config = default_config()
    if path is None:
        return config
    with path.open("rb") as file_handle:
        document: dict[str, Any] = tomllib.load(file_handle)
    workspace_document = document.get("workspace", {})
    root = Path(workspace_document.get("root", config.workspace.root)).expanduser()
    cache_directory = Path(workspace_document.get("cache_directory", root / "cache")).expanduser()
    results_directory = Path(
        workspace_document.get("results_directory", root / "results")
    ).expanduser()
    acquisition_document = document.get("acquisition", {})
    return AppConfig(
        version=str(document.get("version", config.version)),
        workspace=WorkspaceConfig(root, cache_directory, results_directory),
        acquisition=AcquisitionConfig(
            max_attempts=int(
                acquisition_document.get("max_attempts", config.acquisition.max_attempts)
            ),
            retry_delay_seconds=float(
                acquisition_document.get(
                    "retry_delay_seconds", config.acquisition.retry_delay_seconds
                )
            ),
            download_workers=int(
                acquisition_document.get("download_workers", config.acquisition.download_workers)
            ),
        ),
    )


def ensure_workspace(config: AppConfig) -> None:
    """Create the externally configured workspace directories when a service starts."""

    for directory in (
        config.workspace.root,
        config.workspace.cache_directory,
        config.workspace.results_directory,
    ):
        directory.mkdir(parents=True, exist_ok=True)
