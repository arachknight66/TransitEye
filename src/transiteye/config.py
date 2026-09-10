"""Typed, deterministic project configuration."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from transiteye.serialization import canonical_json, content_hash


class ConfigurationError(ValueError):
    """Raised when a configuration file cannot be parsed or validated."""


class ProjectSettings(BaseModel):
    """Non-scientific project identity settings."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    name: str = Field(min_length=1)


class ReproducibilitySettings(BaseModel):
    """Minimal reproducibility settings shared by future stages."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    master_seed: int = Field(ge=0, le=(2**32) - 1)


class PilotCohortSettings(BaseModel):
    """Development-only target counts and deterministic sampling settings."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    positive_target_count: int = Field(gt=0)
    negative_target_count: int = Field(gt=0)
    seed_component: str = Field(min_length=1)


class CatalogSettings(BaseModel):
    """Settings required to retrieve and select a frozen TOI catalog snapshot."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    source: Literal["nasa_exoplanet_archive"]
    table: Literal["toi"]
    endpoint: str = Field(min_length=1)
    requested_fields: tuple[str, ...] = Field(min_length=1)
    disposition_policy_version: str = Field(min_length=1)
    pilot: PilotCohortSettings


class AcquisitionSettings(BaseModel):
    """Narrow MAST search and product-selection settings for the acquisition stage."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    source: Literal["mast"]
    mission: Literal["TESS"]
    product_kind: Literal["lightcurve"]
    preferred_author: str = Field(min_length=1)
    preferred_exposure_seconds: float | None = Field(default=None, gt=0)
    sector_policy: Literal["all_available"]
    duplicate_resolution: Literal["latest_release"]
    allow_fallback: bool = False
    download_pilot_target_limit: int = Field(gt=0, le=3)
    download_pilot_sectors_per_target: int = Field(gt=0)


class ExpansionSettings(BaseModel):
    """Frozen, label-independent real-cohort expansion selection policy."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    product_snapshot_id: str = Field(min_length=1)
    required_author: Literal["TESS-SPOC"]
    required_product_suffix: Literal["_lc.fits"]
    preferred_exposure_seconds: float = Field(gt=0)
    sectors_per_tic: int = Field(gt=0)
    sector_order: Literal["earliest"]


class PreprocessingSettings(BaseModel):
    """Provisional, common MVP preprocessing parameters."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    flux_stream: Literal["PDCSAP_FLUX"]
    gap_days: float = Field(gt=0)
    trend_window_cadences: int = Field(ge=5)
    positive_spike_mad: float = Field(gt=0)


class BlsSettings(BaseModel):
    """Shared, blind MVP Box Least Squares search policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    min_period_days: float = Field(gt=0)
    max_period_days: float = Field(gt=0)
    min_transits: int = Field(ge=2)
    durations_days: tuple[float, ...] = Field(min_length=1)
    frequency_factor: float = Field(gt=0)
    top_k: int = Field(gt=0)
    local_peak_fraction: float = Field(gt=0, lt=1)
    harmonic_tolerance: float = Field(gt=0, lt=1)
    period_match_tolerance: float = Field(gt=0, lt=1)
    phase_match_tolerance: float = Field(gt=0, lt=0.5)


class DatasetSplitSettings(BaseModel):
    """Future final split proportions and deterministic grouping policy."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)
    train_fraction: float = Field(ge=0, le=1)
    validation_fraction: float = Field(ge=0, le=1)
    test_fraction: float = Field(ge=0, le=1)
    seed_component: str = Field(min_length=1)
    grouped_stratification: Literal["none"] = "none"

    @model_validator(mode="after")
    def validate_fraction_sum(self) -> DatasetSplitSettings:
        if abs(self.train_fraction + self.validation_fraction + self.test_fraction - 1) > 1e-12:
            raise ValueError("Dataset split fractions must sum to one.")
        return self


class DatasetSettings(BaseModel):
    """B031--B034 dataset semantics without feature or model configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)
    policy_version: str = Field(min_length=1)
    label_policy_version: str = Field(min_length=1)
    recovery_policy_version: str = Field(min_length=1)
    ambiguity_handling: Literal["preserve_unlabeled"]
    development_split_behavior: Literal["all_development"]
    split: DatasetSplitSettings


class ProjectConfig(BaseModel):
    """Foundation-stage configuration contract.

    Scientific parameters are intentionally absent until their corresponding
    implementation stages have approved values.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    config_version: Literal[1]
    project: ProjectSettings
    reproducibility: ReproducibilitySettings
    catalog: CatalogSettings | None = None
    acquisition: AcquisitionSettings | None = None
    expansion: ExpansionSettings | None = None
    preprocessing: PreprocessingSettings | None = None
    bls: BlsSettings | None = None
    dataset: DatasetSettings | None = None


ConfigInput = ProjectConfig | Mapping[str, Any]


def _validate_config(value: ConfigInput) -> ProjectConfig:
    if isinstance(value, ProjectConfig):
        return value
    try:
        return ProjectConfig.model_validate(value)
    except ValidationError as exc:
        raise ConfigurationError("Configuration does not match the project schema.") from exc


def load_config(path: str | Path) -> ProjectConfig:
    """Load and validate one YAML configuration file."""
    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"Could not load configuration: {config_path}") from exc

    if not isinstance(raw, Mapping):
        raise ConfigurationError("Configuration root must be a YAML mapping.")
    return _validate_config(raw)


def resolve_config(config: ConfigInput) -> dict[str, Any]:
    """Return a JSON-compatible, deterministic representation of a config."""
    validated = _validate_config(config)
    return validated.model_dump(mode="json", exclude_none=True)


def stable_config_json(config: ConfigInput) -> str:
    """Return canonical JSON suitable for a provenance record."""
    return canonical_json(resolve_config(config))


def config_hash(config: ConfigInput) -> str:
    """Return the content hash of the resolved scientific configuration."""
    return content_hash(resolve_config(config))
