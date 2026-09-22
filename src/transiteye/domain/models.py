"""Typed, serializable contracts for scientific analysis results."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import StrEnum
from typing import Any

import numpy as np
from numpy.typing import NDArray


class AnalysisStatus(StrEnum):
    """Terminal or intermediate state for an analysis request."""

    PENDING = "pending"
    COMPLETE = "complete"
    NO_SIGNIFICANT_SIGNAL = "no_significant_signal"
    INSUFFICIENT_DATA = "insufficient_data"
    FAILED = "failed"


class MissingReason(StrEnum):
    """Reason a value or diagnostic is intentionally unavailable."""

    NOT_APPLICABLE = "not_applicable"
    NOT_COMPUTED = "not_computed"
    INSUFFICIENT_DATA = "insufficient_data"
    FIT_FAILED = "fit_failed"
    SOURCE_DATA_MISSING = "source_data_missing"


class MorphologyClass(StrEnum):
    """Mutually exclusive morphology classes, independent from source attribution."""

    TRANSIT_LIKE = "transit_like"
    ECLIPSE_LIKE = "eclipse_like"
    STELLAR_VARIABILITY = "stellar_variability"
    OTHER_ARTIFACT = "other_artifact"
    UNCERTAIN = "uncertain"


class SourceAssessment(StrEnum):
    """Assessment of whether a candidate originates at its requested target."""

    TARGET_CONSISTENT = "target_consistent"
    BLEND_SUSPECTED = "blend_suspected"
    UNRESOLVED = "unresolved"
    NOT_ASSESSED = "not_assessed"


@dataclass(frozen=True, slots=True)
class Provenance:
    """Identifies input data and the conditions under which it was used."""

    source_uri: str
    source_path: str | None
    checksum_sha256: str | None
    product_type: str
    target_id: str | None
    sector: int | None
    time_system: str | None
    time_reference: str | None
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LightCurve:
    """A validated, delivered TESS light curve with no preprocessing applied."""

    time_days: NDArray[np.float64]
    flux: NDArray[np.float64]
    flux_error: NDArray[np.float64] | None
    quality: NDArray[np.int64] | None
    cadence_number: NDArray[np.int64] | None
    centroid_column: NDArray[np.float64] | None
    centroid_row: NDArray[np.float64] | None
    flux_kind: str
    flux_unit: str | None
    provenance: Provenance
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        length = len(self.time_days)
        if length == 0:
            raise ValueError("A light curve must contain at least one cadence.")
        if len(self.flux) != length:
            raise ValueError("time_days and flux must have equal length.")
        for name in (
            "flux_error",
            "quality",
            "cadence_number",
            "centroid_column",
            "centroid_row",
        ):
            values = getattr(self, name)
            if values is not None and len(values) != length:
                raise ValueError(f"{name} must have the same length as time_days.")


@dataclass(frozen=True, slots=True)
class Interval:
    """A two-sided parameter interval with explicit probability coverage."""

    lower: float | None
    upper: float | None
    coverage: float
    missing_reason: MissingReason | None = None

    def __post_init__(self) -> None:
        if not 0 < self.coverage < 1:
            raise ValueError("coverage must be strictly between zero and one.")
        if self.missing_reason is None and (self.lower is None or self.upper is None):
            raise ValueError("An available interval requires lower and upper bounds.")


@dataclass(frozen=True, slots=True)
class Candidate:
    """A periodic or quasi-periodic candidate, before classification."""

    candidate_id: str
    period_days: float | None
    epoch_days: float | None
    duration_hours: float | None
    depth_ppm: float | None
    observed_event_count: int | None
    detection_snr: float | None
    search_significance: float | None
    aliases_days: tuple[float, ...] = ()
    flags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Classification:
    """Calibrated morphology probabilities and independent source assessment."""

    probabilities: Mapping[MorphologyClass, float]
    predicted_class: MorphologyClass | None
    source_assessment: SourceAssessment
    abstained: bool
    model_version: str | None

    def __post_init__(self) -> None:
        if any(probability < 0 or probability > 1 for probability in self.probabilities.values()):
            raise ValueError("Classification probabilities must be in [0, 1].")
        total = sum(self.probabilities.values())
        if self.probabilities and not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError("Classification probabilities must sum to one.")
        if self.abstained and self.predicted_class is not None:
            raise ValueError("An abstained classification cannot have a predicted class.")


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    """The complete versioned result contract returned by backend services."""

    analysis_id: str
    status: AnalysisStatus
    provenance: Provenance
    candidates: tuple[Candidate, ...]
    classifications: Mapping[str, Classification] = field(default_factory=dict)
    parameter_intervals: Mapping[str, Mapping[str, Interval]] = field(default_factory=dict)
    missing_reasons: Mapping[str, MissingReason] = field(default_factory=dict)
    configuration_version: str = "unconfigured"
    schema_version: str = "1"
    error_message: str | None = None

    def __post_init__(self) -> None:
        candidate_ids = {candidate.candidate_id for candidate in self.candidates}
        if not set(self.classifications).issubset(candidate_ids):
            raise ValueError("Classifications must reference returned candidates.")
        if self.status is AnalysisStatus.FAILED and not self.error_message:
            raise ValueError("Failed analyses require an error_message.")


def as_jsonable(value: Any) -> Any:
    """Convert domain records and NumPy scalars/arrays to JSON-compatible values."""

    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return as_jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key.value if isinstance(key, StrEnum) else key): as_jsonable(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple | list):
        return [as_jsonable(item) for item in value]
    return value
