"""Typed B035 feature-column contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ColumnRole(StrEnum):
    """Explicit role used to prevent accidental model-input selection."""

    IDENTITY = "identity"
    FEATURE = "feature"
    LABEL = "label"
    EVALUATION = "evaluation"
    PROVENANCE = "provenance"


@dataclass(frozen=True)
class FeatureDefinition:
    """One stable registry entry."""

    name: str
    role: ColumnRole
    group: str
    description: str
    dtype: str = "float64"

    def portable_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "role": self.role.value,
            "group": self.group,
            "description": self.description,
            "dtype": self.dtype,
        }
