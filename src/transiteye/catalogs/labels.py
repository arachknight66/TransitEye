"""Typed mapping from TOI dispositions to the approved project label policy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class CatalogDisposition(StrEnum):
    """Known TFOPWG disposition codes in the TOI table."""

    APC = "APC"
    CP = "CP"
    FA = "FA"
    FP = "FP"
    KP = "KP"
    PC = "PC"
    UNKNOWN = "UNKNOWN"


class InternalLabel(StrEnum):
    """Project-level supervised label state."""

    GOLD_POSITIVE = "gold_positive"
    GOLD_NEGATIVE = "gold_negative"
    UNLABELED_SECONDARY = "unlabeled_secondary"
    UNKNOWN = "unknown"


class UnknownDispositionError(ValueError):
    """Raised when a strict mapping receives an unexpected source disposition."""


@dataclass(frozen=True)
class DispositionMapping:
    """Preserved source disposition and its internal project interpretation."""

    original_value: str | None
    disposition: CatalogDisposition
    internal_label: InternalLabel

    @property
    def subgroup(self) -> str:
        """Retain CP/KP/FP/FA identity even when labels are binary."""
        return self.disposition.value.lower()

    @property
    def is_known(self) -> bool:
        """Return whether the source disposition was part of the approved policy."""
        return self.disposition is not CatalogDisposition.UNKNOWN


_LABEL_MAP: dict[CatalogDisposition, InternalLabel] = {
    CatalogDisposition.CP: InternalLabel.GOLD_POSITIVE,
    CatalogDisposition.KP: InternalLabel.GOLD_POSITIVE,
    CatalogDisposition.FP: InternalLabel.GOLD_NEGATIVE,
    CatalogDisposition.FA: InternalLabel.GOLD_NEGATIVE,
    CatalogDisposition.PC: InternalLabel.UNLABELED_SECONDARY,
    CatalogDisposition.APC: InternalLabel.UNLABELED_SECONDARY,
}


def map_disposition(value: str | None, *, strict: bool = False) -> DispositionMapping:
    """Map a TFOPWG code while exposing unknown or missing values explicitly."""
    original_value = value.strip() if value is not None else None
    normalized = original_value.upper() if original_value else None
    try:
        disposition = CatalogDisposition(normalized) if normalized else CatalogDisposition.UNKNOWN
    except ValueError:
        disposition = CatalogDisposition.UNKNOWN

    if disposition is CatalogDisposition.UNKNOWN and strict:
        raise UnknownDispositionError(f"Unexpected or missing TFOPWG disposition: {value!r}")
    return DispositionMapping(
        original_value=original_value,
        disposition=disposition,
        internal_label=_LABEL_MAP.get(disposition, InternalLabel.UNKNOWN),
    )
