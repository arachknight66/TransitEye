"""Deterministic synthetic signals applied to in-memory TESS source records."""

from __future__ import annotations

import numpy as np

from transiteye.config import DemoVariantSettings
from transiteye.preprocessing.pipeline import LightCurveRecord


def _trapezoid_multiplier(
    time: np.ndarray, *, period: float, epoch: float, duration: float, depth: float
) -> np.ndarray:
    distance = np.abs((time - epoch + period / 2) % period - period / 2)
    half_duration = duration / 2
    ingress = duration * 0.2
    weight = np.clip((half_duration - distance) / ingress, 0.0, 1.0)
    return 1.0 - depth * weight


def inject_variant(
    source: LightCurveRecord,
    *,
    variant: DemoVariantSettings,
    epoch: float | None,
) -> LightCurveRecord:
    """Return a derived source while leaving the base record and raw FITS untouched."""
    frame = source.data.copy(deep=True)
    if variant.event_class == "no_injection_control":
        return LightCurveRecord(frame, source.metadata.copy())
    if (
        variant.period_days is None
        or variant.duration_days is None
        or variant.depth_or_amplitude is None
        or epoch is None
    ):
        raise ValueError("Injected variant parameters and epoch must be complete.")
    time = frame["time"].to_numpy(float)
    if variant.event_class == "planet_like":
        multiplier = _trapezoid_multiplier(
            time,
            period=variant.period_days,
            epoch=epoch,
            duration=variant.duration_days,
            depth=variant.depth_or_amplitude,
        )
    elif variant.event_class == "eclipsing_binary_like":
        primary = _trapezoid_multiplier(
            time,
            period=variant.period_days,
            epoch=epoch,
            duration=variant.duration_days,
            depth=variant.depth_or_amplitude,
        )
        secondary = _trapezoid_multiplier(
            time,
            period=variant.period_days,
            epoch=epoch + variant.period_days / 2,
            duration=variant.duration_days,
            depth=float(variant.secondary_depth or 0.0),
        )
        multiplier = primary * secondary
    elif variant.event_class == "sinusoidal_variability_like":
        multiplier = 1.0 - variant.depth_or_amplitude * np.cos(
            2 * np.pi * (time - epoch) / variant.period_days
        )
    else:  # pragma: no cover - typed configuration prevents this branch.
        raise ValueError(f"Unsupported synthetic event class: {variant.event_class}")
    for column in ("pdcsap_flux", "sap_flux"):
        frame[column] = frame[column].to_numpy(float) * multiplier
    return LightCurveRecord(frame, source.metadata.copy())
