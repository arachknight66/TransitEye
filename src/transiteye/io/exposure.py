"""Explicit, per-product exposure estimates from preserved FITS keywords."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from transiteye.domain.models import LightCurve


@dataclass(frozen=True, slots=True)
class ExposureResolution:
    days: float | None
    source: str | None
    status: Literal["frame_sum_boxcar_approximation", "user_supplied", "missing"]


def resolve_exposure(
    curve: LightCurve, *, override_seconds: float | None = None,
) -> ExposureResolution:
    """Resolve one cadence's photon-collection duration, never its sampling interval.

    `INT_TIME * NUM_FRM` sums frame integration durations. Treating that sum
    as a continuous boxcar is an explicit approximation because readout gaps
    separate frames. TIMEDEL and aggregate EXPOSURE/LIVETIME are not used.
    """
    if override_seconds is not None:
        seconds = float(override_seconds)
        if not np.isfinite(seconds) or not 0 < seconds < 86400:
            raise ValueError("override_seconds must be finite and between zero and one day.")
        return ExposureResolution(seconds / 86400.0, "override_seconds", "user_supplied")

    metadata = curve.provenance.metadata
    integration = metadata.get("INT_TIME")
    frames = metadata.get("NUM_FRM")
    if integration is None or frames is None:
        return ExposureResolution(None, None, "missing")
    try:
        integration_seconds = float(integration)
    except (TypeError, ValueError) as error:
        raise ValueError("INT_TIME must be a finite positive number of seconds.") from error
    try:
        frame_count = float(frames)
    except (TypeError, ValueError) as error:
        raise ValueError("NUM_FRM must be a positive integer.") from error
    if not np.isfinite(integration_seconds) or integration_seconds <= 0:
        raise ValueError("INT_TIME must be a finite positive number of seconds.")
    if not np.isfinite(frame_count) or frame_count < 1 or not frame_count.is_integer():
        raise ValueError("NUM_FRM must be a positive integer.")
    total_seconds = integration_seconds * frame_count
    if not np.isfinite(total_seconds) or total_seconds >= 86400:
        raise ValueError("INT_TIME*NUM_FRM must be shorter than one day.")
    return ExposureResolution(
        total_seconds / 86400.0, "INT_TIME*NUM_FRM", "frame_sum_boxcar_approximation",
    )
