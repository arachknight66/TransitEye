"""Baseline-aware, blind Astropy BoxLeastSquares searches."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from astropy.timeseries import BoxLeastSquares  # type: ignore[import-untyped]

from transiteye.config import BlsSettings
from transiteye.serialization import content_hash


class BlsInputError(ValueError):
    pass


@dataclass(frozen=True)
class BlsResult:
    periodogram: pd.DataFrame
    best_period: float
    best_duration: float
    best_epoch: float
    best_depth: float
    config_hash: str
    observation_group_id: str


def run_bls(
    cadences: pd.DataFrame,
    *,
    settings: BlsSettings,
    observation_group_id: str,
    preprocessing_hash: str,
) -> BlsResult:
    """Run blind BLS; catalog ephemerides are deliberately not accepted."""
    needed = {"time", "detrended_flux", "valid"}
    if not needed.issubset(cadences):
        raise BlsInputError("Processed cadence columns are missing.")
    frame = cadences.loc[
        cadences.valid & np.isfinite(cadences.time) & np.isfinite(cadences.detrended_flux)
    ].sort_values("time")
    time = frame.time.to_numpy(float)
    flux = frame.detrended_flux.to_numpy(float)
    if len(time) < 20 or np.any(np.diff(time) <= 0):
        raise BlsInputError("Too few or non-increasing valid timestamps.")
    baseline = time[-1] - time[0]
    maximum = min(settings.max_period_days, baseline / settings.min_transits)
    if maximum <= settings.min_period_days:
        raise BlsInputError("Baseline is insufficient for configured period range.")
    durations = np.array(
        [d for d in settings.durations_days if d < settings.min_period_days], float
    )
    if not len(durations):
        raise BlsInputError("No duration is valid for minimum period.")
    model = BoxLeastSquares(time, flux)
    power = model.autopower(
        durations,
        minimum_period=settings.min_period_days,
        maximum_period=maximum,
        frequency_factor=settings.frequency_factor,
    )
    pg = pd.DataFrame(
        {
            "period": np.asarray(power.period),
            "power": power.power,
            "duration": np.asarray(power.duration),
            "epoch": np.asarray(power.transit_time),
            "depth": power.depth,
        }
    )
    best = int(np.nanargmax(power.power))
    config_hash = content_hash(
        {"bls": settings.model_dump(mode="json"), "preprocessing_hash": preprocessing_hash}
    )
    return BlsResult(
        pg,
        float(power.period[best]),
        float(power.duration[best]),
        float(power.transit_time[best]),
        float(power.depth[best]),
        config_hash,
        observation_group_id,
    )
