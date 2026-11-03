"""Uncertainty-weighted Box Least Squares (BLS) periodogram search."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy.timeseries import BoxLeastSquares
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class BlsPeriodogram:
    """Results of a Box Least Squares period search."""

    periods: NDArray[np.float64]
    powers: NDArray[np.float64]
    durations: NDArray[np.float64]
    depths: NDArray[np.float64]
    transit_times: NDArray[np.float64]
    best_period: float
    best_duration_hours: float
    best_epoch_days: float
    best_depth_ppm: float
    best_snr: float
    sde: float


def run_bls(
    time_days: NDArray[np.float64],
    flux: NDArray[np.float64],
    flux_err: NDArray[np.float64] | None = None,
    min_period: float = 0.3,
    max_period: float | None = None,
    min_duration_hours: float = 0.5,
    max_duration_hours: float = 12.0,
    num_durations: int = 15,
    oversample: float = 5.0,
    device: str = "auto",
) -> BlsPeriodogram:
    """Run an uncertainty-weighted BLS search across trial periods and durations."""
    length = len(time_days)
    if length < 50:
        raise ValueError("BLS requires at least 50 cadences.")

    baseline = float(time_days[-1] - time_days[0])
    if baseline <= 0:
        raise ValueError("Time baseline must be positive.")

    if max_period is None:
        max_period = max(min_period + 0.1, 0.5 * baseline)

    # Durations grid in days
    min_dur_days = min_duration_hours / 24.0
    max_dur_days = min(max_duration_hours / 24.0, 0.25 * min_period)
    if max_dur_days <= min_dur_days:
        max_dur_days = min(min_dur_days * 1.5, np.nextafter(min_period, 0.0))

    durations_days = np.geomspace(min_dur_days, max_dur_days, num_durations)

    # Astropy BoxLeastSquares model
    model = BoxLeastSquares(time_days, flux, dy=flux_err)

    # Trial periods grid
    periods = model.autoperiod(
        durations_days,
        minimum_period=min_period,
        maximum_period=max_period,
        frequency_factor=oversample,
    )
    # Astropy's grid may overshoot requested limits by one grid step.
    periods = np.asarray(periods, dtype=np.float64)
    periods = periods[(periods >= min_period) & (periods <= max_period)]
    if periods.size == 0:
        raise ValueError("No trial periods fall within the requested BLS search range.")

    # Run BLS
    power_results = model.power(periods, durations_days)
    powers = np.asarray(power_results.power, dtype=np.float64)

    # Signal Detection Efficiency (SDE)
    mean_power = float(np.mean(powers))
    std_power = float(np.std(powers))
    max_idx = int(np.argmax(powers))
    best_power = float(powers[max_idx])
    sde = float((best_power - mean_power) / std_power) if std_power > 0 else 0.0

    best_period = float(periods[max_idx])
    best_duration = float(power_results.duration[max_idx]) * 24.0  # in hours
    best_epoch = float(power_results.transit_time[max_idx])
    best_depth = float(power_results.depth[max_idx]) * 1e6  # in ppm

    # Compute stats for SNR
    stats = model.compute_stats(
        best_period,
        best_duration / 24.0,
        best_epoch,
    )
    # Estimate depth SNR: depth / error_on_depth
    # Using stats depth and unphased residual scatter
    depth_val = float(stats["depth"][0]) if "depth" in stats else (best_depth / 1e6)
    in_transit_pts = sum(stats.get("per_transit_count", [1]))
    scatter = float(np.std(flux)) if len(flux) > 1 else 1.0
    snr = float(depth_val / (scatter / np.sqrt(max(1, in_transit_pts)))) if scatter > 0 else 0.0

    return BlsPeriodogram(
        periods=np.asarray(periods, dtype=np.float64),
        powers=powers,
        durations=np.asarray(power_results.duration, dtype=np.float64) * 24.0,
        depths=np.asarray(power_results.depth, dtype=np.float64) * 1e6,
        transit_times=np.asarray(power_results.transit_time, dtype=np.float64),
        best_period=best_period,
        best_duration_hours=best_duration,
        best_epoch_days=best_epoch,
        best_depth_ppm=best_depth,
        best_snr=max(0.0, snr),
        sde=sde,
    )
