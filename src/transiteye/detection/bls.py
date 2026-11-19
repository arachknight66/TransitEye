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


class BlsSearchBudgetExceeded(ValueError):
    """The requested BLS period grid exceeds its explicit trial-period budget."""


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
    *,
    max_trial_periods: int | None = None,
) -> BlsPeriodogram:
    """Run BLS, rejecting an oversized period grid when a budget is supplied."""
    if max_trial_periods is not None and (
        isinstance(max_trial_periods, bool)
        or not isinstance(max_trial_periods, (int, np.integer))
        or max_trial_periods < 1
    ):
        raise ValueError("max_trial_periods must be a positive integer or None.")

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

    if max_trial_periods is not None:
        # Astropy's autoperiod uses df = frequency_factor * min(duration) / baseline**2.
        # Its rounded endpoint can add at most one period outside our requested bounds.
        model_baseline = float(np.ptp(time_days))
        if (
            np.isfinite(model_baseline)
            and model_baseline > 0
            and np.isfinite(min_period)
            and np.isfinite(max_period)
            and np.isfinite(oversample)
            and min_period > 0
            and max_period >= min_period
            and oversample > 0
        ):
            frequency_step = oversample * float(np.min(durations_days)) / model_baseline**2
            frequency_span = 1.0 / min_period - 1.0 / max_period
            estimated_steps = frequency_span / frequency_step if frequency_step > 0 else float("inf")
            if estimated_steps > max_trial_periods + 1:
                raise BlsSearchBudgetExceeded(
                    f"BLS trial-period grid exceeds max_trial_periods={max_trial_periods}; "
                    "reduce the requested search range or change the explicit budget."
                )

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
    if max_trial_periods is not None and periods.size > max_trial_periods:
        raise BlsSearchBudgetExceeded(
            f"BLS trial-period grid has {periods.size} periods, exceeding "
            f"max_trial_periods={max_trial_periods}."
        )

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
