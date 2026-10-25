"""Lomb-Scargle periodogram variability search and harmonic relations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy.timeseries import LombScargle
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class VariabilitySummary:
    """Stellar variability diagnostics derived from Lomb-Scargle analysis."""

    best_period_days: float
    peak_power: float
    false_alarm_prob: float
    periods: NDArray[np.float64]
    powers: NDArray[np.float64]


def run_variability_search(
    time_days: NDArray[np.float64],
    flux: NDArray[np.float64],
    flux_err: NDArray[np.float64] | None = None,
    min_period: float = 0.1,
    max_period: float | None = None,
    samples_per_peak: int = 5,
    device: str = "auto",
) -> VariabilitySummary:
    """Run Lomb-Scargle variability search to capture rotation, pulsation, and out-of-transit variations."""
    from transiteye.execution.compute import resolve_device

    _ = resolve_device(device)
    baseline = float(time_days[-1] - time_days[0])
    if max_period is None:
        max_period = max(min_period + 0.1, 0.5 * baseline)

    ls = LombScargle(time_days, flux, dy=flux_err, normalization="standard")

    min_freq = 1.0 / max_period
    max_freq = 1.0 / min_period

    freqs, powers = ls.autopower(
        minimum_frequency=min_freq,
        maximum_frequency=max_freq,
        samples_per_peak=samples_per_peak,
    )

    periods = 1.0 / freqs
    # Sort periods ascending
    sort_idx = np.argsort(periods)
    periods = periods[sort_idx]
    powers = powers[sort_idx]

    max_idx = int(np.argmax(powers))
    best_period = float(periods[max_idx])
    peak_power = float(powers[max_idx])

    try:
        fap = float(ls.false_alarm_probability(peak_power))
    except Exception:
        fap = 1.0

    return VariabilitySummary(
        best_period_days=best_period,
        peak_power=peak_power,
        false_alarm_prob=max(0.0, min(1.0, fap)),
        periods=np.asarray(periods, dtype=np.float64),
        powers=np.asarray(powers, dtype=np.float64),
    )


def compute_harmonic_distance(period_a: float, period_b: float, max_harmonic: int = 4) -> float:
    """Return the minimum fractional distance between period_a and any integer/half-harmonic of period_b."""
    if period_a <= 0 or period_b <= 0:
        return 1.0
    ratios = [1.0, 0.5, 2.0, 1.0 / 3.0, 3.0, 0.25, 4.0]
    best_dist = 1.0
    for ratio in ratios:
        target = period_b * ratio
        dist = abs(period_a - target) / period_a
        if dist < best_dist:
            best_dist = dist
    return float(best_dist)
