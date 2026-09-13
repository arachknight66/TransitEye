"""Compact Lomb--Scargle and segment-local FFT summary features."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle  # type: ignore[import-untyped]

from transiteye.config import FeatureSettings


def _valid_series(cadences: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    required = {"time", "detrended_flux", "valid", "segment_id"}
    missing = required.difference(cadences.columns)
    if missing:
        raise ValueError(f"Processed cadence table lacks columns: {sorted(missing)}")
    mask = (
        cadences["valid"].astype(bool).to_numpy()
        & np.isfinite(cadences["time"].to_numpy(dtype=float))
        & np.isfinite(cadences["detrended_flux"].to_numpy(dtype=float))
        & (cadences["segment_id"].to_numpy(dtype=int) >= 0)
    )
    time = cadences.loc[mask, "time"].to_numpy(dtype=float)
    flux = cadences.loc[mask, "detrended_flux"].to_numpy(dtype=float)
    segment = cadences.loc[mask, "segment_id"].to_numpy(dtype=int)
    order = np.argsort(time, kind="stable")
    return time[order], flux[order], segment[order]


def _entropy(power: np.ndarray) -> float:
    clean = np.clip(np.asarray(power, dtype=float), 0, None)
    total = float(np.sum(clean))
    if clean.size < 2 or total <= 0:
        return float("nan")
    probability = clean / total
    positive = probability > 0
    return float(
        -np.sum(probability[positive] * np.log(probability[positive])) / np.log(clean.size)
    )


def _power_at(frequency: np.ndarray, power: np.ndarray, target: float) -> float:
    if frequency.size == 0 or target <= 0 or target < frequency[0] or target > frequency[-1]:
        return float("nan")
    return float(np.interp(target, frequency, power))


def _ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def lomb_scargle_spectrum(
    cadences: pd.DataFrame, settings: FeatureSettings
) -> tuple[np.ndarray, np.ndarray]:
    """Return the deterministic LS grid and power for one processed curve."""
    time, flux, _ = _valid_series(cadences)
    if time.size < 3 or np.ptp(time) <= 0 or np.std(flux) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    unique_time, indices = np.unique(time, return_index=True)
    flux = flux[indices]
    if unique_time.size < 3:
        return np.array([], dtype=float), np.array([], dtype=float)
    baseline = float(np.ptp(unique_time))
    cadence = float(np.median(np.diff(unique_time)))
    minimum = settings.lomb_scargle.minimum_cycles_per_baseline / baseline
    maximum = min(settings.lomb_scargle.maximum_frequency_per_day, 0.5 / cadence)
    if not maximum > minimum:
        return np.array([], dtype=float), np.array([], dtype=float)
    count = max(
        3,
        int(math.ceil((maximum - minimum) * baseline * settings.lomb_scargle.samples_per_peak)) + 1,
    )
    frequency = np.linspace(minimum, maximum, count, dtype=float)
    power = LombScargle(unique_time, flux - np.mean(flux)).power(frequency)
    finite = np.isfinite(power)
    return frequency[finite], np.asarray(power[finite], dtype=float)


def extract_lomb_scargle_features(
    frequency: np.ndarray,
    power: np.ndarray,
    candidate_period: float,
    settings: FeatureSettings,
) -> dict[str, float]:
    """Summarize a precomputed truth-blind Lomb--Scargle spectrum."""
    names = (
        "ls_dominant_frequency_per_day",
        "ls_dominant_period_days",
        "ls_dominant_power",
        "ls_second_independent_power",
        "ls_primary_secondary_power_ratio",
        "ls_spectral_entropy",
        "ls_power_concentration",
        "ls_power_at_bls_frequency",
        "ls_power_at_twice_bls_frequency",
        "ls_power_at_half_bls_frequency",
        "ls_bls_to_global_power_ratio",
    )
    if frequency.size == 0:
        return dict.fromkeys(names, float("nan"))
    order = np.argsort(power)[::-1]
    primary_index = int(order[0])
    primary_frequency = float(frequency[primary_index])
    primary_power = float(power[primary_index])
    independent = [
        int(index)
        for index in order[1:]
        if abs(float(frequency[index]) / primary_frequency - 1)
        > settings.lomb_scargle.independent_peak_fraction
    ]
    secondary = float(power[independent[0]]) if independent else float("nan")
    candidate_frequency = 1.0 / candidate_period
    at_candidate = _power_at(frequency, power, candidate_frequency)
    total = float(np.sum(np.clip(power, 0, None)))
    return {
        "ls_dominant_frequency_per_day": primary_frequency,
        "ls_dominant_period_days": 1.0 / primary_frequency,
        "ls_dominant_power": primary_power,
        "ls_second_independent_power": secondary,
        "ls_primary_secondary_power_ratio": _ratio(primary_power, secondary),
        "ls_spectral_entropy": _entropy(power),
        "ls_power_concentration": _ratio(primary_power, total),
        "ls_power_at_bls_frequency": at_candidate,
        "ls_power_at_twice_bls_frequency": _power_at(frequency, power, 2 * candidate_frequency),
        "ls_power_at_half_bls_frequency": _power_at(frequency, power, 0.5 * candidate_frequency),
        "ls_bls_to_global_power_ratio": _ratio(at_candidate, primary_power),
    }


def _regular_fft_input(
    cadences: pd.DataFrame, settings: FeatureSettings
) -> tuple[np.ndarray, float] | None:
    time, flux, segment = _valid_series(cadences)
    best: tuple[np.ndarray, float] | None = None
    for segment_id in np.unique(segment):
        selected = segment == segment_id
        segment_time = time[selected]
        segment_flux = flux[selected]
        if segment_time.size < settings.fft.minimum_segment_points:
            continue
        difference = np.diff(segment_time)
        cadence = float(np.median(difference))
        if cadence <= 0:
            continue
        # A larger internal gap starts a new FFT block; it is never interpolated.
        boundary = (
            np.flatnonzero(difference > (settings.fft.maximum_short_gap_cadences + 1.5) * cadence)
            + 1
        )
        for time_block, flux_block in zip(
            np.split(segment_time, boundary), np.split(segment_flux, boundary), strict=True
        ):
            if time_block.size < settings.fft.minimum_segment_points:
                continue
            grid = np.arange(time_block[0], time_block[-1] + cadence / 2, cadence)
            nearest = np.searchsorted(time_block, grid)
            left = np.clip(nearest - 1, 0, time_block.size - 1)
            right = np.clip(nearest, 0, time_block.size - 1)
            distance = np.minimum(abs(grid - time_block[left]), abs(time_block[right] - grid))
            if np.any(distance > (settings.fft.maximum_short_gap_cadences + 0.5) * cadence):
                continue
            regular_flux = np.interp(grid, time_block, flux_block)
            if best is None or regular_flux.size > best[0].size:
                best = regular_flux, cadence
    return best


def fft_spectrum(
    cadences: pd.DataFrame, settings: FeatureSettings
) -> tuple[np.ndarray, np.ndarray]:
    """Build a temporary regular representation for the longest defensible block."""
    regular = _regular_fft_input(cadences, settings)
    if regular is None:
        return np.array([], dtype=float), np.array([], dtype=float)
    flux, cadence = regular
    centered = flux - np.mean(flux)
    if np.std(centered) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    frequency = np.fft.rfftfreq(centered.size, d=cadence)[1:]
    power = np.abs(np.fft.rfft(centered)[1:]) ** 2
    finite = np.isfinite(power) & np.isfinite(frequency) & (frequency > 0)
    return frequency[finite], power[finite]


def extract_fft_features(
    frequency: np.ndarray,
    power: np.ndarray,
    candidate_period: float,
    settings: FeatureSettings,
) -> dict[str, float]:
    """Summarize the temporary FFT branch; unavailable values remain missing."""
    names = (
        "fft_dominant_frequency_per_day",
        "fft_dominant_power",
        "fft_second_peak_power",
        "fft_power_concentration",
        "fft_spectral_entropy",
        "fft_low_band_power_fraction",
        "fft_mid_band_power_fraction",
        "fft_high_band_power_fraction",
        "fft_power_at_bls_frequency",
        "fft_power_at_twice_bls_frequency",
        "fft_power_at_half_bls_frequency",
        "fft_bls_to_global_power_ratio",
    )
    if frequency.size == 0:
        return {"fft_usable": 0.0, **dict.fromkeys(names, float("nan"))}
    order = np.argsort(power)[::-1]
    primary = float(power[order[0]])
    second = float(power[order[1]]) if order.size >= 2 else float("nan")
    total = float(np.sum(np.clip(power, 0, None)))
    low_edge = settings.fft.low_band_max_frequency_per_day
    mid_edge = settings.fft.mid_band_max_frequency_per_day
    candidate_frequency = 1.0 / candidate_period
    at_candidate = _power_at(frequency, power, candidate_frequency)
    return {
        "fft_usable": 1.0,
        "fft_dominant_frequency_per_day": float(frequency[order[0]]),
        "fft_dominant_power": primary,
        "fft_second_peak_power": second,
        "fft_power_concentration": _ratio(primary, total),
        "fft_spectral_entropy": _entropy(power),
        "fft_low_band_power_fraction": _ratio(float(power[frequency < low_edge].sum()), total),
        "fft_mid_band_power_fraction": _ratio(
            float(power[(frequency >= low_edge) & (frequency < mid_edge)].sum()), total
        ),
        "fft_high_band_power_fraction": _ratio(float(power[frequency >= mid_edge].sum()), total),
        "fft_power_at_bls_frequency": at_candidate,
        "fft_power_at_twice_bls_frequency": _power_at(frequency, power, 2 * candidate_frequency),
        "fft_power_at_half_bls_frequency": _power_at(frequency, power, 0.5 * candidate_frequency),
        "fft_bls_to_global_power_ratio": _ratio(at_candidate, primary),
    }
