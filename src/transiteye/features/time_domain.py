"""Robust candidate-centered time-domain summaries."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from transiteye.config import FeatureSettings


def _finite(value: float) -> float:
    return float(value) if np.isfinite(value) else float("nan")


def _mad(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    median = np.median(values)
    return _finite(float(np.median(np.abs(values - median))))


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def _valid_series(cadences: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    required = {"time", "detrended_flux", "valid"}
    missing = required.difference(cadences.columns)
    if missing:
        raise ValueError(f"Processed cadence table lacks columns: {sorted(missing)}")
    mask = (
        cadences["valid"].astype(bool).to_numpy()
        & np.isfinite(cadences["time"].to_numpy(dtype=float))
        & np.isfinite(cadences["detrended_flux"].to_numpy(dtype=float))
    )
    time = cadences.loc[mask, "time"].to_numpy(dtype=float)
    flux = cadences.loc[mask, "detrended_flux"].to_numpy(dtype=float)
    order = np.argsort(time, kind="stable")
    return time[order], flux[order]


def extract_time_domain_features(
    candidate: Mapping[str, Any], cadences: pd.DataFrame, settings: FeatureSettings
) -> dict[str, float]:
    """Extract truth-blind statistics around one blind-BLS hypothesis."""
    time, flux = _valid_series(cadences)
    if flux.size == 0:
        raise ValueError("No finite, valid processed cadences are available.")
    median = float(np.median(flux))
    centered = flux - float(np.mean(flux))
    std = float(np.std(flux))
    skewness = _safe_ratio(float(np.mean(centered**3)), std**3)
    kurtosis = _safe_ratio(float(np.mean(centered**4)), std**4)
    if np.isfinite(kurtosis):
        kurtosis -= 3.0
    p05, p25, p75, p95 = np.percentile(flux, [5, 25, 75, 95])
    result = {
        "td_global_median": median,
        "td_global_std": std,
        "td_global_mad": _mad(flux),
        "td_global_robust_rms": 1.4826 * _mad(flux),
        "td_global_iqr": float(p75 - p25),
        "td_global_skewness": skewness,
        "td_global_excess_kurtosis": kurtosis,
        "td_global_p05": float(p05),
        "td_global_p95": float(p95),
        "td_global_robust_range": float(p95 - p05),
    }

    period = float(candidate["candidate_period"])
    epoch = float(candidate["candidate_epoch"])
    duration = float(candidate["candidate_duration"])
    if not (period > 0 and duration > 0 and duration < period):
        raise ValueError("Candidate period and duration are physically incompatible.")
    signed_phase = (time - epoch + period / 2) % period - period / 2
    in_transit = np.abs(signed_phase) <= duration / 2
    out_transit = ~in_transit
    in_flux = flux[in_transit]
    out_flux = flux[out_transit]
    enough_in = in_flux.size >= settings.minimum_in_transit_points
    in_median = float(np.median(in_flux)) if enough_in else float("nan")
    out_median = float(np.median(out_flux)) if out_flux.size else float("nan")
    depth = out_median - in_median
    in_scatter = 1.4826 * _mad(in_flux) if enough_in else float("nan")
    out_scatter = 1.4826 * _mad(out_flux)

    event_numbers = np.floor((time - epoch) / period + 0.5).astype(np.int64)
    event_depths: list[float] = []
    supported_events: list[int] = []
    for number in np.unique(event_numbers[in_transit]):
        event_mask = in_transit & (event_numbers == number)
        if int(event_mask.sum()) >= settings.minimum_in_transit_points:
            event_depths.append(out_median - float(np.median(flux[event_mask])))
            supported_events.append(int(number))
    depth_values = np.asarray(event_depths, dtype=float)

    local_multiple = settings.local_baseline_duration_multiples
    pre = (signed_phase < -duration / 2) & (signed_phase >= -(0.5 + local_multiple) * duration)
    post = (signed_phase > duration / 2) & (signed_phase <= (0.5 + local_multiple) * duration)
    pre_post = (
        float(np.median(flux[pre])) - float(np.median(flux[post]))
        if pre.any() and post.any()
        else float("nan")
    )
    ingress = in_transit & (signed_phase < 0)
    egress = in_transit & (signed_phase >= 0)
    asymmetry = (
        float(np.median(flux[ingress])) - float(np.median(flux[egress]))
        if ingress.sum() >= 2 and egress.sum() >= 2
        else float("nan")
    )

    odd = depth_values[np.asarray(supported_events) % 2 != 0]
    even = depth_values[np.asarray(supported_events) % 2 == 0]
    if odd.size and even.size:
        odd_depth = float(np.median(odd))
        even_depth = float(np.median(even))
        odd_even = odd_depth - even_depth
        odd_even_normalized = _safe_ratio(odd_even, abs(odd_depth) + abs(even_depth))
    else:
        odd_even = float("nan")
        odd_even_normalized = float("nan")

    result.update(
        {
            "td_in_transit_count": float(in_flux.size),
            "td_out_transit_count": float(out_flux.size),
            "td_in_transit_median": in_median,
            "td_out_transit_median": out_median,
            "td_observed_depth": depth,
            "td_depth_significance": _safe_ratio(depth, out_scatter),
            "td_in_out_scatter_ratio": _safe_ratio(in_scatter, out_scatter),
            "td_transit_occupancy": float(in_flux.size / flux.size),
            "td_observed_transit_windows": float(len(supported_events)),
            "td_per_transit_depth_mad": _mad(depth_values),
            "td_transit_to_transit_scatter": (
                float(np.std(depth_values)) if depth_values.size >= 2 else float("nan")
            ),
            "td_pre_post_flux_difference": pre_post,
            "td_ingress_egress_asymmetry": asymmetry,
            "td_odd_even_depth_difference": odd_even,
            "td_odd_even_depth_normalized_difference": odd_even_normalized,
        }
    )
    return {name: _finite(value) for name, value in result.items()}
