"""Features derived strictly from frozen blind-BLS outputs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from transiteye.config import FeatureSettings


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def extract_bls_features(
    candidate: Mapping[str, Any],
    cadences: pd.DataFrame,
    periodogram: pd.DataFrame,
    group_candidates: pd.DataFrame,
    settings: FeatureSettings,
) -> dict[str, float]:
    """Summarize one candidate without consulting catalog or synthetic truth."""
    period = float(candidate["candidate_period"])
    duration = float(candidate["candidate_duration"])
    power = float(candidate["bls_power"])
    valid_time = cadences.loc[
        cadences["valid"].astype(bool) & np.isfinite(cadences["time"]), "time"
    ].to_numpy(dtype=float)
    if valid_time.size < 2:
        raise ValueError("At least two valid timestamps are required for BLS features.")
    baseline = float(np.max(valid_time) - np.min(valid_time))
    strongest = float(group_candidates["bls_power"].max())

    pg_period = periodogram["period"].to_numpy(dtype=float)
    pg_power = periodogram["power"].to_numpy(dtype=float)
    window = np.abs(pg_period - period) <= settings.bls_peak_neighborhood_fraction * period
    local_period = pg_period[window]
    local_power = pg_power[window]
    if local_power.size >= 3:
        floor = float(np.median(local_power))
        prominence = power - floor
        supported = local_power >= floor + max(prominence, 0.0) / 2
        width = (
            float(np.max(local_period[supported]) - np.min(local_period[supported]))
            if supported.sum() >= 2
            else float("nan")
        )
    else:
        prominence = float("nan")
        width = float("nan")

    relation = candidate.get("relation_to_stronger")
    harmonic = relation is not None and str(relation) not in {"", "None", "nan"}
    harmonic_ratio = candidate.get("detection_harmonic_ratio")
    harmonic_ratio_value = (
        float(harmonic_ratio)
        if harmonic_ratio is not None and np.isfinite(float(harmonic_ratio))
        else float("nan")
    )
    return {
        "bls_candidate_period": period,
        "bls_candidate_duration": duration,
        "bls_candidate_depth": float(candidate["candidate_depth"]),
        "bls_candidate_power": power,
        "bls_candidate_rank": float(candidate["candidate_rank"]),
        "bls_duty_cycle": _safe_ratio(duration, period),
        "bls_estimated_transit_count": float(np.floor(baseline / period) + 1),
        "bls_period_baseline_ratio": _safe_ratio(period, baseline),
        "bls_power_ratio_to_strongest": _safe_ratio(power, strongest),
        "bls_power_difference_from_strongest": strongest - power,
        "bls_local_peak_prominence": float(prominence),
        "bls_local_peak_width_days": width,
        "bls_has_harmonic_relation": float(harmonic),
        "bls_detector_harmonic_ratio": harmonic_ratio_value,
        "bls_effective_max_period_days": float(np.nanmax(pg_period)),
        "bls_observing_baseline_days": baseline,
    }
