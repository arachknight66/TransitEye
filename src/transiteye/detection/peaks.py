"""Candidate peak refinement, alias testing, odd/even checks, and iterative residual search."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy.timeseries import BoxLeastSquares

from transiteye.detection.bls import BlsPeriodogram, run_bls
from transiteye.detection.variability import VariabilitySummary, run_variability_search
from transiteye.domain.models import Candidate
from transiteye.preprocessing.detrend import DualLightCurve


@dataclass(frozen=True, slots=True)
class DetectionResult:
    """Complete blind detection summary including BLS periodograms, variability, and candidates."""

    candidates: tuple[Candidate, ...]
    primary_bls: BlsPeriodogram
    variability: VariabilitySummary
    residual_bls: tuple[BlsPeriodogram, ...]


def find_candidates(
    dual_light_curve: DualLightCurve,
    max_candidates: int = 3,
    snr_threshold: float = 6.0,
    device: str = "auto",
    min_period: float = 0.3,
    max_period: float | None = None,
    min_duration_hours: float = 0.5,
    max_duration_hours: float = 12.0,
    num_durations: int = 15,
    frequency_factor: float = 5.0,
) -> DetectionResult:
    """Search a light curve for periodic transit/eclipse candidates and stellar variability.

    Iteratively extracts up to `max_candidates` distinct dip signals from residual light curves.
    """
    time_days = dual_light_curve.prepared.time_days
    flux_detrended = dual_light_curve.detrended_flux
    flux_err = dual_light_curve.prepared.flux_error_normalized
    target_id = dual_light_curve.prepared.provenance.target_id or "target"

    length = len(time_days)
    baseline = float(time_days[-1] - time_days[0]) if length > 1 else 0.0

    # Guardrails: require at least 200 valid points and 2.0 days baseline
    if length < 200 or baseline < 2.0:
        # Return empty detection result with dummy periodograms
        empty_bls = BlsPeriodogram(
            periods=np.array([1.0]),
            powers=np.array([0.0]),
            durations=np.array([1.0]),
            depths=np.array([0.0]),
            transit_times=np.array([time_days[0] if length > 0 else 0.0]),
            best_period=1.0,
            best_duration_hours=1.0,
            best_epoch_days=time_days[0] if length > 0 else 0.0,
            best_depth_ppm=0.0,
            best_snr=0.0,
            sde=0.0,
        )
        empty_var = VariabilitySummary(
            best_period_days=1.0,
            peak_power=0.0,
            false_alarm_prob=1.0,
            periods=np.array([1.0]),
            powers=np.array([0.0]),
        )
        return DetectionResult(
            candidates=(),
            primary_bls=empty_bls,
            variability=empty_var,
            residual_bls=(),
        )

    # 1. Run variability search on minimally processed (normalized) light curve
    variability = run_variability_search(
        time_days,
        dual_light_curve.prepared.flux_normalized,
        flux_err=flux_err,
        device=device,
    )

    # 2. Run primary BLS search
    primary_bls = run_bls(
        time_days,
        flux_detrended,
        flux_err=flux_err,
        device=device,
        min_period=min_period,
        max_period=max_period,
        min_duration_hours=min_duration_hours,
        max_duration_hours=max_duration_hours,
        num_durations=num_durations,
        oversample=frequency_factor,
    )

    candidates: list[Candidate] = []
    residual_bls: list[BlsPeriodogram] = []
    curr_flux = np.copy(flux_detrended)
    masked_mask = np.zeros(length, dtype=np.bool_)

    curr_bls = primary_bls

    for cand_idx in range(1, max_candidates + 1):
        if curr_bls.best_snr < snr_threshold:
            break

        period = curr_bls.best_period
        epoch = curr_bls.best_epoch_days
        duration_hours = curr_bls.best_duration_hours
        depth_ppm = curr_bls.best_depth_ppm
        snr = curr_bls.best_snr

        # Analyze event statistics via Astropy BoxLeastSquares
        bls_model = BoxLeastSquares(time_days, curr_flux, dy=flux_err)
        dur_days = duration_hours / 24.0
        stats = bls_model.compute_stats(period, dur_days, epoch)

        per_transit_counts = stats.get("per_transit_count", [])
        observed_event_count = int(sum(1 for cnt in per_transit_counts if cnt >= 3))

        flags: list[str] = []
        if observed_event_count < 2:
            flags.append("single_or_unconstrained_event")
        elif observed_event_count == 2:
            flags.append("weak_two_event")

        # Odd / even transit depth check
        depth_odd = stats.get("depth_odd", (None, None))[0]
        depth_even = stats.get("depth_even", (None, None))[0]
        if depth_odd is not None and depth_even is not None and depth_odd > 0 and depth_even > 0:
            diff_ratio = abs(depth_odd - depth_even) / max(depth_odd, depth_even)
            if diff_ratio > 0.4:
                flags.append("odd_even_mismatch")

        # Harmonics check
        aliases = (
            float(round(period / 2.0, 6)),
            float(round(period * 2.0, 6)),
        )

        cand = Candidate(
            candidate_id=f"{target_id}_{cand_idx:02d}",
            period_days=float(period),
            epoch_days=float(epoch),
            duration_hours=float(duration_hours),
            depth_ppm=float(depth_ppm),
            observed_event_count=observed_event_count,
            detection_snr=float(snr),
            search_significance=None,
            aliases_days=aliases,
            flags=tuple(flags),
        )
        candidates.append(cand)

        if cand_idx == max_candidates:
            break

        # Mask in-transit cadences for this candidate
        phase = ((time_days - epoch + 0.5 * period) % period) - 0.5 * period
        in_transit = np.abs(phase) <= (0.75 * dur_days)
        masked_mask |= in_transit

        # Replace in-transit cadences in curr_flux with local out-of-transit median (1.0)
        curr_flux[in_transit] = 1.0

        # Run residual BLS
        if np.count_nonzero(~masked_mask) >= 100:
            next_bls = run_bls(
                time_days,
                curr_flux,
                flux_err=flux_err,
                device=device,
                min_period=min_period,
                max_period=max_period,
                min_duration_hours=min_duration_hours,
                max_duration_hours=max_duration_hours,
                num_durations=num_durations,
                oversample=frequency_factor,
            )
            residual_bls.append(next_bls)
            curr_bls = next_bls
        else:
            break

    return DetectionResult(
        candidates=tuple(candidates),
        primary_bls=primary_bls,
        variability=variability,
        residual_bls=tuple(residual_bls),
    )
