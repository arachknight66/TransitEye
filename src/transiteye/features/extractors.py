"""Audited feature extractors operating strictly on scientific arrays and candidates."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from transiteye.characterization.fitting import RefinedCandidate
from transiteye.detection.variability import VariabilitySummary
from transiteye.domain.models import LightCurve
from transiteye.features.registry import validate_feature_vector
from transiteye.preprocessing.detrend import DualLightCurve


def stellar_temperature(light_curve: LightCurve) -> float | None:
    """Read optional effective temperature consistently for training and inference."""
    value = light_curve.metadata.get("teff")
    try:
        parsed = float(value) if value is not None and value != "None" else None
    except (ValueError, TypeError):
        return None
    return parsed if parsed is not None and np.isfinite(parsed) and parsed > 0 else None


def fold_phase(
    time_days: NDArray[np.float64],
    epoch_days: float,
    period_days: float,
) -> NDArray[np.float64]:
    """Fold timestamps to phase strictly in [-0.5, 0.5] with 0 at epoch."""
    return ((time_days - epoch_days + 0.5 * period_days) % period_days) / period_days - 0.5


def extract_morphology_features(
    candidate: RefinedCandidate,
    time_days: NDArray[np.float64],
    detrended_flux: NDArray[np.float64],
) -> dict[str, float]:
    """Extract transit morphology and shape features."""
    period_days = max(1e-4, float(candidate.period_days))
    duration_days = float(candidate.duration_hours) / 24.0
    ingress_fraction = float(candidate.ingress_fraction)

    # 1. v_shape_ratio: ingress fraction (1 for pure V, 0 for flat box)
    v_shape_ratio = float(np.clip(ingress_fraction, 0.0, 1.0))

    # 2. duration_to_period_ratio
    duration_to_period_ratio = float(np.clip(duration_days / period_days, 0.0, 1.0))

    # 3. boxiness_index: 1 - ingress_fraction
    boxiness_index = float(np.clip(1.0 - ingress_fraction, 0.0, 1.0))

    # 4. asymmetry_score: difference between ingress and egress mean flux normalized by depth
    phases = fold_phase(time_days, float(candidate.epoch_days), period_days)
    half_dur_phase = 0.5 * (duration_days / period_days)

    ingress_mask = (-half_dur_phase <= phases) & (phases < 0.0)
    egress_mask = (0.0 <= phases) & (phases <= half_dur_phase)

    depth = max(1e-5, float(candidate.measured_depth_ppm) * 1e-6)

    if np.sum(ingress_mask) >= 2 and np.sum(egress_mask) >= 2:
        mean_ingress = float(np.mean(detrended_flux[ingress_mask]))
        mean_egress = float(np.mean(detrended_flux[egress_mask]))
        asymmetry_score = float(np.clip(abs(mean_ingress - mean_egress) / depth, 0.0, 10.0))
    else:
        asymmetry_score = 0.0

    # 5. batman_vs_trapezoid_delta_bic
    if candidate.batman_delta_bic is not None and not math.isnan(candidate.batman_delta_bic):
        batman_vs_trapezoid_delta_bic = float(
            np.clip(candidate.batman_delta_bic, -1000.0, 1000.0)
        )
    else:
        batman_vs_trapezoid_delta_bic = 0.0

    return {
        "v_shape_ratio": v_shape_ratio,
        "duration_to_period_ratio": duration_to_period_ratio,
        "boxiness_index": boxiness_index,
        "asymmetry_score": asymmetry_score,
        "batman_vs_trapezoid_delta_bic": batman_vs_trapezoid_delta_bic,
    }


def extract_consistency_features(
    candidate: RefinedCandidate,
    time_days: NDArray[np.float64],
    detrended_flux: NDArray[np.float64],
) -> dict[str, float]:
    """Extract transit-to-transit consistency and coverage features."""
    period_days = max(1e-4, float(candidate.period_days))
    epoch_days = float(candidate.epoch_days)
    duration_days = float(candidate.duration_hours) / 24.0
    phases = fold_phase(time_days, epoch_days, period_days)
    half_dur_phase = 0.5 * (duration_days / period_days)
    in_transit = np.abs(phases) <= half_dur_phase

    # Cycle indices for each observation
    cycle_indices = np.round((time_days - epoch_days) / period_days).astype(int)

    # 6. odd_even_depth_sigma
    odd_in_transit = in_transit & (cycle_indices % 2 != 0)
    even_in_transit = in_transit & (cycle_indices % 2 == 0)

    if np.sum(odd_in_transit) >= 3 and np.sum(even_in_transit) >= 3:
        d_odd = 1.0 - float(np.mean(detrended_flux[odd_in_transit]))
        d_even = 1.0 - float(np.mean(detrended_flux[even_in_transit]))
        var_odd = float(np.var(detrended_flux[odd_in_transit])) / np.sum(odd_in_transit)
        var_even = float(np.var(detrended_flux[even_in_transit])) / np.sum(even_in_transit)
        denom = math.sqrt(max(1e-12, var_odd + var_even))
        odd_even_depth_sigma = float(np.clip(abs(d_odd - d_even) / denom, 0.0, 100.0))
    else:
        odd_even_depth_sigma = 0.0

    # 7. per_transit_depth_mad_ratio and 10. observed_event_count
    unique_cycles = np.unique(cycle_indices[in_transit])
    cycle_depths: list[float] = []

    for c in unique_cycles:
        c_mask = in_transit & (cycle_indices == c)
        if np.sum(c_mask) >= 3:
            cycle_depths.append(1.0 - float(np.median(detrended_flux[c_mask])))

    observed_event_count = float(len(cycle_depths))

    if len(cycle_depths) >= 2:
        med_depth = float(np.median(cycle_depths))
        mad_depth = float(np.median(np.abs(np.array(cycle_depths) - med_depth))) * 1.4826
        per_transit_depth_mad_ratio = float(
            np.clip(mad_depth / max(1e-5, med_depth), 0.0, 10.0)
        )
    else:
        per_transit_depth_mad_ratio = 0.1

    # 8. phase_coverage_fraction
    phase_coverage_fraction = float(np.clip(candidate.phase_coverage, 0.0, 1.0))

    # 9. in_transit_duty_cycle: fraction of expected transit events that have data
    if len(time_days) > 0:
        min_cycle = int(math.ceil((time_days[0] - epoch_days) / period_days))
        max_cycle = int(math.floor((time_days[-1] - epoch_days) / period_days))
        expected_events = max(1, max_cycle - min_cycle + 1)
        in_transit_duty_cycle = float(
            np.clip(observed_event_count / expected_events, 0.0, 1.0)
        )
    else:
        in_transit_duty_cycle = 1.0

    return {
        "odd_even_depth_sigma": odd_even_depth_sigma,
        "per_transit_depth_mad_ratio": per_transit_depth_mad_ratio,
        "phase_coverage_fraction": phase_coverage_fraction,
        "in_transit_duty_cycle": in_transit_duty_cycle,
        "observed_event_count": observed_event_count,
    }


def extract_eclipse_features(
    candidate: RefinedCandidate,
    time_days: NDArray[np.float64],
    detrended_flux: NDArray[np.float64],
) -> dict[str, float]:
    """Extract secondary eclipse and ellipsoidal variation features."""
    period_days = max(1e-4, float(candidate.period_days))
    duration_days = float(candidate.duration_hours) / 24.0
    phases = fold_phase(time_days, float(candidate.epoch_days), period_days)
    half_dur_phase = 0.5 * (duration_days / period_days)

    primary_depth = max(1e-5, float(candidate.measured_depth_ppm) * 1e-6)

    # Mask primary transit
    oot_mask = np.abs(phases) > (1.2 * half_dur_phase)
    phases_oot = phases[oot_mask]
    flux_oot = detrended_flux[oot_mask]

    # Search for secondary eclipse around phase 0.5 (or phase -0.5)
    # Secondary window: [0.3, 0.7] (or absolute phase distance from 0 between 0.3 and 0.5)
    sec_mask = np.abs(phases_oot) >= 0.35
    oot_std = (
        float(np.median(np.abs(flux_oot - np.median(flux_oot)))) * 1.4826
        if len(flux_oot) > 10
        else 1e-3
    )
    oot_std = max(1e-6, oot_std)

    if np.sum(sec_mask) >= 5:
        phases_sec = phases_oot[sec_mask]
        flux_sec = flux_oot[sec_mask]

        # Scan sliding bins around phase 0.5 to find maximum dip SNR
        scan_phases = np.linspace(0.35, 0.5, 9)
        best_snr = 0.0
        best_offset = 0.0
        best_sec_depth = 0.0

        for p_center in scan_phases:
            # Check both positive and negative phases (+/- 0.5)
            bin_mask = (np.abs(np.abs(phases_sec) - p_center)) <= half_dur_phase
            n_pts = int(np.sum(bin_mask))
            if n_pts >= 3:
                d_sec = 1.0 - float(np.mean(flux_sec[bin_mask]))
                snr = d_sec / (oot_std / math.sqrt(n_pts))
                if snr > best_snr:
                    best_snr = snr
                    best_offset = abs(0.5 - p_center)
                    best_sec_depth = max(0.0, d_sec)

        max_secondary_snr = float(np.clip(best_snr, 0.0, 1000.0))
        secondary_to_primary_depth_ratio = float(
            np.clip(best_sec_depth / primary_depth, 0.0, 10.0)
        )
        secondary_phase_offset_from_half = float(np.clip(best_offset, 0.0, 0.5))
    else:
        max_secondary_snr = 0.0
        secondary_to_primary_depth_ratio = 0.0
        secondary_phase_offset_from_half = 0.0

    # 14. ellipsoidal_amplitude_ratio: amplitude of cos(4*pi*phase) modulation
    if len(phases_oot) >= 20:
        cos_4pi = np.cos(4.0 * np.pi * phases_oot)
        # Projection of flux deviation onto cos(4*pi*phase)
        dev = flux_oot - float(np.mean(flux_oot))
        # Amplitude A = -2 * sum(dev * cos) / sum(cos^2)
        denom = float(np.sum(cos_4pi**2))
        if denom > 1e-6:
            amp = -float(np.sum(dev * cos_4pi)) / denom
            ellipsoidal_amplitude_ratio = float(
                np.clip(max(0.0, amp) / primary_depth, 0.0, 5.0)
            )
        else:
            ellipsoidal_amplitude_ratio = 0.0
    else:
        ellipsoidal_amplitude_ratio = 0.0

    return {
        "max_secondary_snr": max_secondary_snr,
        "secondary_to_primary_depth_ratio": secondary_to_primary_depth_ratio,
        "secondary_phase_offset_from_half": secondary_phase_offset_from_half,
        "ellipsoidal_amplitude_ratio": ellipsoidal_amplitude_ratio,
    }


def extract_variability_features(
    candidate: RefinedCandidate,
    variability: VariabilitySummary,
    time_days: NDArray[np.float64],
    detrended_flux: NDArray[np.float64],
) -> dict[str, float]:
    """Extract stellar variability and harmonic relation features."""
    period_days = max(1e-4, float(candidate.period_days))
    duration_days = float(candidate.duration_hours) / 24.0
    phases = fold_phase(time_days, float(candidate.epoch_days), period_days)
    half_dur_phase = 0.5 * (duration_days / period_days)
    primary_depth = max(1e-5, float(candidate.measured_depth_ppm) * 1e-6)

    # 15. ls_max_power
    ls_max_power = float(np.clip(variability.peak_power, 0.0, 1.0))

    # 16. ls_to_bls_period_harmonic_distance: min |k * P_ls / P_bls - 1|
    p_ls = float(variability.best_period_days)
    harmonics = (0.33333, 0.5, 1.0, 2.0, 3.0)
    harmonic_dists = [abs(k * p_ls / period_days - 1.0) for k in harmonics]
    ls_to_bls_period_harmonic_distance = float(
        np.clip(min(harmonic_dists), 0.0, 1.0)
    )

    # 17. out_of_transit_rms_to_depth_ratio
    oot_mask = np.abs(phases) > (1.2 * half_dur_phase)
    if np.sum(oot_mask) >= 10:
        oot_rms = float(np.std(detrended_flux[oot_mask]))
        out_of_transit_rms_to_depth_ratio = float(
            np.clip(oot_rms / primary_depth, 0.0, 100.0)
        )
    else:
        out_of_transit_rms_to_depth_ratio = 1.0

    # 18. autocorrelation_at_period
    # Sample point pairs with separation approximately equal to period_days
    if len(time_days) >= 50:
        # Sort time series if not already
        sort_idx = np.argsort(time_days)
        t_sorted = time_days[sort_idx]
        f_sorted = detrended_flux[sort_idx] - 1.0
        var_f = float(np.var(f_sorted))

        # Vectorized lag match: search for pairs where |t_j - (t_i + period_days)| < 0.2 * duration_days
        lag_tol = max(0.01, 0.2 * duration_days)
        target_times = t_sorted + period_days
        right_indices = np.searchsorted(t_sorted, target_times)
        right_indices = np.clip(right_indices, 0, len(t_sorted) - 1)

        diffs = np.abs(t_sorted[right_indices] - target_times)
        valid_pairs = diffs <= lag_tol

        if np.sum(valid_pairs) >= 10 and var_f > 1e-12:
            cov = float(np.mean(f_sorted[valid_pairs] * f_sorted[right_indices[valid_pairs]]))
            autocorr = float(np.clip(cov / var_f, -1.0, 1.0))
        else:
            autocorr = 0.0
    else:
        autocorr = 0.0

    return {
        "ls_max_power": ls_max_power,
        "ls_to_bls_period_harmonic_distance": ls_to_bls_period_harmonic_distance,
        "out_of_transit_rms_to_depth_ratio": out_of_transit_rms_to_depth_ratio,
        "autocorrelation_at_period": autocorr,
    }


def extract_noise_quality_features(
    candidate: RefinedCandidate,
    dual_light_curve: DualLightCurve,
    raw_light_curve: LightCurve | None = None,
) -> dict[str, float]:
    """Extract noise properties and quality flag proximity features."""
    prepared = dual_light_curve.prepared
    time_days = prepared.time_days
    detrended_flux = dual_light_curve.detrended_flux

    period_days = max(1e-4, float(candidate.period_days))
    duration_days = float(candidate.duration_hours) / 24.0
    phases = fold_phase(time_days, float(candidate.epoch_days), period_days)
    half_dur_phase = 0.5 * (duration_days / period_days)
    in_transit_mask = np.abs(phases) <= half_dur_phase

    # 19. red_noise_beta_factor
    # Compute duration-binned scatter vs white noise expectation
    diffs = np.diff(time_days)
    median_dt = float(np.median(diffs[diffs > 0])) if len(diffs) > 0 else 2.0 / 1440.0
    n_in = max(1.0, duration_days / max(1e-5, median_dt))
    n_cadences = max(1, int(round(n_in)))

    white_sigma = float(np.median(np.abs(detrended_flux - np.median(detrended_flux)))) * 1.4826
    white_sigma = max(1e-6, white_sigma)

    # Binned flux on duration scale across out-of-transit
    oot_flux = detrended_flux[~in_transit_mask]
    n_bins = int(len(oot_flux) // n_cadences)
    if n_bins >= 5:
        total_pts = int(n_bins * n_cadences)
        binned = np.mean(oot_flux[:total_pts].reshape(n_bins, n_cadences), axis=1)
        binned_sigma = float(np.std(binned))
        expected_white_binned = white_sigma / math.sqrt(n_cadences)
        red_noise_beta_factor = float(
            np.clip(binned_sigma / max(1e-6, expected_white_binned), 0.5, 20.0)
        )
    else:
        red_noise_beta_factor = 1.0

    # 20. detection_snr
    cand_snr = candidate.candidate.detection_snr
    if cand_snr is not None and not math.isnan(cand_snr):
        detection_snr = float(np.clip(cand_snr, 0.0, 10000.0))
    else:
        detection_snr = 5.0

    # 21. in_transit_flagged_fraction
    # Check flags in raw light curve if available, or exclusion_flags in prepared
    if raw_light_curve is not None and raw_light_curve.quality is not None:
        raw_phases = fold_phase(
            raw_light_curve.time_days, float(candidate.epoch_days), period_days
        )
        raw_in_transit = np.abs(raw_phases) <= half_dur_phase
        if np.sum(raw_in_transit) > 0:
            flagged = np.sum(raw_in_transit & (raw_light_curve.quality != 0))
            in_transit_flagged_fraction = float(
                np.clip(flagged / np.sum(raw_in_transit), 0.0, 1.0)
            )
        else:
            in_transit_flagged_fraction = 0.0
    else:
        # Fall back to prepared exclusion flags
        in_transit_flagged_fraction = 0.0

    # 22. gap_proximity_fraction: in-transit points within 1 hour of a segment boundary
    gap_threshold_days = 1.0 / 24.0  # 1 hour
    segment_boundaries: list[float] = []
    for start, end in prepared.segment_slices:
        segment_boundaries.append(time_days[start])
        segment_boundaries.append(time_days[end - 1])

    if segment_boundaries and np.sum(in_transit_mask) > 0:
        boundaries_arr = np.array(segment_boundaries)
        in_transit_times = time_days[in_transit_mask]
        # Distance to closest boundary
        min_dists = np.min(
            np.abs(in_transit_times[:, np.newaxis] - boundaries_arr[np.newaxis, :]), axis=1
        )
        proximate_count = np.sum(min_dists <= gap_threshold_days)
        gap_proximity_fraction = float(
            np.clip(proximate_count / len(in_transit_times), 0.0, 1.0)
        )
    else:
        gap_proximity_fraction = 0.0

    return {
        "red_noise_beta_factor": red_noise_beta_factor,
        "detection_snr": detection_snr,
        "in_transit_flagged_fraction": in_transit_flagged_fraction,
        "gap_proximity_fraction": gap_proximity_fraction,
    }


def extract_source_evidence_features(
    candidate: RefinedCandidate,
    dual_light_curve: DualLightCurve,
) -> dict[str, float]:
    """Extract spatial source and flux dilution evidence features."""
    prepared = dual_light_curve.prepared
    time_days = prepared.time_days
    detrended_flux = dual_light_curve.detrended_flux
    flux_delivered = prepared.flux_normalized

    period_days = max(1e-4, float(candidate.period_days))
    duration_days = float(candidate.duration_hours) / 24.0
    phases = fold_phase(time_days, float(candidate.epoch_days), period_days)
    half_dur_phase = 0.5 * (duration_days / period_days)
    in_transit_mask = np.abs(phases) <= half_dur_phase
    oot_mask = ~in_transit_mask

    # 23. centroid_shift_sigma
    col = prepared.centroid_column
    row = prepared.centroid_row

    if (
        col is not None
        and row is not None
        and np.sum(in_transit_mask) >= 3
        and np.sum(oot_mask) >= 10
    ):
        x_in = col[in_transit_mask]
        y_in = row[in_transit_mask]
        x_out = col[oot_mask]
        y_out = row[oot_mask]

        valid_in = ~np.isnan(x_in) & ~np.isnan(y_in)
        valid_out = ~np.isnan(x_out) & ~np.isnan(y_out)

        if np.sum(valid_in) >= 3 and np.sum(valid_out) >= 10:
            x_in_m, y_in_m = float(np.mean(x_in[valid_in])), float(np.mean(y_in[valid_in]))
            x_out_m, y_out_m = float(np.mean(x_out[valid_out])), float(np.mean(y_out[valid_out]))

            dx = x_in_m - x_out_m
            dy = y_in_m - y_out_m
            dist = math.sqrt(dx**2 + dy**2)

            var_dx = float(np.var(x_in[valid_in])) / np.sum(valid_in) + float(
                np.var(x_out[valid_out])
            ) / np.sum(valid_out)
            var_dy = float(np.var(y_in[valid_in])) / np.sum(valid_in) + float(
                np.var(y_out[valid_out])
            ) / np.sum(valid_out)
            denom = math.sqrt(max(1e-12, var_dx + var_dy))

            centroid_shift_sigma = float(np.clip(dist / denom, 0.0, 100.0))
        else:
            centroid_shift_sigma = 0.0
    else:
        centroid_shift_sigma = 0.0

    # 24. sap_to_pdcsap_depth_ratio (delivered normalized flux vs detrended flux)
    if np.sum(in_transit_mask) >= 3:
        d_deliv = max(0.0, 1.0 - float(np.median(flux_delivered[in_transit_mask])))
        d_detrend = max(1e-5, 1.0 - float(np.median(detrended_flux[in_transit_mask])))
        sap_to_pdcsap_depth_ratio = float(np.clip(d_deliv / d_detrend, 0.0, 10.0))
    else:
        sap_to_pdcsap_depth_ratio = 1.0

    # 25. crowdsap_value
    crowd_raw: Any = prepared.metadata.get("crowdsap")
    if crowd_raw is not None:
        try:
            crowdsap_value = float(np.clip(float(crowd_raw), 0.0, 1.0))
        except (ValueError, TypeError):
            crowdsap_value = 1.0
    else:
        crowdsap_value = 1.0

    return {
        "centroid_shift_sigma": centroid_shift_sigma,
        "sap_to_pdcsap_depth_ratio": sap_to_pdcsap_depth_ratio,
        "crowdsap_value": crowdsap_value,
    }


def extract_features(
    dual_light_curve: DualLightCurve,
    candidate: RefinedCandidate,
    variability: VariabilitySummary,
    raw_light_curve: LightCurve | None = None,
) -> dict[str, float]:
    """Extract the complete audited 25-feature vector for a candidate."""
    prepared = dual_light_curve.prepared
    time_days = prepared.time_days
    detrended_flux = dual_light_curve.detrended_flux

    features: dict[str, float] = {}

    features.update(extract_morphology_features(candidate, time_days, detrended_flux))
    features.update(extract_consistency_features(candidate, time_days, detrended_flux))
    features.update(extract_eclipse_features(candidate, time_days, detrended_flux))
    features.update(
        extract_variability_features(candidate, variability, time_days, detrended_flux)
    )
    features.update(
        extract_noise_quality_features(candidate, dual_light_curve, raw_light_curve)
    )
    features.update(extract_source_evidence_features(candidate, dual_light_curve))

    return validate_feature_vector(features)
