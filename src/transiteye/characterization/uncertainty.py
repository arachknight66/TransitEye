"""Red-noise corrected SNR, block-bootstrap parameter intervals, and empirical significance."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from transiteye.characterization.fitting import RefinedCandidate, fit_trapezoid
from transiteye.domain.models import Interval, MissingReason


def compute_red_noise_snr(
    flux_detrended: NDArray[np.float64],
    model_curve: NDArray[np.float64],
    time_days: NDArray[np.float64],
    depth_ppm: float,
    duration_hours: float,
    observed_event_count: int,
) -> float:
    """Compute Pont et al. (2006) red-noise corrected transit depth SNR.

    SNR_red = depth / sqrt(sigma_w^2 / N_in + sigma_r^2 / N_transits)
    """
    if depth_ppm <= 0 or duration_hours <= 0:
        return 0.0

    residuals = flux_detrended - model_curve
    n_total = len(residuals)
    if n_total < 10:
        return 0.0

    # Robust white noise estimate
    mad_w = float(np.median(np.abs(residuals - np.median(residuals))))
    sigma_w = 1.4826 * mad_w

    # Duration in days and average cadence
    dur_days = duration_hours / 24.0
    cadence_days = float(np.median(np.diff(time_days))) if n_total > 1 else dur_days
    if cadence_days <= 0:
        cadence_days = dur_days
    n_pts_in_transit = max(1, int(round(dur_days / cadence_days)))

    # Compute variance of binned residuals
    bin_size = max(2, n_pts_in_transit)
    n_bins = n_total // bin_size
    if n_bins >= 3:
        binned = [
            float(np.mean(residuals[b * bin_size : (b + 1) * bin_size])) for b in range(n_bins)
        ]
        binned_arr = np.array(binned)
        mad_b = float(np.median(np.abs(binned_arr - np.median(binned_arr))))
        sigma_b = 1.4826 * mad_b
        # Red noise: sigma_r^2 = max(0, sigma_b^2 - sigma_w^2 / bin_size)
        sigma_r2 = max(0.0, (sigma_b**2) - (sigma_w**2) / bin_size)
    else:
        sigma_r2 = 0.0

    total_in_transit_pts = n_pts_in_transit * max(1, observed_event_count)
    depth_fraction = depth_ppm / 1e6

    variance_depth = (sigma_w**2) / max(1, total_in_transit_pts) + sigma_r2 / max(
        1, observed_event_count
    )
    if variance_depth <= 0.0:
        return 0.0

    return float(depth_fraction / np.sqrt(variance_depth))


def compute_parameter_intervals(
    time_days: NDArray[np.float64],
    flux_detrended: NDArray[np.float64],
    flux_err: NDArray[np.float64] | None,
    refined: RefinedCandidate,
    n_boot: int = 40,
    random_seed: int = 42,
) -> dict[str, dict[str, Interval]]:
    """Compute 68% and 95% intervals for candidate parameters via residual block-bootstrap."""
    residuals = flux_detrended - refined.trapezoid_model_curve
    n_pts = len(residuals)

    dur_days = refined.duration_hours / 24.0
    cadence_days = float(np.median(np.diff(time_days))) if n_pts > 1 else 0.002
    block_size = max(5, int(round(max(0.2, dur_days * 2.0) / max(1e-4, cadence_days))))

    rng = np.random.default_rng(random_seed)
    n_blocks = (n_pts + block_size - 1) // block_size

    # Storage for bootstrap parameters
    boot_periods: list[float] = []
    boot_epochs: list[float] = []
    boot_durations: list[float] = []
    boot_depths: list[float] = []

    for _ in range(n_boot):
        # Sample blocks with replacement
        start_indices = rng.integers(0, max(1, n_pts - block_size + 1), size=n_blocks)
        boot_res_blocks = [residuals[s : s + block_size] for s in start_indices]
        boot_res = np.concatenate(boot_res_blocks)[:n_pts]
        boot_flux = refined.trapezoid_model_curve + boot_res

        try:
            params, _, _, _ = fit_trapezoid(
                time_days,
                boot_flux,
                flux_err=flux_err,
                init_epoch=refined.epoch_days,
                init_period=refined.period_days,
                init_duration_hours=refined.duration_hours,
                init_depth_ppm=refined.measured_depth_ppm,
            )
            boot_periods.append(params["period_days"])
            boot_epochs.append(params["epoch_days"])
            boot_durations.append(params["duration_hours"])
            boot_depths.append(params["depth_ppm"])
        except Exception:
            continue

    def _make_interval(values: list[float], cov: float, default_val: float) -> Interval:
        if len(values) < 10:
            return Interval(
                lower=None,
                upper=None,
                coverage=cov,
                missing_reason=MissingReason.FIT_FAILED,
            )
        arr = np.sort(values)
        alpha = 1.0 - cov
        low = float(np.percentile(arr, 100.0 * (alpha / 2.0)))
        high = float(np.percentile(arr, 100.0 * (1.0 - alpha / 2.0)))
        return Interval(lower=low, upper=high, coverage=cov)

    param_map = {
        "period_days": (boot_periods, refined.period_days),
        "epoch_days": (boot_epochs, refined.epoch_days),
        "duration_hours": (boot_durations, refined.duration_hours),
        "depth_ppm": (boot_depths, refined.measured_depth_ppm),
    }

    results: dict[str, dict[str, Interval]] = {}
    for name, (vals, def_val) in param_map.items():
        results[name] = {
            "68": _make_interval(vals, 0.68, def_val),
            "95": _make_interval(vals, 0.95, def_val),
        }
    return results


def compute_empirical_significance(
    time_days: NDArray[np.float64],
    flux_detrended: NDArray[np.float64],
    model_curve: NDArray[np.float64],
    candidate_period: float,
    candidate_duration_hours: float,
    observed_snr: float,
    n_surrogates: int = 200,
    random_seed: int = 42,
    device: str = "auto",
) -> tuple[float, float]:
    """Compute noise-surrogate empirical search significance and resolution.

    Returns (empirical_significance, surrogate_resolution).
    Resolution = 1.0 / (n_surrogates + 1).
    """
    from transiteye.execution.compute import has_cuda, resolve_device

    resolution = 1.0 / (n_surrogates + 1)
    if observed_snr <= 0:
        return 0.0, resolution

    residuals = flux_detrended - model_curve
    n_pts = len(residuals)

    dur_days = candidate_duration_hours / 24.0
    cadence_days = float(np.median(np.diff(time_days))) if n_pts > 1 else 0.002
    block_size = max(5, int(round(dur_days / max(1e-4, cadence_days))))

    rng = np.random.default_rng(random_seed)
    n_blocks = (n_pts + block_size - 1) // block_size

    # Test frequency around candidate period (+- 5%)
    phase = ((time_days + 0.5 * candidate_period) % candidate_period) - 0.5 * candidate_period
    in_box = np.abs(phase) <= (0.5 * dur_days)
    n_in = int(np.count_nonzero(in_box))

    if n_in < 2:
        return 0.5, resolution

    observed_stat = observed_snr
    mad = float(np.median(np.abs(residuals - np.median(residuals))))
    std_est = 1.4826 * mad
    if std_est <= 0:
        return 1.0, resolution

    # Pre-generate deterministic block start indices on CPU so GPU and CPU match 100%
    all_start_indices = rng.integers(
        0, max(1, n_pts - block_size + 1), size=(n_surrogates, n_blocks)
    )
    offsets = np.arange(block_size, dtype=np.int64)
    gather_idx = (all_start_indices[:, :, None] + offsets[None, None, :]).reshape(
        n_surrogates, -1
    )[:, :n_pts]

    dev = resolve_device(device)
    if dev == "cuda" and has_cuda():
        import cupy as cp

        res_gpu = cp.asarray(residuals)
        idx_gpu = cp.asarray(gather_idx)
        in_box_gpu = cp.asarray(in_box)

        surr_matrix = res_gpu[idx_gpu]
        in_flux = 1.0 + surr_matrix[:, in_box_gpu]
        dip_vals = 1.0 - cp.mean(in_flux, axis=1)
        surr_snrs = dip_vals / (std_est / np.sqrt(n_in))
        exceedances = int(cp.count_nonzero(surr_snrs >= observed_stat).item())
    else:
        exceedances = 0
        for s_idx in range(n_surrogates):
            res_surr = residuals[gather_idx[s_idx]]
            in_flux = 1.0 + res_surr[in_box]
            dip_val = 1.0 - float(np.mean(in_flux))
            surr_snr = dip_val / (std_est / np.sqrt(n_in))
            if surr_snr >= observed_stat:
                exceedances += 1

    p_emp = (1 + exceedances) * resolution
    empirical_significance = float(max(0.0, 1.0 - p_emp))
    return empirical_significance, resolution
