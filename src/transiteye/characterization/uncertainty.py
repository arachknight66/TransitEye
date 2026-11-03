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
                exposure_days=refined.exposure_days,
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
    """Reject the invalid historical fixed-phase statistic.

    A valid search-wide estimate requires applying the entire candidate search to
    every noise surrogate under a separately validated null model.
    """
    raise RuntimeError(
        "Search-wide empirical significance is unavailable: the historical "
        "fixed-phase surrogate did not repeat the candidate search."
    )
