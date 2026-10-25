"""Exposure-integrated trapezoid fitting and optional batman limb-darkened transit refinement."""

from __future__ import annotations

from dataclasses import dataclass

import batman
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from transiteye.domain.models import Candidate


@dataclass(frozen=True, slots=True)
class RefinedCandidate:
    """Refined parameter estimates and model curves for a detected candidate."""

    candidate: Candidate
    period_days: float
    epoch_days: float
    duration_hours: float
    measured_depth_ppm: float
    dilution_corrected_depth_ppm: float | None
    ingress_fraction: float
    chi2_reduced: float
    bic: float
    batman_delta_bic: float | None
    phase_coverage: float
    trapezoid_model_curve: NDArray[np.float64]
    batman_model_curve: NDArray[np.float64] | None


def evaluate_trapezoid(
    time_days: NDArray[np.float64],
    epoch_days: float,
    period_days: float,
    duration_days: float,
    depth: float,
    ingress_fraction: float,
    baseline: float = 1.0,
    supersample: int = 5,
    exposure_days: float = 2.0 / 1440.0,
) -> NDArray[np.float64]:
    """Evaluate a symmetrical exposure-integrated trapezoid model.

    duration_days: Total transit duration (T1 to T4).
    ingress_fraction: Fraction of half-duration spent in ingress/egress (in [0, 1]).
    """
    if supersample <= 1:
        offsets = np.array([0.0])
    else:
        offsets = np.linspace(-0.5 * exposure_days, 0.5 * exposure_days, supersample)

    # Broadcast evaluation over sub-exposure offsets
    # time shape: (N,), offsets shape: (S,) -> (N, S)
    t_mesh = time_days[:, np.newaxis] + offsets[np.newaxis, :]
    half_dur = 0.5 * duration_days
    t_ing = ingress_fraction * half_dur
    t_flat = max(0.0, half_dur - t_ing)

    phase = ((t_mesh - epoch_days + 0.5 * period_days) % period_days) - 0.5 * period_days
    abs_phase = np.abs(phase)

    # Default out of transit is baseline
    flux_mesh = np.full_like(t_mesh, baseline)

    # In flat bottom
    flat_mask = abs_phase <= t_flat
    flux_mesh[flat_mask] = baseline - depth

    # In ingress / egress ramp
    ramp_mask = (abs_phase > t_flat) & (abs_phase < half_dur)
    if t_ing > 0:
        ramp_factor = (half_dur - abs_phase[ramp_mask]) / t_ing
        flux_mesh[ramp_mask] = baseline - depth * ramp_factor

    # Average over sub-exposure offsets
    return np.mean(flux_mesh, axis=1)


def fit_trapezoid(
    time_days: NDArray[np.float64],
    flux: NDArray[np.float64],
    flux_err: NDArray[np.float64] | None,
    init_epoch: float,
    init_period: float,
    init_duration_hours: float,
    init_depth_ppm: float,
) -> tuple[dict[str, float], NDArray[np.float64], float, float]:
    """Fit a symmetrical trapezoid model using bounded non-linear least squares."""
    weights = 1.0 / flux_err if flux_err is not None else np.ones_like(flux)
    weights = np.where(weights <= 0, 1.0, weights)

    init_dur_days = init_duration_hours / 24.0
    init_depth = max(1e-5, init_depth_ppm / 1e6)

    # Window optimization: fit on near-transit cadences for 50x speedup
    phase_init = ((time_days - init_epoch + 0.5 * init_period) % init_period) - 0.5 * init_period
    near_mask = np.abs(phase_init) <= max(0.1, 3.5 * init_dur_days)
    if np.sum(near_mask) < 15:
        near_mask = np.ones_like(time_days, dtype=bool)

    t_fit = time_days[near_mask]
    f_fit = flux[near_mask]
    w_fit = weights[near_mask]

    # Initial vector: [epoch, period, duration_days, depth, ingress_frac, baseline]
    p0 = [init_epoch, init_period, init_dur_days, init_depth, 0.2, 1.0]

    bounds_lower = [
        init_epoch - 0.2 * init_period,
        0.8 * init_period,
        max(1e-3, 0.2 * init_dur_days),
        0.0,
        0.01,
        0.95,
    ]
    bounds_upper = [
        init_epoch + 0.2 * init_period,
        1.2 * init_period,
        max(1e-2, 3.0 * init_dur_days),
        0.5,
        0.99,
        1.05,
    ]

    def residual(p: list[float]) -> np.ndarray:
        model = evaluate_trapezoid(
            t_fit,
            epoch_days=p[0],
            period_days=p[1],
            duration_days=p[2],
            depth=p[3],
            ingress_fraction=p[4],
            baseline=p[5],
        )
        return (f_fit - model) * w_fit

    res = least_squares(
        residual,
        p0,
        bounds=(bounds_lower, bounds_upper),
        loss="soft_l1",
        f_scale=0.1,
        max_nfev=300,
    )
    p_opt = res.x

    model_curve = evaluate_trapezoid(
        time_days,
        epoch_days=p_opt[0],
        period_days=p_opt[1],
        duration_days=p_opt[2],
        depth=p_opt[3],
        ingress_fraction=p_opt[4],
        baseline=p_opt[5],
    )

    n_data = len(flux)
    n_params = len(p_opt)
    chi2 = float(np.sum(((flux - model_curve) * weights) ** 2))
    chi2_nu = chi2 / max(1, n_data - n_params)
    bic = float(chi2 + n_params * np.log(max(1, n_data)))

    params = {
        "epoch_days": float(p_opt[0]),
        "period_days": float(p_opt[1]),
        "duration_hours": float(p_opt[2] * 24.0),
        "depth_ppm": float(p_opt[3] * 1e6),
        "ingress_fraction": float(p_opt[4]),
        "baseline": float(p_opt[5]),
    }
    return params, model_curve, chi2_nu, bic


def fit_batman_refinement(
    time_days: NDArray[np.float64],
    flux: NDArray[np.float64],
    flux_err: NDArray[np.float64] | None,
    init_epoch: float,
    init_period: float,
    init_duration_days: float,
    init_depth: float,
) -> tuple[dict[str, float], NDArray[np.float64], float] | None:
    """Refine candidate parameters using batman quadratic limb-darkened transit model."""
    weights = 1.0 / flux_err if flux_err is not None else np.ones_like(flux)
    weights = np.where(weights <= 0, 1.0, weights)

    # Initial Rp/Rstar approximation
    init_rp = np.sqrt(max(1e-6, init_depth))

    # Rough estimate of semi-major axis (a/Rstar) from period and duration:
    # duration = P/pi * arcsin(Rstar/a * sqrt((1+rp)^2 - b^2))
    # For b=0, a/Rstar ~ P / (pi * duration)
    init_a = max(2.0, init_period / (np.pi * max(1e-3, init_duration_days)))

    # Parameter vector to optimize: [t0, period, rp, a, inc]
    p0 = [init_epoch, init_period, init_rp, init_a, 88.0]
    bounds_lower = [init_epoch - 0.1 * init_period, 0.9 * init_period, 0.001, 1.5, 60.0]
    bounds_upper = [init_epoch + 0.1 * init_period, 1.1 * init_period, 0.5, 200.0, 90.0]

    def residual(p: list[float]) -> np.ndarray:
        params = batman.TransitParams()
        params.t0 = p[0]
        params.per = p[1]
        params.rp = p[2]
        params.a = p[3]
        params.inc = p[4]
        params.ecc = 0.0
        params.w = 90.0
        params.u = [0.3, 0.2]
        params.limb_dark = "quadratic"

        model = batman.TransitModel(params, time_days)
        m_flux = model.light_curve(params)
        return (flux - m_flux) * weights

    try:
        res = least_squares(
            residual,
            p0,
            bounds=(bounds_lower, bounds_upper),
            loss="soft_l1",
            max_nfev=200,
        )
        p_opt = res.x

        params = batman.TransitParams()
        params.t0 = p_opt[0]
        params.per = p_opt[1]
        params.rp = p_opt[2]
        params.a = p_opt[3]
        params.inc = p_opt[4]
        params.ecc = 0.0
        params.w = 90.0
        params.u = [0.3, 0.2]
        params.limb_dark = "quadratic"
        model_obj = batman.TransitModel(params, time_days)
        model_curve = model_obj.light_curve(params)

        n_data = len(flux)
        n_params = len(p_opt)
        chi2 = float(np.sum(((flux - model_curve) * weights) ** 2))
        bic = float(chi2 + n_params * np.log(max(1, n_data)))

        res_dict = {
            "epoch_days": float(p_opt[0]),
            "period_days": float(p_opt[1]),
            "rp_rstar": float(p_opt[2]),
            "a_rstar": float(p_opt[3]),
            "inc_deg": float(p_opt[4]),
        }
        return res_dict, model_curve, bic
    except Exception:
        return None


def compute_phase_coverage(
    time_days: NDArray[np.float64],
    epoch_days: float,
    period_days: float,
    duration_days: float,
    n_bins: int = 20,
) -> float:
    """Compute fraction of in-transit phase bins that contain at least one observation."""
    half_dur = 0.5 * duration_days
    phase = ((time_days - epoch_days + 0.5 * period_days) % period_days) - 0.5 * period_days
    in_transit_phases = phase[np.abs(phase) <= half_dur] / half_dur  # in [-1, 1]

    if len(in_transit_phases) == 0:
        return 0.0

    bins = np.linspace(-1.0, 1.0, n_bins + 1)
    counts, _ = np.histogram(in_transit_phases, bins=bins)
    occupied = np.count_nonzero(counts > 0)
    return float(occupied / n_bins)


def refine_candidate(
    candidate: Candidate,
    time_days: NDArray[np.float64],
    flux_detrended: NDArray[np.float64],
    flux_err: NDArray[np.float64] | None,
    crowdsap: float | None = None,
    batman_snr_threshold: float = 7.0,
) -> RefinedCandidate:
    """Refine detected candidate with trapezoid fitting and optional batman refinement."""
    trap_params, trap_curve, chi2_nu, trap_bic = fit_trapezoid(
        time_days,
        flux_detrended,
        flux_err,
        init_epoch=candidate.epoch_days or 0.0,
        init_period=candidate.period_days or 1.0,
        init_duration_hours=candidate.duration_hours or 2.0,
        init_depth_ppm=candidate.depth_ppm or 1000.0,
    )

    # Optional batman refinement for transit-like candidates with SNR >= batman_snr_threshold
    batman_res = None
    batman_delta_bic = None
    batman_curve = None
    if (candidate.detection_snr or 0.0) >= batman_snr_threshold and trap_params["ingress_fraction"] < 0.8:
        batman_res = fit_batman_refinement(
            time_days,
            flux_detrended,
            flux_err,
            init_epoch=trap_params["epoch_days"],
            init_period=trap_params["period_days"],
            init_duration_days=trap_params["duration_hours"] / 24.0,
            init_depth=trap_params["depth_ppm"] / 1e6,
        )
        if batman_res is not None:
            _, batman_curve, batman_bic = batman_res
            batman_delta_bic = float(batman_bic - trap_bic)

    # Dilution correction from CROWDSAP
    measured_depth = trap_params["depth_ppm"]
    dilution_corrected_depth = None
    if crowdsap is not None and 0.0 < crowdsap <= 1.0:
        dilution_corrected_depth = float(measured_depth / crowdsap)

    # Phase coverage in transit
    coverage = compute_phase_coverage(
        time_days,
        epoch_days=trap_params["epoch_days"],
        period_days=trap_params["period_days"],
        duration_days=trap_params["duration_hours"] / 24.0,
    )

    return RefinedCandidate(
        candidate=candidate,
        period_days=trap_params["period_days"],
        epoch_days=trap_params["epoch_days"],
        duration_hours=trap_params["duration_hours"],
        measured_depth_ppm=measured_depth,
        dilution_corrected_depth_ppm=dilution_corrected_depth,
        ingress_fraction=trap_params["ingress_fraction"],
        chi2_reduced=chi2_nu,
        bic=trap_bic,
        batman_delta_bic=batman_delta_bic,
        phase_coverage=coverage,
        trapezoid_model_curve=trap_curve,
        batman_model_curve=batman_curve,
    )
