"""Tests for trapezoid and batman transit fitting, parameter estimation, and dilution correction."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import trapezoid

from transiteye.characterization.fitting import (
    compute_phase_coverage,
    estimate_limb_darkening_coefficients,
    evaluate_trapezoid,
    fit_batman_refinement,
    fit_trapezoid,
    refine_candidate,
)
from transiteye.domain.models import Candidate


def test_evaluate_trapezoid_morphology() -> None:
    t = np.array([0.0, 0.02, 0.05, 0.1, 0.5])
    # T0=0.0, P=1.0, Dur=0.1d (2.4h), Depth=0.01 (10,000 ppm), Ingress=0.5 (half duration = 0.05d, ramp=0.025d)
    flux = evaluate_trapezoid(t, epoch_days=0.0, period_days=1.0, duration_days=0.1, depth=0.01, ingress_fraction=0.5)

    # Center is at minimum (1.0 - 0.01 = 0.99)
    assert np.isclose(flux[0], 0.99, atol=1e-4)
    # Outside duration (0.5d) is baseline 1.0
    assert np.isclose(flux[-1], 1.0, atol=1e-4)


def test_exposure_integral_converges_and_changes_ingress_shape() -> None:
    # Independent numerical integration of the instantaneous model over each exposure.
    times = np.array([-0.05, -0.035, 0.0, 0.035, 0.05])
    exposure = 0.02
    offsets = np.linspace(-exposure / 2, exposure / 2, 10001)
    instantaneous = evaluate_trapezoid(
        (times[:, None] + offsets).ravel(), 0.0, 1.0, 0.1, 0.01, 0.5, supersample=1
    ).reshape(len(times), -1)
    reference = trapezoid(instantaneous, offsets, axis=1) / exposure
    integrated = evaluate_trapezoid(
        times, 0.0, 1.0, 0.1, 0.01, 0.5, supersample=1001, exposure_days=exposure
    )
    point_sample = evaluate_trapezoid(times, 0.0, 1.0, 0.1, 0.01, 0.5, supersample=1)
    np.testing.assert_allclose(integrated, reference, atol=2e-6)
    assert abs(point_sample[0] - integrated[0]) > 1e-4


@pytest.mark.parametrize("ingress_fraction", [0.0, 0.2, 1.0])
@pytest.mark.parametrize("exposure", [0.001, 0.2, 1.4])
def test_exact_integral_matches_independent_quadrature(
    ingress_fraction: float, exposure: float,
) -> None:
    times = np.array([-0.5, -0.04, 0.0, 0.04, 0.5])
    offsets = np.linspace(-exposure / 2, exposure / 2, 50001)
    reference_flux = evaluate_trapezoid(
        (times[:, None] + offsets).ravel(), 0.0, 1.0, 0.1, 0.01,
        ingress_fraction, supersample=1,
    ).reshape(len(times), -1)
    reference = trapezoid(reference_flux, offsets, axis=1) / exposure
    exact = evaluate_trapezoid(
        times, 0.0, 1.0, 0.1, 0.01, ingress_fraction,
        supersample=5, exposure_days=exposure,
    )
    np.testing.assert_allclose(exact, reference, atol=2e-7, rtol=0)


def test_fit_recovers_long_exposure_trapezoid() -> None:
    times = np.arange(0.0, 6.0, 2.0 / 1440.0)
    exposure = 20.0 / 1440.0
    offsets = np.linspace(-exposure / 2, exposure / 2, 201)
    instantaneous = evaluate_trapezoid(
        (times[:, None] + offsets).ravel(), 1.0, 2.0, 0.1, 0.01, 0.2,
        supersample=1,
    ).reshape(len(times), -1)
    flux = trapezoid(instantaneous, offsets, axis=1) / exposure
    params, _, _, _ = fit_trapezoid(
        times, flux, None, init_epoch=1.0, init_period=2.0,
        init_duration_hours=2.4, init_depth_ppm=10000.0, exposure_days=exposure,
    )
    assert abs(params["duration_hours"] - 2.4) < 0.01
    assert abs(params["ingress_fraction"] - 0.2) < 0.01


def test_fit_trapezoid_recovers_synthetic_signal() -> None:
    t = np.linspace(0.0, 10.0, 2000)
    # True: T0=1.5, P=2.5, Dur=3.0h (0.125d), Depth=5000 ppm (0.005)
    true_flux = evaluate_trapezoid(
        t, epoch_days=1.5, period_days=2.5, duration_days=3.0 / 24.0, depth=0.005, ingress_fraction=0.3
    )
    rng = np.random.default_rng(42)
    noisy_flux = true_flux + rng.normal(0, 0.0002, size=len(t))

    params, _, chi2_nu, bic = fit_trapezoid(
        t,
        noisy_flux,
        flux_err=None,
        init_epoch=1.48,
        init_period=2.51,
        init_duration_hours=2.8,
        init_depth_ppm=4500.0,
    )

    assert abs(params["period_days"] - 2.5) < 0.02
    assert abs(params["epoch_days"] - 1.5) < 0.05
    assert abs(params["depth_ppm"] - 5000.0) < 300.0
    assert abs(params["duration_hours"] - 3.0) < 0.3
    assert chi2_nu > 0.0
    assert np.isfinite(bic)


def test_fit_trapezoid_accepts_deep_eclipse_and_rejects_impossible_depth() -> None:
    times = np.linspace(0.0, 6.0, 1500)
    flux = evaluate_trapezoid(times, 1.0, 2.0, 0.15, 0.76, 0.2)
    params, _, _, _ = fit_trapezoid(times, flux, None, 1.0, 2.0, 3.6, 760_000.0)
    assert abs(params["depth_ppm"] - 760_000.0) < 100.0
    for depth in (-1.0, 1_000_001.0, float("nan")):
        with pytest.raises(ValueError, match="initial depth must be finite"):
            fit_trapezoid(times, flux, None, 1.0, 2.0, 3.6, depth)


def test_fit_batman_refinement() -> None:
    t = np.linspace(0.0, 5.0, 1000)
    true_flux = evaluate_trapezoid(
        t, epoch_days=1.0, period_days=2.0, duration_days=2.4 / 24.0, depth=0.01, ingress_fraction=0.2
    )

    res = fit_batman_refinement(
        t,
        true_flux,
        flux_err=None,
        init_epoch=1.0,
        init_period=2.0,
        init_duration_days=0.1,
        init_depth=0.01,
    )
    assert res is not None
    params, curve, bic = res
    assert abs(params["period_days"] - 2.0) < 0.05
    assert params["rp_rstar"] > 0.05
    assert len(curve) == len(t)
    assert np.isfinite(bic)


def test_nonconverged_optimizer_does_not_produce_fit(monkeypatch) -> None:
    monkeypatch.setattr(
        "transiteye.characterization.fitting.least_squares",
        lambda *args, **kwargs: SimpleNamespace(success=False, message="evaluation budget exhausted"),
    )
    times = np.linspace(0.0, 3.0, 300)
    flux = np.ones_like(times)
    with pytest.raises(RuntimeError, match="did not converge.*evaluation budget exhausted"):
        fit_trapezoid(times, flux, None, 1.0, 2.0, 2.4, 1000.0)
    assert fit_batman_refinement(times, flux, None, 1.0, 2.0, 0.1, 0.001) is None


def test_phase_coverage_calculation() -> None:
    # Uniform dense sampling in transit
    t_dense = np.linspace(-0.05, 0.05, 50)
    cov_dense = compute_phase_coverage(t_dense, epoch_days=0.0, period_days=1.0, duration_days=0.1, n_bins=10)
    assert cov_dense > 0.8

    # Single point in transit
    t_sparse = np.array([0.0])
    cov_sparse = compute_phase_coverage(t_sparse, epoch_days=0.0, period_days=1.0, duration_days=0.1, n_bins=10)
    assert np.isclose(cov_sparse, 0.1, atol=1e-3)


def test_refine_candidate_dilution_correction() -> None:
    t = np.linspace(0.0, 10.0, 2000)
    flux = evaluate_trapezoid(
        t, epoch_days=1.0, period_days=2.0, duration_days=2.0 / 24.0, depth=0.008, ingress_fraction=0.25
    )
    cand = Candidate(
        candidate_id="TEST_01",
        period_days=2.0,
        epoch_days=1.0,
        duration_hours=2.0,
        depth_ppm=8000.0,
        observed_event_count=5,
        detection_snr=15.0,
        search_significance=None,
    )
    # Crowdsap = 0.80 -> Dilution corrected depth should be 8000 / 0.80 = 10000 ppm
    refined = refine_candidate(cand, t, flux, flux_err=None, crowdsap=0.80)

    assert abs(refined.measured_depth_ppm - 8000.0) < 500.0
    assert refined.dilution_corrected_depth_ppm is not None
    assert np.isclose(
        refined.dilution_corrected_depth_ppm,
        refined.measured_depth_ppm / 0.80,
        atol=1e-3,
    )
    assert refined.batman_delta_bic is not None
    assert refined.phase_coverage > 0.5


def test_estimate_limb_darkening_coefficients() -> None:
    # Test cool M dwarf
    u_cool = estimate_limb_darkening_coefficients(teff=3200.0, logg=4.8)
    assert np.isclose(u_cool[0], 0.48, atol=0.01)
    assert np.isclose(u_cool[1], 0.28, atol=0.01)

    # Test solar analog (5770 K -> 0.32, 0.26)
    u_sol = estimate_limb_darkening_coefficients(teff=5770.0, logg=4.44)
    assert np.isclose(u_sol[0], 0.32, atol=0.01)
    assert np.isclose(u_sol[1], 0.26, atol=0.01)

    # Test hot star (9000 K -> 0.18, 0.22)
    u_hot = estimate_limb_darkening_coefficients(teff=9000.0, logg=4.0)
    assert np.isclose(u_hot[0], 0.18, atol=0.01)
    assert np.isclose(u_hot[1], 0.22, atol=0.01)

    # Test missing / None fallback (0.30, 0.22)
    u_none = estimate_limb_darkening_coefficients(teff=None, logg=None)
    assert np.isclose(u_none[0], 0.30, atol=0.01)
    assert np.isclose(u_none[1], 0.22, atol=0.01)

