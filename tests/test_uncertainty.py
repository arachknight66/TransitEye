"""Tests for red-noise SNR, bootstrap intervals, empirical significance, and result bundles."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits

from transiteye.characterization.fitting import (
    evaluate_trapezoid,
    refine_candidate,
)
from transiteye.characterization.uncertainty import (
    compute_empirical_significance,
    compute_parameter_intervals,
    compute_red_noise_snr,
)
from transiteye.domain.models import Candidate
from transiteye.execution.compute import has_cuda
from transiteye.execution.service import analyze_light_curve
from transiteye.io.fits import read_tess_light_curve
from transiteye.reporting.bundles import read_result_bundle


def test_red_noise_snr_accounts_for_correlation() -> None:
    t = np.linspace(0.0, 10.0, 1000)
    # Model with depth 0.01
    model = evaluate_trapezoid(t, epoch_days=1.0, period_days=2.0, duration_days=2.4 / 24.0, depth=0.01, ingress_fraction=0.2)

    # 1. White noise residuals
    rng = np.random.default_rng(42)
    white_noise = rng.normal(0, 0.001, size=len(t))
    flux_white = model + white_noise

    snr_white = compute_red_noise_snr(
        flux_white, model, t, depth_ppm=10000.0, duration_hours=2.4, observed_event_count=5
    )

    # 2. Correlated red noise residuals (white noise + correlated low-frequency noise)
    red_component = 0.001 * np.sin(2 * np.pi * t / 0.5)
    flux_red = model + white_noise + red_component

    snr_red = compute_red_noise_snr(
        flux_red, model, t, depth_ppm=10000.0, duration_hours=2.4, observed_event_count=5
    )

    assert snr_white > 0.0
    assert snr_red > 0.0
    # Correlated noise should yield a lower SNR than white noise
    assert snr_white > snr_red


def test_empirical_significance_resolution_and_determinism() -> None:
    t = np.linspace(0.0, 10.0, 1000)
    model = evaluate_trapezoid(t, epoch_days=1.0, period_days=2.0, duration_days=2.4 / 24.0, depth=0.01, ingress_fraction=0.2)
    flux = model + np.random.normal(0, 0.0005, size=len(t))

    sig1, res1 = compute_empirical_significance(
        t, flux, model, candidate_period=2.0, candidate_duration_hours=2.4, observed_snr=15.0, n_surrogates=200, random_seed=42
    )
    sig2, res2 = compute_empirical_significance(
        t, flux, model, candidate_period=2.0, candidate_duration_hours=2.4, observed_snr=15.0, n_surrogates=200, random_seed=42
    )

    # Determinism from recorded seed
    assert sig1 == sig2
    assert res1 == res2
    assert np.isclose(res1, 1.0 / 201.0, atol=1e-5)
    assert sig1 > 0.95  # Significant injection


def test_parameter_intervals_coverage() -> None:
    t = np.linspace(0.0, 10.0, 1500)
    true_period = 2.5
    true_depth_ppm = 8000.0
    true_dur_hours = 2.4
    true_epoch = 1.0

    model = evaluate_trapezoid(
        t, epoch_days=true_epoch, period_days=true_period, duration_days=true_dur_hours / 24.0, depth=true_depth_ppm / 1e6, ingress_fraction=0.25
    )
    rng = np.random.default_rng(123)
    flux = model + rng.normal(0, 0.0002, size=len(t))

    cand = Candidate(
        candidate_id="TEST_BOOT",
        period_days=true_period,
        epoch_days=true_epoch,
        duration_hours=true_dur_hours,
        depth_ppm=true_depth_ppm,
        observed_event_count=4,
        detection_snr=20.0,
        search_significance=None,
    )
    refined = refine_candidate(cand, t, flux, flux_err=None)

    intervals = compute_parameter_intervals(t, flux, flux_err=None, refined=refined, n_boot=50, random_seed=123)

    assert "period_days" in intervals
    assert "depth_ppm" in intervals
    p_int_95 = intervals["period_days"]["95"]
    assert p_int_95.lower is not None and p_int_95.upper is not None
    # True period must be contained within 95% interval
    assert p_int_95.lower <= true_period <= p_int_95.upper


def _write_synthetic_fits(path: Path, t: np.ndarray, f: np.ndarray) -> None:
    cols = [
        fits.Column(name="TIME", format="D", array=t),
        fits.Column(name="PDCSAP_FLUX", format="D", array=f),
        fits.Column(name="PDCSAP_FLUX_ERR", format="D", array=np.full_like(f, 0.0005)),
        fits.Column(name="QUALITY", format="K", array=np.zeros(len(t), dtype=np.int64)),
        fits.Column(name="CADENCENO", format="K", array=np.arange(len(t), dtype=np.int64)),
        fits.Column(name="MOM_CENTR1", format="D", array=np.zeros(len(t))),
        fits.Column(name="MOM_CENTR2", format="D", array=np.zeros(len(t))),
    ]
    table = fits.BinTableHDU.from_columns(cols)
    table.header["TIMESYS"] = "TDB"
    table.header["BJDREFI"] = 2457000
    table.header["CROWDSAP"] = 0.90
    primary = fits.PrimaryHDU()
    primary.header["TICID"] = 999888
    primary.header["SECTOR"] = 1
    primary.header["MISSION"] = "TESS"
    primary.header["CREATOR"] = "test"
    fits.HDUList([primary, table]).writeto(path)


def test_end_to_end_analysis_service_and_bundle(tmp_path: Path) -> None:
    # 15 days, 2 min cadence = 10800 pts
    t = np.linspace(1.0, 16.0, 3000)
    model = evaluate_trapezoid(
        t, epoch_days=2.0, period_days=3.0, duration_days=3.0 / 24.0, depth=0.006, ingress_fraction=0.25
    )
    flux = model + np.random.default_rng(42).normal(0, 0.0003, size=len(t))

    fits_file = tmp_path / "test_target_lc.fits"
    _write_synthetic_fits(fits_file, t, flux)

    lc = read_tess_light_curve(fits_file)
    output_dir = tmp_path / "bundle_out"

    progress_events = []

    def _prog(e: dict[str, object]) -> None:
        progress_events.append(e)

    res, bundle_dir = analyze_light_curve(
        lc,
        max_candidates=1,
        snr_threshold=6.0,
        device="cpu",
        output_dir=output_dir,
        progress_callback=_prog,
    )

    assert res.status.value == "complete"
    assert len(res.candidates) == 1
    cand = res.candidates[0]
    assert abs(cand.period_days - 3.0) < 0.05
    assert cand.detection_snr > 10.0
    assert cand.search_significance is not None and cand.search_significance > 0.90

    # Verify progress events streamed
    assert len(progress_events) >= 5
    assert progress_events[-1]["stage"] == "complete"

    # Verify bundle structure
    bundle = read_result_bundle(bundle_dir)
    assert bundle["result"]["analysis_id"] == res.analysis_id
    assert "time_days" in bundle["arrays"]
    assert "flux_detrended" in bundle["arrays"]
    assert "bls_periods" in bundle["arrays"]
    assert "cand_00_trapezoid_model" in bundle["arrays"]
    assert bundle["candidates"].num_rows == 1
    assert bundle["candidates"].column("candidate_id")[0].as_py() == cand.candidate_id
    assert "predicted_class" in bundle["candidates"].column_names
    assert "source_assessment" in bundle["candidates"].column_names
    assert cand.candidate_id in res.classifications
    assert res.classifications[cand.candidate_id].source_assessment.value == "target_consistent"


def test_end_to_end_cuda_acceleration_if_available(tmp_path: Path) -> None:
    if not has_cuda():
        return

    t = np.linspace(1.0, 16.0, 2000)
    model = evaluate_trapezoid(t, epoch_days=2.0, period_days=2.5, duration_days=2.4 / 24.0, depth=0.008, ingress_fraction=0.25)
    flux = model + np.random.default_rng(42).normal(0, 0.0003, size=len(t))

    fits_file = tmp_path / "cuda_target_lc.fits"
    _write_synthetic_fits(fits_file, t, flux)

    lc = read_tess_light_curve(fits_file)
    output_dir = tmp_path / "bundle_cuda_out"

    res, bundle_dir = analyze_light_curve(
        lc,
        device="cuda",
        output_dir=output_dir,
    )
    assert res.status.value == "complete"
    assert len(res.candidates) >= 1
    assert (bundle_dir / "candidates.parquet").is_file()
