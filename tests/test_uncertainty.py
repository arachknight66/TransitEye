"""Tests for red-noise SNR, bootstrap intervals, empirical significance, and result bundles."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
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
from transiteye.domain.models import AnalysisSettings, Candidate
from transiteye.execution.compute import has_cuda
from transiteye.execution.service import analyze_light_curve
from transiteye.io.fits import read_tess_light_curve
from transiteye.reporting.bundles import read_result_bundle, verify_bundle_integrity


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


def test_unvalidated_fixed_phase_significance_is_disabled() -> None:
    t = np.linspace(0.0, 10.0, 1000)
    model = evaluate_trapezoid(t, epoch_days=1.0, period_days=2.0, duration_days=2.4 / 24.0, depth=0.01, ingress_fraction=0.2)
    with pytest.raises(RuntimeError, match="Search-wide empirical significance is unavailable"):
        compute_empirical_significance(
            t, model, model, candidate_period=2.0,
            candidate_duration_hours=2.4, observed_snr=15.0,
        )


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


def test_bootstrap_refits_use_candidate_exposure(monkeypatch) -> None:
    from transiteye.characterization import uncertainty

    t = np.linspace(0.0, 6.0, 800)
    exposure = 20.0 / 1440.0
    flux = evaluate_trapezoid(
        t, epoch_days=1.0, period_days=2.0, duration_days=0.1,
        depth=0.01, ingress_fraction=0.2, exposure_days=exposure,
    )
    candidate = Candidate(
        candidate_id="exposure", period_days=2.0, epoch_days=1.0,
        duration_hours=2.4, depth_ppm=10000.0, observed_event_count=3,
        detection_snr=3.0, search_significance=None,
    )
    refined = refine_candidate(candidate, t, flux, flux_err=None, exposure_days=exposure)
    original_fit = uncertainty.fit_trapezoid
    recorded = []

    def capture_fit(*args, **kwargs):
        recorded.append(kwargs["exposure_days"])
        return original_fit(*args, **kwargs)

    monkeypatch.setattr(uncertainty, "fit_trapezoid", capture_fit)
    compute_parameter_intervals(t, flux, None, refined, n_boot=10, random_seed=9)
    assert recorded == [exposure] * 10


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
    assert cand.search_significance is None
    assert res.missing_reasons[f"{cand.candidate_id}.search_significance"].value == "not_computed"
    assert res.missing_reasons[f"{cand.candidate_id}.classification"].value == "not_computed"
    assert res.classifications[cand.candidate_id].probabilities == {}
    assert res.classifications[cand.candidate_id].abstained

    # Verify progress events streamed
    assert len(progress_events) >= 5
    assert progress_events[-1]["stage"] == "complete"

    # Verify bundle structure
    bundle = read_result_bundle(bundle_dir)
    assert bundle["result"]["analysis_id"] == res.analysis_id
    assert verify_bundle_integrity(bundle_dir)
    assert bundle["result"]["schema_version"] == "3"
    assert bundle["result"]["analysis_settings"]["device"] == "cpu"
    assert bundle["result"]["resolved_device"] == "cpu"
    assert bundle["result"]["analysis_settings"]["max_candidates"] == 1
    assert bundle["result"]["analysis_settings"]["exposure_status"] == "fixed_assumption_unverified"
    assert len(bundle["result"]["code_sha256"]) == 64
    assert bundle["result"]["model_id"] is None
    assert bundle["result"]["model_sha256"] is None
    assert isinstance(bundle["result"]["candidate_seeds"][cand.candidate_id], int)
    assert "time_days" in bundle["arrays"]
    assert "flux_detrended" in bundle["arrays"]
    assert "bls_periods" in bundle["arrays"]
    assert "cand_00_trapezoid_model" in bundle["arrays"]
    assert bundle["candidates"].num_rows == 1
    assert bundle["candidates"].column("candidate_id")[0].as_py() == cand.candidate_id
    assert "predicted_class" in bundle["candidates"].column_names
    assert bundle["candidates"].column("search_significance")[0].as_py() is None
    assert bundle["candidates"].column("calibrated_confidence")[0].as_py() is None
    assert "source_assessment" in bundle["candidates"].column_names
    assert cand.candidate_id in res.classifications
    assert res.classifications[cand.candidate_id].source_assessment.value == "unresolved"
    assert res.missing_reasons[f"{cand.candidate_id}.crowdsap"].value == "source_data_missing"
    assert res.missing_reasons[f"{cand.candidate_id}.sap_to_pdcsap_depth_ratio"].value == "source_data_missing"


def test_short_light_curve_is_insufficient_not_a_non_detection(tmp_path: Path) -> None:
    t = np.linspace(1.0, 2.0, 100)
    fits_file = tmp_path / "short_lc.fits"
    _write_synthetic_fits(fits_file, t, np.ones_like(t))
    result, bundle_path = analyze_light_curve(
        read_tess_light_curve(fits_file), output_dir=tmp_path / "short_bundle",
        analysis_settings=AnalysisSettings(
            max_candidates=1, snr_threshold=7.0, screen_snr_threshold=5.0,
            device="cpu", baseline_window_days=1.0, bootstrap_replicates=10,
            trapezoid_exposure_days=0.01,
        ),
    )
    assert result.status.value == "insufficient_data"
    assert "200 valid cadences" in result.error_message
    assert result.candidates == ()
    document = read_result_bundle(bundle_path)["result"]
    assert document["status"] == "insufficient_data"
    assert document["analysis_settings"]["baseline_window_days"] == 1.0
    assert document["analysis_settings"]["bootstrap_replicates"] == 10
    assert document["analysis_settings"]["trapezoid_exposure_days"] == 0.01
    assert document["analysis_settings"]["exposure_status"] == "user_supplied"


def test_failed_physical_fit_is_reported_but_skipped_fit_is_not(monkeypatch, tmp_path: Path) -> None:
    from dataclasses import replace

    from transiteye.execution import service

    times = np.linspace(0.0, 10.0, 2000)
    flux = evaluate_trapezoid(times, 1.0, 2.0, 0.1, 0.01, 0.2)
    flux += np.random.default_rng(39).normal(0, 0.0002, len(times))
    fits_file = tmp_path / "physical_fit.fits"
    _write_synthetic_fits(fits_file, times, flux)
    light_curve = read_tess_light_curve(fits_file)
    real_refine = service.refine_candidate

    def no_physical_model(*args, **kwargs):
        kwargs["batman_snr_threshold"] = 1e9
        return replace(real_refine(*args, **kwargs), ingress_fraction=0.2)

    monkeypatch.setattr(service, "refine_candidate", no_physical_model)
    for screen_threshold, expected_failure in ((5.0, True), (1e9, False)):
        settings = AnalysisSettings(
            max_candidates=1, snr_threshold=6.0, screen_snr_threshold=screen_threshold,
            device="cpu", bootstrap_replicates=10,
        )
        result, path = service.analyze_light_curve(
            light_curve, analysis_settings=settings,
            output_dir=tmp_path / f"screen-{screen_threshold}",
        )
        candidate = result.candidates[0]
        assert ("physical_fit_failed" in candidate.flags) is expected_failure
        key = f"{candidate.candidate_id}.physical_fit"
        assert (key in result.missing_reasons) is expected_failure
        document = read_result_bundle(path)
        assert ("physical_fit_failed" in document["candidates"].column("flags")[0].as_py()) is expected_failure


def test_failed_candidate_fit_does_not_drop_other_candidates(monkeypatch, tmp_path: Path) -> None:
    from dataclasses import replace

    from transiteye.execution import service

    t = np.linspace(0.0, 10.0, 2000)
    flux = evaluate_trapezoid(
        t, epoch_days=1.0, period_days=2.0, duration_days=0.1,
        depth=0.01, ingress_fraction=0.2,
    ) + np.random.default_rng(71).normal(0, 0.0002, len(t))
    fits_file = tmp_path / "multi_lc.fits"
    _write_synthetic_fits(fits_file, t, flux)
    real_search = service.find_candidates
    real_refine = service.refine_candidate

    def two_candidates(*args, **kwargs):
        detection = real_search(*args, **kwargs)
        assert detection.candidates
        candidate = detection.candidates[0]
        return replace(detection, candidates=(
            replace(candidate, candidate_id="bad"), replace(candidate, candidate_id="good")
        ))

    def sometimes_fail(candidate, *args, **kwargs):
        if candidate.candidate_id == "bad":
            raise ValueError("deliberate fit failure")
        refined = real_refine(candidate, *args, **kwargs)
        return replace(refined, batman_model_curve=np.ones_like(refined.trapezoid_model_curve))

    monkeypatch.setattr(service, "find_candidates", two_candidates)
    monkeypatch.setattr(service, "refine_candidate", sometimes_fail)
    result, bundle_dir = service.analyze_light_curve(
        read_tess_light_curve(fits_file), max_candidates=2, device="cpu",
        screen_snr_threshold=1e9, output_dir=tmp_path / "bundle",
    )
    assert [candidate.candidate_id for candidate in result.candidates] == ["good", "bad"]
    assert "physical_quadrature_unvalidated" in result.candidates[0].flags
    assert "fit_failed" in result.candidates[1].flags
    assert result.candidate_errors["bad"] == "ValueError: deliberate fit failure"
    assert result.classifications["bad"].source_assessment.value == "not_assessed"
    assert result.missing_reasons["bad.fit"].value == "fit_failed"
    bundle = read_result_bundle(bundle_dir)
    assert bundle["candidates"].column("candidate_id").to_pylist() == ["good", "bad"]
    assert "physical_quadrature_unvalidated" in bundle["candidates"].column("flags")[0].as_py()
    assert "physical_quadrature_unvalidated" in bundle["result"]["candidates"][0]["flags"]
    assert "cand_00_trapezoid_model" in bundle["arrays"]
    assert "cand_01_trapezoid_model" not in bundle["arrays"]
    assert verify_bundle_integrity(bundle_dir)
    from transiteye_desktop.bundle_loader import load_result_bundle

    desktop_bundle = load_result_bundle(bundle_dir)
    assert "physical_quadrature_unvalidated" in desktop_bundle.candidates[0].flags
    assert "fit_failed" in desktop_bundle.candidates[1].flags
    assert desktop_bundle.candidates[1].trapezoid_model_curve is None
    assert desktop_bundle.candidates[1].source_assessment == "not_assessed"


def test_training_and_inference_use_same_candidate_features(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from transiteye.classification.models import MORPHOLOGY_FEATURE_NAMES
    from transiteye.classification.train import _extract_lc_feature_vector
    from transiteye.domain.models import Classification

    t = np.linspace(0.0, 10.0, 2000)
    flux = evaluate_trapezoid(
        t, epoch_days=1.0, period_days=2.0, duration_days=0.1,
        depth=0.01, ingress_fraction=0.2,
    ) + np.random.default_rng(23).normal(0, 0.0002, len(t))
    fits_file = tmp_path / "candidate_lc.fits"
    _write_synthetic_fits(fits_file, t, flux)
    with fits.open(fits_file, mode="update") as hdul:
        hdul[0].header["TEFF"] = 3200.0
    light_curve = read_tess_light_curve(fits_file)
    training_vector = _extract_lc_feature_vector(light_curve, MORPHOLOGY_FEATURE_NAMES)
    assert training_vector is not None

    class CaptureModel:
        metadata = SimpleNamespace(model_id="test")
        artifact_sha256 = None
        features = None

        def predict(self, features, source_assessment):
            self.features = features
            return Classification(
                probabilities={}, predicted_class=None, abstained=True,
                source_assessment=source_assessment, model_version="test",
            )

    model = CaptureModel()
    result, _ = analyze_light_curve(
        light_curve, model_bundle=model, output_dir=tmp_path / "inference",
        analysis_settings=AnalysisSettings(
            max_candidates=1, snr_threshold=1.5, screen_snr_threshold=0.0,
            device="auto", bootstrap_replicates=10,
        ),
    )
    assert result.candidates and model.features is not None
    inference_vector = [model.features.get(name, 0.0) for name in MORPHOLOGY_FEATURE_NAMES]
    np.testing.assert_allclose(training_vector, inference_vector, rtol=1e-10, atol=1e-10)


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
