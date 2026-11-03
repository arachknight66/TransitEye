"""Tests for BLS periodogram, Lomb-Scargle variability, and iterative candidate search."""

from __future__ import annotations

import numpy as np

from transiteye.detection.bls import run_bls
from transiteye.detection.peaks import find_candidates
from transiteye.detection.variability import compute_harmonic_distance, run_variability_search
from transiteye.domain.models import LightCurve, Provenance
from transiteye.preprocessing.detrend import two_pass_detrend
from transiteye.preprocessing.representation import prepare_light_curve


def _make_dual_curve(
    num_days: float = 20.0,
    cadence_minutes: float = 2.0,
    transit_params: tuple[tuple[float, float, float, float], ...] = (),
    sin_period: float | None = None,
) -> any:
    pts_per_day = 1440.0 / cadence_minutes
    total_pts = int(num_days * pts_per_day)
    t = np.linspace(1.0, 1.0 + num_days, total_pts)
    f = np.ones(total_pts)

    if sin_period is not None:
        f += 0.01 * np.sin(2 * np.pi * t / sin_period)

    # Injected transits: (epoch, period, duration_hours, depth_ppm)
    for t0, p, dur_h, depth_ppm in transit_params:
        dur_days = dur_h / 24.0
        phase = ((t - t0 + 0.5 * p) % p) - 0.5 * p
        in_tr = np.abs(phase) <= (0.5 * dur_days)
        f[in_tr] -= depth_ppm / 1e6

    prov = Provenance(
        source_uri="file:///test/lc.fits",
        source_path="/test/lc.fits",
        checksum_sha256="0" * 64,
        product_type="tess_light_curve",
        target_id="TIC_12345",
        sector=1,
        time_system="TDB",
        time_reference="2457000",
    )
    lc = LightCurve(
        time_days=t,
        flux=f,
        flux_error=np.full_like(f, 0.0005),
        quality=np.zeros_like(f, dtype=np.int64),
        cadence_number=np.arange(len(f), dtype=np.int64),
        centroid_column=None,
        centroid_row=None,
        flux_kind="PDCSAP_FLUX",
        flux_unit="e-/s",
        provenance=prov,
    )
    prep = prepare_light_curve(lc)
    dual = two_pass_detrend(prep, baseline_window_days=1.0)
    return dual


def test_bls_detects_injected_transit() -> None:
    # 1 injected planet: P = 3.5 days, T0 = 2.1 days, dur = 3.0 h, depth = 5000 ppm (0.5%)
    dual = _make_dual_curve(
        num_days=21.0,
        transit_params=((2.1, 3.5, 3.0, 5000.0),),
    )
    bls = run_bls(dual.prepared.time_days, dual.detrended_flux, min_period=1.0, max_period=10.0)

    # Assert period recovery within 1%
    assert abs(bls.best_period - 3.5) / 3.5 < 0.01
    assert bls.sde > 5.0
    assert bls.best_snr > 10.0
    assert abs(bls.best_epoch_days - 2.1) < 0.2


def test_candidate_search_respects_custom_period_range() -> None:
    dual = _make_dual_curve(
        num_days=8.0, cadence_minutes=20.0,
        transit_params=((1.5, 2.0, 1.0, 10000.0),),
    )
    result = find_candidates(
        dual, max_candidates=1, min_period=1.5, max_period=2.5,
        min_duration_hours=0.5, max_duration_hours=2.0, device="cpu",
    )
    assert np.min(result.primary_bls.periods) >= 1.5
    assert np.max(result.primary_bls.periods) <= 2.5


def test_variability_search_detects_sinusoid() -> None:
    dual = _make_dual_curve(
        num_days=20.0,
        sin_period=5.25,
    )
    var = run_variability_search(dual.prepared.time_days, dual.prepared.flux_normalized)

    # Assert period recovery within 2%
    assert abs(var.best_period_days - 5.25) / 5.25 < 0.02
    assert var.peak_power > 0.5
    assert var.false_alarm_prob < 1e-4


def test_harmonic_distance_calculation() -> None:
    # Identical periods
    assert compute_harmonic_distance(3.0, 3.0) == 0.0
    # Double period
    assert compute_harmonic_distance(6.0, 3.0) == 0.0
    # Half period
    assert compute_harmonic_distance(1.5, 3.0) == 0.0
    # Non-harmonic
    assert compute_harmonic_distance(3.1415, 7.89) > 0.05


def test_find_candidates_guardrails_on_short_data() -> None:
    t = np.linspace(1.0, 1.5, 100)  # only 100 points, 0.5 day baseline
    prov = Provenance(
        source_uri="file:///test/lc.fits",
        source_path="/test/lc.fits",
        checksum_sha256="0" * 64,
        product_type="tess_light_curve",
        target_id="SHORT",
        sector=1,
        time_system="TDB",
        time_reference="2457000",
    )
    lc = LightCurve(
        time_days=t,
        flux=np.ones_like(t),
        flux_error=None,
        quality=None,
        cadence_number=None,
        centroid_column=None,
        centroid_row=None,
        flux_kind="PDCSAP_FLUX",
        flux_unit=None,
        provenance=prov,
    )
    prep = prepare_light_curve(lc)
    dual = two_pass_detrend(prep)
    res = find_candidates(dual)
    assert len(res.candidates) == 0


def test_find_candidates_iterative_multi_planet() -> None:
    # Inject 2 distinct planets:
    # Planet 1: P = 2.5d, depth = 8000 ppm
    # Planet 2: P = 6.0d, depth = 6000 ppm
    dual = _make_dual_curve(
        num_days=25.0,
        transit_params=(
            (1.5, 2.5, 2.5, 8000.0),
            (2.2, 6.0, 3.5, 6000.0),
        ),
    )
    res = find_candidates(dual, max_candidates=2, snr_threshold=6.0)

    assert len(res.candidates) >= 2
    c1, c2 = res.candidates[0], res.candidates[1]

    # Verify both periods recovered
    periods = {round(c1.period_days, 1), round(c2.period_days, 1)}
    assert 2.5 in periods
    assert 6.0 in periods

    # Verify candidate properties
    assert c1.observed_event_count >= 2
    assert len(c1.aliases_days) == 2
    assert c1.candidate_id.startswith("TIC_12345_")
