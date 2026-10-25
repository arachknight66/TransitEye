"""Tests for duration-aware two-pass biweight detrending and compute abstraction."""

from __future__ import annotations

import numpy as np

from transiteye.domain.models import LightCurve, Provenance
from transiteye.execution.compute import ComputeDevice, has_cuda, resolve_device
from transiteye.preprocessing.detrend import (
    build_in_transit_mask,
    detrend_biweight,
    two_pass_detrend,
)
from transiteye.preprocessing.representation import prepare_light_curve


def _make_prepared_curve(
    period_var: float = 4.0,
    transit_depth: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, any]:
    # 15 days baseline, 2-minute cadence (~10,800 points)
    t = np.linspace(1.0, 15.0, 3000)
    # Slow stellar variability (amplitude 1%)
    variability = 0.01 * np.sin(2 * np.pi * t / period_var)
    f = 1.0 + variability

    # Inject periodic transits: period = 3.0 days, epoch = 2.0 days, duration = 3.0 hours (0.125 days)
    p_tr = 3.0
    t0_tr = 2.0
    dur_days = 3.0 / 24.0
    phase = ((t - t0_tr + 0.5 * p_tr) % p_tr) - 0.5 * p_tr
    in_tr = np.abs(phase) <= (0.5 * dur_days)
    f[in_tr] -= transit_depth

    # Wrap in LightCurve
    prov = Provenance(
        source_uri="file:///test/lc.fits",
        source_path="/test/lc.fits",
        checksum_sha256="0" * 64,
        product_type="tess_light_curve",
        target_id="123456",
        sector=1,
        time_system="TDB",
        time_reference="2457000",
    )
    lc = LightCurve(
        time_days=t,
        flux=f,
        flux_error=np.full_like(f, 0.001),
        quality=np.zeros_like(f, dtype=np.int64),
        cadence_number=np.arange(len(f), dtype=np.int64),
        centroid_column=None,
        centroid_row=None,
        flux_kind="PDCSAP_FLUX",
        flux_unit="e-/s",
        provenance=prov,
    )
    prep = prepare_light_curve(lc)
    return t, in_tr, prep


def test_compute_device_resolution() -> None:
    assert resolve_device(ComputeDevice.CPU) == "cpu"
    auto_dev = resolve_device(ComputeDevice.AUTO)
    assert auto_dev in ("cuda", "cpu")
    if has_cuda():
        assert resolve_device(ComputeDevice.CUDA) == "cuda"


def test_build_in_transit_mask() -> None:
    t = np.linspace(0.0, 10.0, 1000)
    events = ((1.0, 2.0, 2.4),)  # T0=1.0, P=2.0, Dur=2.4h (0.1d)
    mask = build_in_transit_mask(t, events, buffer_factor=1.0)
    assert np.any(mask)
    # Check that t=1.0, 3.0, 5.0 are masked
    for center in (1.0, 3.0, 5.0, 7.0, 9.0):
        idx = np.argmin(np.abs(t - center))
        assert mask[idx]


def test_two_pass_detrend_preserves_transit_depth() -> None:
    t, in_tr, prep = _make_prepared_curve(period_var=5.0, transit_depth=0.015)

    # Pass 1: unmasked detrend
    dual_p1 = two_pass_detrend(prep, baseline_window_days=0.75, device="cpu")
    assert dual_p1.pass_number == 1
    assert not np.any(dual_p1.in_transit_mask)

    # Pass 2: masked detrend with candidate event
    candidate_events = ((2.0, 3.0, 3.0),)
    dual_p2 = two_pass_detrend(prep, candidate_events=candidate_events, device="cpu")
    assert dual_p2.pass_number == 2
    assert np.any(dual_p2.in_transit_mask)

    # The recovered transit depth in Pass 2 should be closer to true 0.015 than in unmasked Pass 1
    depth_p1 = 1.0 - float(np.mean(dual_p1.detrended_flux[in_tr]))
    depth_p2 = 1.0 - float(np.mean(dual_p2.detrended_flux[in_tr]))

    # Pass 2 masked detrend should recover >= 90% of injected depth
    assert depth_p2 >= 0.0135
    # Pass 2 recovery is strictly better or equal to unmasked Pass 1
    assert depth_p2 >= depth_p1


def test_cpu_and_cuda_parity_if_available() -> None:
    if not has_cuda():
        return

    t, _, prep = _make_prepared_curve()
    flat_cpu, trend_cpu = detrend_biweight(
        prep.time_days, prep.flux_normalized, prep.segment_slices, window_days=0.75, device="cpu"
    )
    flat_cuda, trend_cuda = detrend_biweight(
        prep.time_days, prep.flux_normalized, prep.segment_slices, window_days=0.75, device="cuda"
    )

    # Assert close agreement between CPU and CUDA paths
    np.testing.assert_allclose(trend_cuda, trend_cpu, rtol=1e-2, atol=1e-3)
    np.testing.assert_allclose(flat_cuda, flat_cpu, rtol=1e-2, atol=1e-3)
