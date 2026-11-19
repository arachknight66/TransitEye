"""Integration checks for the bounded search and joint fit of one source."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from numpy.typing import NDArray

from transiteye.domain.models import LightCurve, Provenance
from transiteye.execution.combined import CombinedSearchSettings, analyze_observation_set
from transiteye.io.observation_set import build_observation_set

_PERIOD = 2.0
_EPOCH = 0.5
_DURATION = 0.125
_REFERENCE = 2457000.0
_CADENCE = 1 / 64
_SETTINGS = CombinedSearchSettings(1.8, 2.2, 150)


def _curve(
    observation: int, time: NDArray[np.float64], depth: float,
    *, exposure: float = 0.008, baseline: float = 100.0, with_error: bool = True,
) -> LightCurve:
    # Integrate each piecewise linear edge using an independent clipped-ramp primitive.
    phase = ((time - _EPOCH + _PERIOD / 2) % _PERIOD) - _PERIOD / 2
    ingress = _DURATION / 4
    contacts = np.array([-_DURATION / 2, -_DURATION / 2 + ingress,
                         _DURATION / 2 - ingress, _DURATION / 2])

    def primitive(phase_days: NDArray[np.float64]) -> NDArray[np.float64]:
        ramps = np.maximum(phase_days[:, None] - contacts, 0.0) ** 2
        return ramps @ np.array([1.0, -1.0, -1.0, 1.0]) / (2 * ingress)

    dip = (primitive(phase + exposure / 2) - primitive(phase - exposure / 2)) / exposure
    return LightCurve(
        time_days=time, flux=baseline * (1.0 - depth * np.clip(dip, 0.0, 1.0)),
        flux_error=np.full_like(time, 0.001 * baseline) if with_error else None,
        quality=None, cadence_number=None, centroid_column=None, centroid_row=None,
        flux_kind="PDCSAP_FLUX", flux_unit="e-/s",
        provenance=Provenance(
            source_uri=f"mast:fixture-{observation}", source_path=None,
            checksum_sha256=f"{observation:064x}", product_type="light_curve",
            target_id="42", sector=observation, time_system="TDB",
            time_reference=str(_REFERENCE),
            metadata={"mission": "TESS", "TIMEDEL": str(_CADENCE),
                      "INT_TIME": str(exposure * 86400 / 10), "NUM_FRM": "10"},
        ),
        metadata={"time_unit": "d"},
    )


@pytest.fixture
def observations() -> tuple[LightCurve, LightCurve]:
    # Sample different phases of the shared event in the two observing windows.
    phase_offset = 1 / 256
    return (
        _curve(1, np.arange(0, 3, _CADENCE), 0.012, exposure=0.008),
        _curve(2, np.arange(6, 9, _CADENCE) + phase_offset, 0.025, exposure=0.010, baseline=1300.0),
    )


def test_combined_search_and_fit_recover_synthetic_event(
    observations: tuple[LightCurve, LightCurve],
) -> None:
    result = analyze_observation_set(build_observation_set(observations), _SETTINGS)

    assert result.status == "fitted", result.detail
    assert result.preliminary is not None
    assert result.preliminary.detection_snr >= _SETTINGS.min_bls_snr
    # The manufactured signal above has P=2 d, duration=0.125 d and absolute T0=2457000.5.
    assert result.preliminary.period_days == pytest.approx(_PERIOD, abs=0.03)
    assert result.fitted_period_days == pytest.approx(_PERIOD, abs=0.003)
    assert result.fitted_epoch_bjd_tdb is not None
    phase_error = (result.fitted_epoch_bjd_tdb - _REFERENCE - _EPOCH + 1) % _PERIOD - 1
    assert phase_error == pytest.approx(0.0, abs=0.003)
    assert result.fitted_duration_days == pytest.approx(_DURATION, abs=0.006)
    assert [row.status for row in result.observations] == ["fitted", "fitted"]
    # Distinct manufactured fractional depths and frame-integration times above.
    assert [row.depth for row in result.observations] == pytest.approx([0.012, 0.025], abs=0.001)
    assert [row.exposure_days for row in result.observations] == pytest.approx([0.008, 0.010])
    assert all(row.exposure_status == "frame_sum_boxcar_approximation" for row in result.observations)


def test_combined_budget_exceeded_keeps_all_products(
    observations: tuple[LightCurve, LightCurve],
) -> None:
    source = build_observation_set(observations)
    result = analyze_observation_set(source, replace(_SETTINGS, max_trial_periods=1))

    assert result.status == "budget_exceeded"
    assert "max_trial_periods=1" in result.detail
    assert result.preliminary is None
    assert tuple(row.input_checksum for row in result.observations) == source.input_checksums
    assert all(row.depth is None for row in result.observations)


@pytest.mark.parametrize("integration, exposure_status", [(None, "missing"), ("0", "invalid")])
def test_combined_missing_exposure_retains_preliminary_and_product_statuses(
    observations: tuple[LightCurve, LightCurve], integration: str | None, exposure_status: str,
) -> None:
    first, second = observations
    metadata = dict(first.provenance.metadata)
    if integration is None:
        del metadata["INT_TIME"]
    else:
        metadata["INT_TIME"] = integration
    first = replace(first, provenance=replace(first.provenance, metadata=metadata))
    source = build_observation_set((first, second))
    result = analyze_observation_set(source, _SETTINGS)

    assert result.status == "exposure_missing"
    assert result.preliminary is not None
    assert result.fitted_period_days is None
    assert tuple(row.input_checksum for row in result.observations) == source.input_checksums
    assert result.observations[0].exposure_status == exposure_status
    assert result.observations[0].exposure_days is None
    # The second fixture has a frame-derived integration duration of 0.010 d.
    assert result.observations[1].exposure_days == pytest.approx(0.010)


def test_combined_flat_light_curves_fail_the_screen(
    observations: tuple[LightCurve, LightCurve],
) -> None:
    curves = tuple(replace(curve, flux=np.ones_like(curve.flux)) for curve in observations)
    result = analyze_observation_set(build_observation_set(curves), _SETTINGS)

    assert result.status == "no_significant_signal"
    assert result.preliminary is not None
    assert result.preliminary.detection_snr < _SETTINGS.min_bls_snr
    assert result.fitted_period_days is None
    assert all(row.depth is None for row in result.observations)


def test_combined_unresolved_overlap_is_retained_as_a_failure(
    observations: tuple[LightCurve, LightCurve],
) -> None:
    first, _ = observations
    singleton = _curve(2, np.array([0.5]), 0.0)
    source = build_observation_set((first, singleton))
    result = analyze_observation_set(source, _SETTINGS)

    assert result.status == "preparation_failed"
    assert "Unresolved overlap" in result.detail
    assert result.preliminary is None
    assert tuple(row.input_checksum for row in result.observations) == source.input_checksums


def test_combined_mixed_measured_and_missing_errors_skip_search(
    observations: tuple[LightCurve, LightCurve],
) -> None:
    first, second = observations
    source = build_observation_set((first, replace(second, flux_error=None)))
    result = analyze_observation_set(source, _SETTINGS)

    assert result.status == "incomplete_errors"
    assert result.preliminary is None
    assert result.fitted_period_days is None
    assert tuple(row.input_checksum for row in result.observations) == source.input_checksums


def test_combined_fit_validation_failure_keeps_preliminary_signal(
    observations: tuple[LightCurve, LightCurve],
) -> None:
    first, second = observations
    # 10 x 180 s exceeds the fixture's sampling interval of 1/64 d.
    metadata = {**first.provenance.metadata, "INT_TIME": "180"}
    first = replace(first, provenance=replace(first.provenance, metadata=metadata))
    source = build_observation_set((first, second))
    result = analyze_observation_set(source, _SETTINGS)

    assert result.status == "joint_fit_failed"
    assert "exposure intervals overlap" in result.detail
    assert result.preliminary is not None
    assert result.fitted_period_days is None
    assert tuple(row.input_checksum for row in result.observations) == source.input_checksums
    assert all(row.status == "not_fitted" and row.depth is None for row in result.observations)


def test_combined_rows_keep_input_order_and_observations_without_an_event(
    observations: tuple[LightCurve, LightCurve],
) -> None:
    first, second = observations
    off_event = _curve(3, np.arange(10.875, 11.75, _CADENCE), 0.015)
    source = build_observation_set((second, off_event, first))
    result = analyze_observation_set(source, _SETTINGS)

    assert result.status == "fitted", result.detail
    assert tuple(row.input_checksum for row in result.observations) == source.input_checksums
    assert [row.status for row in result.observations] == ["fitted", "no_event", "fitted"]
    assert result.observations[1].depth is None
    # Depth stays associated with the product's own manufactured dip even in nonchronological order.
    assert result.observations[0].depth == pytest.approx(0.025, abs=0.001)
    assert result.observations[2].depth == pytest.approx(0.012, abs=0.001)
