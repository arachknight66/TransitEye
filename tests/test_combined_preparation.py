"""Invariants for per-observation preparation on a common time axis."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from transiteye.domain.models import LightCurve, Provenance
from transiteye.io.observation_set import build_observation_set
from transiteye.preprocessing.combined import (
    IncompleteObservationSetError,
    prepare_observation_set,
)


def _curve(
    times: list[float], flux: list[float], *, checksum: str, observation: int,
    reference: str = "2457000.0", mission: str = "TESS", width: float | None = 1.0,
    error: list[float] | None = None, quality: list[int] | None = None,
) -> LightCurve:
    metadata = {"mission": mission}
    if width is not None:
        metadata["TIMEDEL"] = str(width)
    return LightCurve(
        time_days=np.array(times, dtype=float), flux=np.array(flux, dtype=float),
        flux_error=np.array(error, dtype=float) if error is not None else None,
        quality=np.array(quality, dtype=np.int64) if quality is not None else None,
        cadence_number=None, centroid_column=None, centroid_row=None,
        flux_kind="PDCSAP_FLUX", flux_unit="e-/s",
        provenance=Provenance(
            source_uri=f"mast:{observation}", source_path=None,
            checksum_sha256=checksum, product_type="light_curve", target_id="42",
            sector=observation, time_system="TDB", time_reference=reference,
            metadata=metadata,
        ),
        metadata={"time_unit": "d"},
    )


def test_independent_baselines_and_raw_round_trip() -> None:
    first = _curve([2, 0, 1, 3], [100, 100, 100, 100], checksum="a" * 64, observation=1)
    second = _curve([4, 5, 6, 7], [1000] * 4, checksum="b" * 64, observation=2)
    result = prepare_observation_set(build_observation_set((first, second)))
    # Per-product median normalization: both delivered baselines are exactly unity.
    np.testing.assert_allclose(result.flux_normalized, 1.0)
    np.testing.assert_allclose(result.detrended_flux, 1.0)
    # BJD coordinates come from each product's original reference and raw row.
    for time, product, row, checksum in zip(
        result.time_days, result.observation_index, result.raw_cadence_index,
        result.input_checksum, strict=True,
    ):
        source = result.observations[product].observation.source
        assert time + result.time_origin_bjd_tdb == pytest.approx(
            float(source.provenance.time_reference) + source.time_days[row]
        )
        assert checksum == source.provenance.checksum_sha256
    assert np.all(np.diff(result.time_days) > 0)
    assert result.observations[0].observation.source is first
    assert result.observations[1].observation.source is second


def test_mission_quality_mask_and_mixed_errors() -> None:
    first = _curve(
        [0, 1, 2, 3], [100] * 4, checksum="a" * 64, observation=1,
        mission="Kepler", quality=[128, 65536, 0, 0], error=[2] * 4,
    )
    second = _curve(
        [4, 5, 6, 7], [200] * 4, checksum="b" * 64,
        observation=2, mission="Kepler",
    )
    result = prepare_observation_set(build_observation_set((first, second)))
    assert result.raw_cadence_index[result.observation_index == 0].tolist() == [0, 2, 3]
    assert result.observations[0].observation.selection_reason[1] == "quality_flag"
    assert result.flux_error_detrended is not None
    # First-product error 2 / median flux 100 / unity trend; second lacks errors.
    np.testing.assert_allclose(result.flux_error_detrended[:3], 0.02)
    assert np.isnan(result.flux_error_detrended[3:]).all()


def test_other_product_cannot_change_first_product_trend() -> None:
    times = [index / 24 for index in range(24)]
    flux = [100 + 0.5 * index + (1 if index % 3 == 0 else 0) for index in range(24)]
    first = _curve(times, flux, checksum="a" * 64, observation=1, width=1 / 24)
    second = _curve(
        [10 + time for time in times], [1000] * 24,
        checksum="b" * 64, observation=2, width=1 / 24,
    )
    changed_second = replace(second, flux=np.array([2000.0 + 3 * index for index in range(24)]))
    before = prepare_observation_set(build_observation_set((first, second)))
    after = prepare_observation_set(build_observation_set((first, changed_second)))
    # Separability invariant: changing one product cannot change another product's trend.
    np.testing.assert_allclose(
        before.observations[0].dual.trend_flux,
        after.observations[0].dual.trend_flux,
    )
    np.testing.assert_allclose(
        before.observations[0].dual.detrended_flux,
        after.observations[0].dual.detrended_flux,
    )
    assert not np.allclose(
        before.observations[1].dual.prepared.flux_normalized,
        after.observations[1].dual.prepared.flux_normalized,
    )


def test_overlap_exclusion_is_retained_as_empty_observation() -> None:
    first = _curve([0, 1, 2, 3], [100] * 4, checksum="a" * 64, observation=1)
    second = _curve([1, 2], [200] * 2, checksum="b" * 64, observation=2)
    result = prepare_observation_set(build_observation_set((first, second)))
    assert len(result.observations) == 2
    assert result.observations[1].status == "no_usable_data"
    assert result.observations[1].dual is None
    assert result.observations[1].observation.source is second


def test_unresolved_overlap_and_no_usable_data_are_explicit() -> None:
    first = _curve([0], [100], checksum="a" * 64, observation=1, width=None)
    second = _curve([0], [200], checksum="b" * 64, observation=2)
    with pytest.raises(IncompleteObservationSetError, match="Unresolved overlap"):
        prepare_observation_set(build_observation_set((first, second)))
    invalid = replace(first, flux=np.array([np.nan]))
    with pytest.raises(IncompleteObservationSetError, match="No usable selected"):
        prepare_observation_set(build_observation_set((invalid,)))


def test_duplicate_selected_times_fail_explicitly() -> None:
    curve = _curve([0, 0, 1, 2], [100] * 4, checksum="a" * 64, observation=1)
    selected = build_observation_set((curve,))
    # A manually altered selection must still be rejected at this API boundary.
    observation = replace(selected.observations[0], analysis_mask=np.ones(4, dtype=bool))
    selected = replace(selected, observations=(observation,))
    with pytest.raises(IncompleteObservationSetError, match="strictly increasing"):
        prepare_observation_set(selected)
