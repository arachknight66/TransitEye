"""Independent per-product preparation for a shared numerical time axis."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from transiteye.io.observation_set import Observation, ObservationSet
from transiteye.preprocessing.detrend import DualLightCurve, two_pass_detrend
from transiteye.preprocessing.quality import quality_mask_for_mission
from transiteye.preprocessing.representation import prepare_light_curve


class IncompleteObservationSetError(ValueError):
    """Combined preparation cannot proceed without a complete selection."""


@dataclass(frozen=True, slots=True)
class PreparedObservation:
    """One original product and its independently prepared selected rows."""

    observation: Observation
    selected_raw_indices: NDArray[np.int64]
    dual: DualLightCurve | None
    status: str


@dataclass(frozen=True, slots=True)
class CombinedPreparation:
    """Audit records plus sorted arrays for numerical search; no search is run."""

    observation_set: ObservationSet
    time_origin_bjd_tdb: float
    observations: tuple[PreparedObservation, ...]
    time_days: NDArray[np.float64]
    flux_normalized: NDArray[np.float64]
    detrended_flux: NDArray[np.float64]
    flux_error_detrended: NDArray[np.float64] | None
    observation_index: NDArray[np.int64]
    input_checksum: NDArray[np.str_]
    raw_cadence_index: NDArray[np.int64]


def prepare_observation_set(
    observation_set: ObservationSet,
    *,
    gap_threshold_hours: float = 6.0,
    reject_positive_outliers_sigma: float | None = 5.0,
    baseline_window_days: float = 0.75,
    device: str = "cpu",
) -> CombinedPreparation:
    """Prepare selected rows independently and combine only their numerical arrays.

    An observation with no selected rows is retained with ``status='no_usable_data'``.
    A product whose selected rows all fail quality assessment raises an explicit
    incomplete error. Missing per-product errors become NaN in the combined
    error array when other products have errors; consumers must handle them.
    """
    if not observation_set.observations:
        raise IncompleteObservationSetError("Observation set has no products.")
    unresolved = [
        observation_set.input_checksums[index]
        for index, observation in enumerate(observation_set.observations)
        if "overlap_unresolved" in observation.selection_reason
    ]
    if unresolved:
        raise IncompleteObservationSetError(
            f"Unresolved overlap makes combined preparation incomplete: {unresolved}"
        )
    selected = [
        np.flatnonzero(observation.analysis_mask)
        for observation in observation_set.observations
    ]
    if not any(len(indices) for indices in selected):
        raise IncompleteObservationSetError("No usable selected cadences in observation set.")
    origin = min(
        float(np.min(observation.time_bjd_tdb[indices]))
        for observation, indices in zip(observation_set.observations, selected, strict=True)
        if len(indices)
    )
    quality_mask = quality_mask_for_mission(observation_set.mission)
    records: list[PreparedObservation] = []
    times: list[NDArray[np.float64]] = []
    normalized: list[NDArray[np.float64]] = []
    detrended: list[NDArray[np.float64]] = []
    errors: list[NDArray[np.float64] | None] = []
    identities: list[NDArray[np.int64]] = []
    checksums: list[NDArray[np.str_]] = []
    raw_indices: list[NDArray[np.int64]] = []
    for index, (observation, indices) in enumerate(
        zip(observation_set.observations, selected, strict=True)
    ):
        # The detrending and gap logic expect ascending time within each product.
        raw = np.asarray(
            indices[np.argsort(observation.time_bjd_tdb[indices], kind="stable")],
            dtype=np.int64,
        )
        raw.setflags(write=False)
        if not len(raw):
            records.append(PreparedObservation(observation, raw, None, "no_usable_data"))
            continue
        source = observation.source
        subset = replace(
            source,
            time_days=observation.time_bjd_tdb[raw] - origin,
            flux=source.flux[raw],
            flux_error=source.flux_error[raw] if source.flux_error is not None else None,
            quality=source.quality[raw] if source.quality is not None else None,
            cadence_number=(source.cadence_number[raw]
                            if source.cadence_number is not None else None),
            centroid_column=(source.centroid_column[raw]
                             if source.centroid_column is not None else None),
            centroid_row=(source.centroid_row[raw]
                          if source.centroid_row is not None else None),
        )
        try:
            prepared = prepare_light_curve(
                subset,
                quality_mask=quality_mask,
                gap_threshold_hours=gap_threshold_hours,
                reject_positive_outliers_sigma=reject_positive_outliers_sigma,
            )
            dual = two_pass_detrend(
                prepared, baseline_window_days=baseline_window_days, device=device
            )
        except Exception as error:
            checksum = observation_set.input_checksums[index]
            raise IncompleteObservationSetError(
                f"Observation {checksum} failed combined preparation: {error}"
            ) from error
        kept = raw[prepared.raw_cadence_mask]
        if not np.all(np.isfinite(dual.detrended_flux)) or not np.all(
            np.isfinite(dual.trend_flux) & (dual.trend_flux > 0)
        ):
            raise IncompleteObservationSetError(
                f"Observation {observation_set.input_checksums[index]} has invalid detrending."
            )
        records.append(PreparedObservation(observation, raw, dual, "prepared"))
        times.append(prepared.time_days)
        normalized.append(prepared.flux_normalized)
        detrended.append(dual.detrended_flux)
        errors.append(
            prepared.flux_error_normalized / dual.trend_flux
            if prepared.flux_error_normalized is not None else None
        )
        identities.append(np.full(len(kept), index, dtype=np.int64))
        checksums.append(np.full(len(kept), observation_set.input_checksums[index]))
        raw_indices.append(kept)
    time = np.concatenate(times)
    order = np.argsort(time, kind="stable")
    time = time[order]
    if not np.all(np.isfinite(time)) or np.any(np.diff(time) <= 0):
        raise IncompleteObservationSetError(
            "Selected combined times must be finite, strictly increasing, and unique."
        )
    combined_error = None
    if any(error is not None for error in errors):
        combined_error = np.concatenate([
            error if error is not None else np.full(len(chunk), np.nan)
            for error, chunk in zip(errors, times, strict=True)
        ])[order]
    return CombinedPreparation(
        observation_set=observation_set,
        time_origin_bjd_tdb=origin,
        observations=tuple(records),
        time_days=time,
        flux_normalized=np.concatenate(normalized)[order],
        detrended_flux=np.concatenate(detrended)[order],
        flux_error_detrended=combined_error,
        observation_index=np.concatenate(identities)[order],
        input_checksum=np.concatenate(checksums)[order],
        raw_cadence_index=np.concatenate(raw_indices)[order],
    )
