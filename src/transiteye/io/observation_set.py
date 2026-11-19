"""Lossless, mission-aware input record for future combined observation analysis."""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from transiteye.domain.models import LightCurve
from transiteye.io.fits import (
    read_kepler_light_curve,
    read_tess_light_curve,
    validate_day_time_axis,
)
from transiteye.io.lookup import ArchiveProduct, TargetReference, verify_downloaded_product
from transiteye.preprocessing.quality import quality_mask_for_mission


@dataclass(frozen=True, slots=True)
class Observation:
    source: LightCurve
    time_bjd_tdb: NDArray[np.float64]
    archive_uri: str
    provider: str
    cadence_days: float | None
    cadence_source: str
    overlap_mask: NDArray[np.bool_]
    analysis_mask: NDArray[np.bool_]
    selection_reason: tuple[str, ...]
    overlap_winner_checksum: tuple[str | None, ...]

    @property
    def selected_count(self) -> int:
        return int(np.count_nonzero(self.analysis_mask))

    @property
    def excluded_count(self) -> int:
        return len(self.analysis_mask) - self.selected_count


@dataclass(frozen=True, slots=True)
class ObservationSet:
    mission: str
    target_id: str
    observations: tuple[Observation, ...]
    input_checksums: tuple[str, ...]


def build_observation_set(
    light_curves: tuple[LightCurve, ...], *, archive_uris: tuple[str, ...] | None = None,
    providers: tuple[str, ...] | None = None,
) -> ObservationSet:
    """Retain every product and select one valid cadence in overlapping sample windows.

    TIMEDEL can describe time resolution rather than sample spacing. Three or
    more distinct finite timestamps establish the lower median of positive
    spacings, which avoids mistaking a large gap for a cadence. With exactly
    two distinct timestamps, TIMEDEL is used only if within 20% of their
    spacing. A singleton's TIMEDEL cannot establish a sampling width.
    Unknown widths in intersecting product spans remain unresolved.
    The mask is an input selection, not a joint flux fit or quality assessment.
    """
    if not light_curves:
        raise ValueError("At least one observation is required.")
    if archive_uris is not None and len(archive_uris) != len(light_curves):
        raise ValueError("Every observation requires one archive URI.")
    if providers is not None and len(providers) != len(light_curves):
        raise ValueError("Every observation requires one provider.")
    missions = {str(curve.provenance.metadata.get("mission", "")) for curve in light_curves}
    targets = {curve.provenance.target_id for curve in light_curves}
    if len(missions) != 1 or next(iter(missions)) not in ("TESS", "Kepler"):
        raise ValueError("Observations must have one supported mission.")
    if len(targets) != 1 or not next(iter(targets)):
        raise ValueError("Observations must have one known target identity.")
    checksums = tuple(curve.provenance.checksum_sha256 or "" for curve in light_curves)
    if any(len(checksum) != 64 for checksum in checksums):
        raise ValueError("Every observation requires a SHA-256 input checksum.")
    if len(set(checksums)) != len(checksums):
        raise ValueError("Observations contain a duplicate input product.")
    prepared: list[tuple[LightCurve, NDArray[np.float64], str, str, float | None, str]] = []
    for index, curve in enumerate(light_curves):
        validate_day_time_axis(curve)
        try:
            reference = float(curve.provenance.time_reference or "")
        except ValueError as error:
            raise ValueError("Observation has no valid BJD time reference.") from error
        if not np.isfinite(reference) or not 2_400_000.0 <= reference <= 2_600_000.0:
            raise ValueError("Observation has no valid BJD time reference.")
        absolute = np.asarray(curve.time_days, dtype=np.float64) + reference
        absolute.setflags(write=False)
        uri = curve.provenance.source_uri if archive_uris is None else archive_uris[index]
        provider = (
            str(curve.provenance.metadata.get("provider", ""))
            if providers is None else providers[index]
        )
        cadence, cadence_source = _cadence_width(curve, absolute)
        prepared.append((curve, absolute, uri, provider, cadence, cadence_source))

    masks = [np.zeros(len(item[1]), dtype=np.bool_) for item in prepared]
    overlap_masks = [np.zeros(len(item[1]), dtype=np.bool_) for item in prepared]
    reasons = [["invalid_time" if not np.isfinite(t) else "unresolved"
                for t in item[1]] for item in prepared]
    winners: list[list[str | None]] = [[None] * len(item[1]) for item in prepared]
    unresolved = _unresolved_widths(prepared)
    candidates: list[tuple[int, float, str, str, int, int, float, float]] = []
    mission = next(iter(missions))
    for observation_index, (_curve, absolute, uri, provider, cadence, _) in enumerate(prepared):
        width = 0.0 if cadence is None else cadence
        rank = _provider_rank(mission, provider)
        for cadence_index in np.flatnonzero(np.isfinite(absolute)):
            if unresolved[observation_index][cadence_index]:
                reasons[observation_index][cadence_index] = "overlap_unresolved"
                continue
            time = float(absolute[cadence_index])
            candidates.append((
                rank, width if cadence is not None else float("inf"),
                checksums[observation_index], uri, observation_index,
                int(cadence_index), time - width / 2, time + width / 2,
            ))

    # Stable product identity decides exact ties independently of inventory order.
    candidates.sort()
    quality_mask = quality_mask_for_mission(mission)
    starts: list[float] = []
    ends: list[float] = []
    selected_checksums: list[str] = []
    for _, _, checksum, _, observation_index, cadence_index, start, end in candidates:
        position = bisect_left(starts, start)
        if position > 0 and _overlap(starts[position - 1], ends[position - 1], start, end):
            reasons[observation_index][cadence_index] = "overlap_excluded"
            winners[observation_index][cadence_index] = selected_checksums[position - 1]
            continue
        if position < len(starts) and _overlap(starts[position], ends[position], start, end):
            reasons[observation_index][cadence_index] = "overlap_excluded"
            winners[observation_index][cadence_index] = selected_checksums[position]
            continue
        starts.insert(position, start)
        ends.insert(position, end)
        selected_checksums.insert(position, checksum)
        overlap_masks[observation_index][cadence_index] = True
        curve = prepared[observation_index][0]
        if not np.isfinite(curve.flux[cadence_index]) or curve.flux[cadence_index] <= 0:
            reasons[observation_index][cadence_index] = "invalid_flux"
        elif curve.flux_error is not None and (
            not np.isfinite(curve.flux_error[cadence_index])
            or curve.flux_error[cadence_index] <= 0
        ):
            reasons[observation_index][cadence_index] = "invalid_error"
        elif curve.quality is not None and curve.quality[cadence_index] & quality_mask:
            reasons[observation_index][cadence_index] = "quality_flag"
        else:
            reasons[observation_index][cadence_index] = "selected"
            masks[observation_index][cadence_index] = True

    observations = []
    for (curve, absolute, uri, provider, cadence, cadence_source), overlap_mask, mask, reason, winner in zip(
        prepared, overlap_masks, masks, reasons, winners, strict=True
    ):
        overlap_mask.setflags(write=False)
        mask.setflags(write=False)
        observations.append(Observation(
            curve, absolute, uri, provider, cadence, cadence_source, overlap_mask, mask,
            tuple(reason), tuple(winner),
        ))
    return ObservationSet(mission, str(next(iter(targets))), tuple(observations), checksums)


def _provider_rank(mission: str, provider: str) -> int:
    if mission == "TESS":
        return {"SPOC": 0, "TESS-SPOC": 1}.get(provider.upper(), 2)
    return 0


def _cadence_width(curve: LightCurve, absolute: NDArray[np.float64]) -> tuple[float | None, str]:
    raw = curve.provenance.metadata.get("TIMEDEL") or curve.metadata.get("time_resolution")
    width = float("nan")
    if raw is not None:
        try:
            width = float(raw)
        except (TypeError, ValueError):
            pass
    times = np.unique(absolute[np.isfinite(absolute)])
    if len(times) >= 3:
        # Distinct timestamps remove duplicates; lower median resists a gap in
        # sparse products with one ordinary interval and one missing interval.
        positive = np.diff(times)
        spacing = float(np.sort(positive)[(len(positive) - 1) // 2])
        return spacing, "median_positive_spacing"
    if len(times) == 2:
        spacing = float(times[1] - times[0])
        if np.isfinite(width) and width > 0 and abs(width - spacing) <= 0.2 * spacing:
            return width, "TIMEDEL_spacing_consistent"
        return None, "unknown_two_timestamps"
    if len(times) == 1 and np.isfinite(width) and width > 0:
        return None, "TIMEDEL_singleton_unverified"
    return None, "unknown"


def _overlap(first_start: float, first_end: float, start: float, end: float) -> bool:
    if first_start == first_end:
        return start <= first_start < end or start == end == first_start
    if start == end:
        return first_start <= start < first_end
    return first_start < end and start < first_end


def _unresolved_widths(
    prepared: list[tuple[LightCurve, NDArray[np.float64], str, str, float | None, str]],
) -> list[NDArray[np.bool_]]:
    """Flag intersecting product spans when either sampling width is unknown."""
    flags = [np.zeros(len(item[1]), dtype=np.bool_) for item in prepared]
    for left, (_, left_times, _, _, left_width, _) in enumerate(prepared):
        for right in range(left + 1, len(prepared)):
            right_times = prepared[right][1]
            right_width = prepared[right][4]
            if left_width is not None and right_width is not None:
                continue
            finite_left = left_times[np.isfinite(left_times)]
            finite_right = right_times[np.isfinite(right_times)]
            if not len(finite_left) or not len(finite_right):
                continue
            left_half = 0.0 if left_width is None else left_width / 2
            right_half = 0.0 if right_width is None else right_width / 2
            left_low = float(np.min(finite_left)) - left_half
            left_high = float(np.max(finite_left)) + left_half
            right_low = float(np.min(finite_right)) - right_half
            right_high = float(np.max(finite_right)) + right_half
            if left_low <= right_high and right_low <= left_high:
                flags[left] |= ((left_times + left_half >= right_low)
                                & (left_times - left_half <= right_high))
                flags[right] |= ((right_times + right_half >= left_low)
                                 & (right_times - right_half <= left_high))
    return flags


def load_verified_observation_set(
    target: TargetReference,
    products: tuple[ArchiveProduct, ...],
    downloaded: Mapping[str, Path],
) -> ObservationSet:
    """Load a complete discovered inventory, checking every file against its MAST record."""
    expected = {product.data_uri for product in products}
    if not products or len(expected) != len(products):
        raise ValueError("Observation inventory is empty or contains duplicate products.")
    if set(downloaded) != expected:
        raise ValueError("Observation inventory has missing or unexpected downloaded products.")
    curves: list[LightCurve] = []
    for product in products:
        if product.mission != target.mission or product.target_id != target.canonical_id:
            raise ValueError("Observation product does not match requested target identity.")
        path = downloaded[product.data_uri]
        verify_downloaded_product(product, path)
        curves.append(
            read_tess_light_curve(path)
            if target.mission == "TESS" else read_kepler_light_curve(path)
        )
    return build_observation_set(
        tuple(curves), archive_uris=tuple(product.data_uri for product in products),
        providers=tuple(product.provider for product in products),
    )
