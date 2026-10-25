"""Dual light-curve representation, segmentation, normalization, and scatter metrics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from transiteye.domain.models import LightCurve, Provenance
from transiteye.preprocessing.quality import DEFAULT_QUALITY_MASK, ExclusionReason, assess_quality


@dataclass(frozen=True, slots=True)
class PreparedLightCurve:
    """A quality-filtered, segment-normalized minimally processed light curve.

    Preserves original cadences via `raw_cadence_mask` and `exclusion_flags`.
    Valid cadences are segmented across data gaps and normalized per segment.
    """

    time_days: NDArray[np.float64]
    flux_normalized: NDArray[np.float64]
    flux_error_normalized: NDArray[np.float64] | None
    quality: NDArray[np.int64] | None
    cadence_number: NDArray[np.int64] | None
    centroid_column: NDArray[np.float64] | None
    centroid_row: NDArray[np.float64] | None
    raw_cadence_mask: NDArray[np.bool_]
    exclusion_flags: NDArray[np.int64]
    segment_slices: tuple[tuple[int, int], ...]
    scatter_ptp_ppm: float
    cdpp_1h_ppm: float
    cdpp_2h_ppm: float
    cdpp_6h_ppm: float
    provenance: Provenance
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        length = len(self.time_days)
        if length == 0:
            raise ValueError("PreparedLightCurve must contain at least one valid cadence.")
        if len(self.flux_normalized) != length:
            raise ValueError("flux_normalized length must match time_days.")
        for name in (
            "flux_error_normalized",
            "quality",
            "cadence_number",
            "centroid_column",
            "centroid_row",
        ):
            values = getattr(self, name)
            if values is not None and len(values) != length:
                raise ValueError(f"{name} length must match time_days.")
        if len(self.raw_cadence_mask) != len(self.exclusion_flags):
            raise ValueError("raw_cadence_mask and exclusion_flags must have equal length.")


def compute_segment_slices(
    time_days: NDArray[np.float64],
    gap_threshold_days: float = 0.25,
) -> tuple[tuple[int, int], ...]:
    """Identify contiguous segments separated by data gaps larger than threshold."""
    if len(time_days) == 0:
        return ()
    diffs = np.diff(time_days)
    split_indices = np.where(diffs > gap_threshold_days)[0] + 1
    start_indices = np.concatenate(([0], split_indices))
    end_indices = np.concatenate((split_indices, [len(time_days)]))
    return tuple((int(s), int(e)) for s, e in zip(start_indices, end_indices, strict=True))


def compute_robust_ptp_scatter(
    flux: NDArray[np.float64],
    segment_slices: tuple[tuple[int, int], ...],
) -> float:
    """Compute point-to-point scatter strictly within segments (never crossing gaps)."""
    segment_diffs: list[NDArray[np.float64]] = []
    for start, end in segment_slices:
        if end - start >= 2:
            segment_diffs.append(np.diff(flux[start:end]))
    if not segment_diffs:
        return 0.0
    all_diffs = np.concatenate(segment_diffs)
    if len(all_diffs) == 0:
        return 0.0
    mad = float(np.median(np.abs(all_diffs - np.median(all_diffs))))
    # Scaling factor 1.4826 / sqrt(2) for adjacent difference of i.i.d. Gaussian
    return float(1.4826 / np.sqrt(2.0) * mad * 1e6)


def compute_cdpp_proxy(
    flux: NDArray[np.float64],
    time_days: NDArray[np.float64],
    segment_slices: tuple[tuple[int, int], ...],
    duration_hours: float,
) -> float:
    """Compute robust Combined Differential Photometric Precision (CDPP) in ppm."""
    if len(time_days) < 2:
        return 0.0
    duration_days = duration_hours / 24.0
    median_cadence = float(np.median(np.diff(time_days)))
    if median_cadence <= 0:
        return 0.0
    window_size = max(1, int(round(duration_days / median_cadence)))

    means: list[float] = []
    for start, end in segment_slices:
        seg_flux = flux[start:end]
        if len(seg_flux) >= window_size:
            # Use non-overlapping blocks within the segment
            n_blocks = len(seg_flux) // window_size
            for b in range(n_blocks):
                means.append(float(np.mean(seg_flux[b * window_size : (b + 1) * window_size])))

    if len(means) < 3:
        # Fallback to ptp scatter scaled by 1 / sqrt(window_size)
        ptp = compute_robust_ptp_scatter(flux, segment_slices)
        return float(ptp / np.sqrt(window_size))

    arr = np.array(means)
    mad = float(np.median(np.abs(arr - np.median(arr))))
    return float(1.4826 * mad * 1e6)


def prepare_light_curve(
    light_curve: LightCurve,
    quality_mask: int = DEFAULT_QUALITY_MASK,
    gap_threshold_hours: float = 6.0,
    reject_positive_outliers_sigma: float | None = 5.0,
) -> PreparedLightCurve:
    """Filter invalid cadences, segment across gaps, and normalize flux per segment.

    Preserves full audit trail: exclusion flags are retained on the raw cadence grid.
    """
    exclusion_flags = assess_quality(
        light_curve,
        quality_mask=quality_mask,
        reject_positive_outliers_sigma=reject_positive_outliers_sigma,
    )
    raw_cadence_mask = exclusion_flags == ExclusionReason.VALID
    valid_count = int(np.count_nonzero(raw_cadence_mask))

    if valid_count == 0:
        raise ValueError(
            f"No valid cadences remaining in light curve (target_id={light_curve.provenance.target_id})."
        )

    valid_time = light_curve.time_days[raw_cadence_mask]
    valid_flux = light_curve.flux[raw_cadence_mask]
    valid_err = (
        light_curve.flux_error[raw_cadence_mask] if light_curve.flux_error is not None else None
    )
    valid_quality = light_curve.quality[raw_cadence_mask] if light_curve.quality is not None else None
    valid_cadenceno = (
        light_curve.cadence_number[raw_cadence_mask]
        if light_curve.cadence_number is not None
        else None
    )
    valid_cent_col = (
        light_curve.centroid_column[raw_cadence_mask]
        if light_curve.centroid_column is not None
        else None
    )
    valid_cent_row = (
        light_curve.centroid_row[raw_cadence_mask]
        if light_curve.centroid_row is not None
        else None
    )

    # Compute segments on valid cadences
    gap_threshold_days = gap_threshold_hours / 24.0
    segment_slices = compute_segment_slices(valid_time, gap_threshold_days=gap_threshold_days)

    # Normalize flux per segment to median = 1.0
    flux_norm = np.copy(valid_flux)
    flux_err_norm = np.copy(valid_err) if valid_err is not None else None

    for start, end in segment_slices:
        seg_median = float(np.median(valid_flux[start:end]))
        if seg_median > 0:
            flux_norm[start:end] /= seg_median
            if flux_err_norm is not None:
                flux_err_norm[start:end] /= seg_median

    ptp_scatter = compute_robust_ptp_scatter(flux_norm, segment_slices)
    cdpp_1h = compute_cdpp_proxy(flux_norm, valid_time, segment_slices, duration_hours=1.0)
    cdpp_2h = compute_cdpp_proxy(flux_norm, valid_time, segment_slices, duration_hours=2.0)
    cdpp_6h = compute_cdpp_proxy(flux_norm, valid_time, segment_slices, duration_hours=6.0)

    return PreparedLightCurve(
        time_days=valid_time,
        flux_normalized=flux_norm,
        flux_error_normalized=flux_err_norm,
        quality=valid_quality,
        cadence_number=valid_cadenceno,
        centroid_column=valid_cent_col,
        centroid_row=valid_cent_row,
        raw_cadence_mask=raw_cadence_mask,
        exclusion_flags=exclusion_flags,
        segment_slices=segment_slices,
        scatter_ptp_ppm=ptp_scatter,
        cdpp_1h_ppm=cdpp_1h,
        cdpp_2h_ppm=cdpp_2h,
        cdpp_6h_ppm=cdpp_6h,
        provenance=light_curve.provenance,
        metadata={
            "initial_cadences": len(light_curve.time_days),
            "valid_cadences": valid_count,
            "excluded_cadences": len(light_curve.time_days) - valid_count,
            "segment_count": len(segment_slices),
            "gap_threshold_hours": gap_threshold_hours,
        },
    )
