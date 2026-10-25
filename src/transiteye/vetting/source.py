"""Two-tier independent spatial source vetting pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from transiteye.characterization.fitting import RefinedCandidate
from transiteye.domain.models import SourceAssessment
from transiteye.features.extractors import fold_phase
from transiteye.io.tpf import TargetPixelData
from transiteye.preprocessing.detrend import DualLightCurve


@dataclass(frozen=True, slots=True)
class VettingEvidence:
    """Quantitative evidence and diagnostic metrics from source vetting."""

    assessment: SourceAssessment
    tier: int
    centroid_shift_sigma: float
    centroid_shift_pixels: float
    sap_to_pdcsap_depth_ratio: float
    crowdsap: float
    difference_image_offset_pixels: float | None
    difference_image_offset_sigma: float | None
    aperture_depth_growth_ratio: float | None
    summary_reason: str
    details: dict[str, Any]


def run_tier1_source_vetting(
    dual_light_curve: DualLightCurve,
    candidate: RefinedCandidate,
) -> VettingEvidence:
    """Run Tier 1 source vetting using delivered light-curve columns and headers."""
    prepared = dual_light_curve.prepared
    time_days = prepared.time_days
    detrended_flux = dual_light_curve.detrended_flux
    flux_delivered = prepared.flux_normalized

    period_days = max(1e-4, float(candidate.period_days))
    duration_days = float(candidate.duration_hours) / 24.0
    phases = fold_phase(time_days, float(candidate.epoch_days), period_days)
    half_dur_phase = 0.5 * (duration_days / period_days)
    in_transit_mask = np.abs(phases) <= half_dur_phase
    oot_mask = ~in_transit_mask

    # 1. CROWDSAP value
    crowd_val: Any = prepared.metadata.get("crowdsap")
    try:
        crowdsap = float(np.clip(float(crowd_val), 0.0, 1.0)) if crowd_val is not None else 1.0
    except (ValueError, TypeError):
        crowdsap = 1.0

    # 2. SAP to PDCSAP depth ratio
    if np.sum(in_transit_mask) >= 3:
        d_deliv = max(0.0, 1.0 - float(np.median(flux_delivered[in_transit_mask])))
        d_detrend = max(1e-5, 1.0 - float(np.median(detrended_flux[in_transit_mask])))
        sap_to_pdcsap_depth_ratio = float(d_deliv / d_detrend)
    else:
        sap_to_pdcsap_depth_ratio = 1.0

    # 3. Centroid Shift
    col = prepared.centroid_column
    row = prepared.centroid_row

    dist_px = 0.0
    shift_sigma = 0.0

    if (
        col is not None
        and row is not None
        and np.sum(in_transit_mask) >= 3
        and np.sum(oot_mask) >= 10
    ):
        x_in = col[in_transit_mask]
        y_in = row[in_transit_mask]
        x_out = col[oot_mask]
        y_out = row[oot_mask]

        valid_in = ~np.isnan(x_in) & ~np.isnan(y_in)
        valid_out = ~np.isnan(x_out) & ~np.isnan(y_out)

        if np.sum(valid_in) >= 3 and np.sum(valid_out) >= 10:
            x_in_m, y_in_m = float(np.mean(x_in[valid_in])), float(np.mean(y_in[valid_in]))
            x_out_m, y_out_m = float(np.mean(x_out[valid_out])), float(np.mean(y_out[valid_out]))

            dx = x_in_m - x_out_m
            dy = y_in_m - y_out_m
            dist_px = math.sqrt(dx**2 + dy**2)

            var_dx = float(np.var(x_in[valid_in])) / np.sum(valid_in) + float(
                np.var(x_out[valid_out])
            ) / np.sum(valid_out)
            var_dy = float(np.var(y_in[valid_in])) / np.sum(valid_in) + float(
                np.var(y_out[valid_out])
            ) / np.sum(valid_out)
            denom = math.sqrt(max(1e-12, var_dx + var_dy))

            shift_sigma = float(dist_px / denom)

    # Decision policy
    # Statistically significant centroid shift indicates off-target source
    if shift_sigma >= 3.5 and dist_px >= 0.2:
        assessment = SourceAssessment.BLEND_SUSPECTED
        reason = (
            f"Statistically significant centroid shift during transit: "
            f"{dist_px:.3f} px ({shift_sigma:.1f} sigma)."
        )
    elif crowdsap < 0.85:
        # Crowded field with no pixel evidence to resolve
        assessment = SourceAssessment.UNRESOLVED
        reason = (
            f"Crowded aperture (CROWDSAP={crowdsap:.2f} < 0.85) without pixel evidence; "
            f"potential background blend cannot be ruled out."
        )
    else:
        assessment = SourceAssessment.TARGET_CONSISTENT
        reason = (
            f"Centroid is stable within {dist_px:.3f} px ({shift_sigma:.1f} sigma) "
            f"in an uncrowded field (CROWDSAP={crowdsap:.2f})."
        )

    return VettingEvidence(
        assessment=assessment,
        tier=1,
        centroid_shift_sigma=shift_sigma,
        centroid_shift_pixels=dist_px,
        sap_to_pdcsap_depth_ratio=sap_to_pdcsap_depth_ratio,
        crowdsap=crowdsap,
        difference_image_offset_pixels=None,
        difference_image_offset_sigma=None,
        aperture_depth_growth_ratio=None,
        summary_reason=reason,
        details={"flux_depth_delivered": d_deliv if "d_deliv" in locals() else 0.0},
    )


def run_tier2_source_vetting(
    dual_light_curve: DualLightCurve,
    candidate: RefinedCandidate,
    tpf_data: TargetPixelData,
) -> VettingEvidence:
    """Run Tier 2 source vetting with difference-image centroiding and aperture growth curves."""
    # First get Tier 1 baseline
    tier1 = run_tier1_source_vetting(dual_light_curve, candidate)

    time_days = tpf_data.time_days
    flux_cube = tpf_data.flux_cube  # (N_time, rows, cols)
    n_time, rows, cols = flux_cube.shape

    period_days = max(1e-4, float(candidate.period_days))
    duration_days = float(candidate.duration_hours) / 24.0
    phases = fold_phase(time_days, float(candidate.epoch_days), period_days)
    half_dur_phase = 0.5 * (duration_days / period_days)

    in_transit_mask = np.abs(phases) <= half_dur_phase
    oot_mask = (np.abs(phases) > (1.2 * half_dur_phase)) & (np.abs(phases) <= (3.0 * half_dur_phase))

    if np.sum(in_transit_mask) < 3 or np.sum(oot_mask) < 10:
        # Insufficient pixel frames, fall back to Tier 1
        return VettingEvidence(
            assessment=tier1.assessment,
            tier=1,
            centroid_shift_sigma=tier1.centroid_shift_sigma,
            centroid_shift_pixels=tier1.centroid_shift_pixels,
            sap_to_pdcsap_depth_ratio=tier1.sap_to_pdcsap_depth_ratio,
            crowdsap=tier1.crowdsap,
            difference_image_offset_pixels=None,
            difference_image_offset_sigma=None,
            aperture_depth_growth_ratio=None,
            summary_reason=f"TPF available but in-transit frames insufficient ({np.sum(in_transit_mask)} < 3); fell back to Tier 1.",
            details={},
        )

    # In-transit and out-of-transit mean images
    in_img = np.nanmean(flux_cube[in_transit_mask], axis=0)
    oot_img = np.nanmean(flux_cube[oot_mask], axis=0)
    diff_img = oot_img - in_img  # Positive for transit flux loss

    # Difference image flux centroid
    # Only consider positive flux drops in difference image
    pos_diff = np.maximum(0.0, diff_img)
    total_drop = float(np.sum(pos_diff))

    x_target, y_target = tpf_data.target_pixel_coord

    if total_drop > 1e-6:
        # Meshgrid of pixel coordinates (x=column, y=row)
        y_grid, x_grid = np.mgrid[0:rows, 0:cols]
        x_diff_centroid = float(np.sum(x_grid * pos_diff) / total_drop)
        y_diff_centroid = float(np.sum(y_grid * pos_diff) / total_drop)

        dx = x_diff_centroid - x_target
        dy = y_diff_centroid - y_target
        offset_dist_px = math.sqrt(dx**2 + dy**2)

        # Estimate centroid uncertainty from background pixel variance
        oot_pixel_std = float(np.nanstd(flux_cube[oot_mask])) / math.sqrt(
            float(np.sum(in_transit_mask))
        )
        offset_sigma = float(offset_dist_px / max(0.05, (oot_pixel_std / max(1.0, total_drop))))
    else:
        offset_dist_px = 0.0
        offset_sigma = 0.0

    # Concentric Aperture Growth Test
    # Core aperture: radius <= 1.2 px around target
    # Large aperture: radius <= 2.8 px around target
    y_grid, x_grid = np.mgrid[0:rows, 0:cols]
    r_dist = np.sqrt((x_grid - x_target) ** 2 + (y_grid - y_target) ** 2)

    mask_core = r_dist <= 1.2
    mask_large = r_dist <= 2.8

    # Core flux sum across time
    f_core = np.nansum(flux_cube[:, mask_core], axis=1)
    f_large = np.nansum(flux_cube[:, mask_large], axis=1)

    d_core = 0.0
    d_large = 0.0
    growth_ratio = 1.0

    if len(f_core) > 0 and len(f_large) > 0:
        base_core = np.nanmedian(f_core[oot_mask])
        base_large = np.nanmedian(f_large[oot_mask])
        if base_core > 0 and base_large > 0:
            d_core = max(0.0, 1.0 - float(np.nanmedian(f_core[in_transit_mask]) / base_core))
            d_large = max(0.0, 1.0 - float(np.nanmedian(f_large[in_transit_mask]) / base_large))
            if d_core > 1e-5:
                growth_ratio = float(d_large / d_core)

    # Tier 2 Decision Policy
    if offset_dist_px >= 0.5 and offset_sigma >= 3.0:
        assessment = SourceAssessment.BLEND_SUSPECTED
        reason = (
            f"Difference image centroid is displaced from target by {offset_dist_px:.2f} px "
            f"({offset_sigma:.1f} sigma), confirming an off-target blend."
        )
    elif growth_ratio >= 1.4 and offset_dist_px >= 0.3:
        assessment = SourceAssessment.BLEND_SUSPECTED
        reason = (
            f"Transit depth grows with larger aperture (growth ratio={growth_ratio:.2f}) "
            f"and difference centroid offset={offset_dist_px:.2f} px, indicating companion blend."
        )
    elif offset_dist_px < 0.5 and offset_sigma < 3.0:
        assessment = SourceAssessment.TARGET_CONSISTENT
        reason = (
            f"Difference image centroid coincides with target star within {offset_dist_px:.2f} px "
            f"({offset_sigma:.1f} sigma), confirming on-target origin."
        )
    else:
        assessment = SourceAssessment.UNRESOLVED
        reason = (
            f"Pixel difference offset {offset_dist_px:.2f} px is ambiguous; "
            f"crowdsap={tier1.crowdsap:.2f}."
        )

    return VettingEvidence(
        assessment=assessment,
        tier=2,
        centroid_shift_sigma=tier1.centroid_shift_sigma,
        centroid_shift_pixels=tier1.centroid_shift_pixels,
        sap_to_pdcsap_depth_ratio=tier1.sap_to_pdcsap_depth_ratio,
        crowdsap=tier1.crowdsap,
        difference_image_offset_pixels=offset_dist_px,
        difference_image_offset_sigma=offset_sigma,
        aperture_depth_growth_ratio=growth_ratio,
        summary_reason=reason,
        details={
            "total_drop": total_drop,
            "core_depth": d_core,
            "large_depth": d_large,
            "tpf_stamp": oot_img,
            "tpf_diff": diff_img,
        },
    )


def vet_source(
    dual_light_curve: DualLightCurve,
    candidate: RefinedCandidate,
    tpf_data: TargetPixelData | None = None,
) -> VettingEvidence:
    """Run source vetting on a candidate using Tier 1 (LC) and optional Tier 2 (TPF)."""
    if tpf_data is not None:
        return run_tier2_source_vetting(dual_light_curve, candidate, tpf_data)
    return run_tier1_source_vetting(dual_light_curve, candidate)
