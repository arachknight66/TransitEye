"""Documented TESS quality bitmask assessment and cadence filtering."""

from __future__ import annotations

from enum import IntFlag

import numpy as np
from numpy.typing import NDArray

from transiteye.domain.models import LightCurve


class ExclusionReason(IntFlag):
    """Reason a raw cadence was excluded from analysis."""

    VALID = 0
    NAN_OR_INF = 1 << 0
    NON_POSITIVE_FLUX = 1 << 1
    INVALID_ERROR = 1 << 2
    TESS_QUALITY_FLAG = 1 << 3
    QUALITY_FLAG = TESS_QUALITY_FLAG  # Mission-neutral alias; retain saved flag value.
    SIGMA_OUTLIER = 1 << 4


# Historical TESS mask retained for single-file result compatibility.
DEFAULT_QUALITY_MASK: int = 0x0BEE

# Lightkurve's Kepler default: attitude tweak, safe/coarse/Earth pointing,
# desaturation, manual exclusion, detector anomaly, no data, thruster firing.
# Corrected aperture cosmic rays (128) remain eligible.
KEPLER_DEFAULT_QUALITY_MASK: int = (
    1 | 2 | 4 | 8 | 32 | 256 | 16384 | 65536 | 1048576
)


def quality_mask_for_mission(mission: str) -> int:
    """Return the recorded default quality policy for a supported mission."""
    if mission == "TESS":
        return DEFAULT_QUALITY_MASK
    if mission == "Kepler":
        return KEPLER_DEFAULT_QUALITY_MASK
    raise ValueError(f"Unsupported mission for quality policy: {mission}")


def assess_quality(
    light_curve: LightCurve,
    quality_mask: int = DEFAULT_QUALITY_MASK,
    reject_positive_outliers_sigma: float | None = 5.0,
) -> NDArray[np.int64]:
    """Evaluate quality and record bitmask exclusion reasons for each cadence.

    Negative excursions (potential transits or eclipses) are preserved unconditionally.
    Only invalid numerical values, non-positive fluxes, selected quality flags, and extreme
    positive-only spikes (e.g. flares/cosmic rays) receive exclusion reasons.
    """
    length = len(light_curve.time_days)
    reasons = np.zeros(length, dtype=np.int64)

    # 1. Check time validity
    invalid_time = ~np.isfinite(light_curve.time_days)
    reasons[invalid_time] |= ExclusionReason.NAN_OR_INF

    # 2. Check flux validity
    invalid_flux = ~np.isfinite(light_curve.flux)
    reasons[invalid_flux] |= ExclusionReason.NAN_OR_INF

    non_positive = (light_curve.flux <= 0) & ~invalid_flux
    reasons[non_positive] |= ExclusionReason.NON_POSITIVE_FLUX

    # 3. Check flux error validity (if provided)
    if light_curve.flux_error is not None:
        invalid_err = ~np.isfinite(light_curve.flux_error) | (light_curve.flux_error <= 0)
        reasons[invalid_err] |= ExclusionReason.INVALID_ERROR

    # 4. Check the caller's recorded mission-specific QUALITY mask.
    if light_curve.quality is not None and quality_mask > 0:
        flagged = (light_curve.quality & quality_mask) != 0
        reasons[flagged] |= ExclusionReason.QUALITY_FLAG

    # 5. Optional positive outlier rejection (flares, cosmic rays)
    if reject_positive_outliers_sigma is not None:
        valid_so_far = reasons == 0
        if np.count_nonzero(valid_so_far) > 10:
            flux_valid = light_curve.flux[valid_so_far]
            median = float(np.median(flux_valid))
            mad = float(np.median(np.abs(flux_valid - median)))
            sigma = 1.4826 * mad
            if sigma > 0:
                high_threshold = median + reject_positive_outliers_sigma * sigma
                # Only positive outliers are flagged; negative excursions (dips) are never flagged here
                positive_spikes = (light_curve.flux > high_threshold) & valid_so_far
                reasons[positive_spikes] |= ExclusionReason.SIGMA_OUTLIER

    return reasons
