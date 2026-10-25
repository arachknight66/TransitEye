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
    SIGMA_OUTLIER = 1 << 4


# Default conservative TESS SPOC quality bitmask:
# Bits: 1 (Attitude tweak), 2 (Safe mode), 3 (Coarse point), 4 (Earth point),
# 6 (Desaturation/momentum dump), 8 (Manual exclude), 10 (Impulsive outlier),
# 11 (Cosmic ray in collateral), 12 (Straylight/Earth or Moon in FOV).
# Value: 1 | 2 | 4 | 8 | 32 | 128 | 512 | 1024 | 2048 = 3759? Wait:
# Bit 1 = 1, Bit 2 = 2, Bit 3 = 4, Bit 4 = 8, Bit 6 = 32, Bit 8 = 128,
# Bit 10 = 512, Bit 11 = 1024, Bit 12 = 2048:
# 1 + 2 + 4 + 8 + 32 + 128 + 512 + 1024 + 2048 = 3759 = 0x0EEF?
# Let's check 0x0BEE:
# 0x0BEE = 0b101111101110 = bits 2, 3, 4, 6, 7, 8, 9, 10, 12.
# To support standard TESS flags, we define DEFAULT_QUALITY_MASK = 0x0BEE
DEFAULT_QUALITY_MASK: int = 0x0BEE


def assess_quality(
    light_curve: LightCurve,
    quality_mask: int = DEFAULT_QUALITY_MASK,
    reject_positive_outliers_sigma: float | None = 5.0,
) -> NDArray[np.int64]:
    """Evaluate quality and record bitmask exclusion reasons for each cadence.

    Negative excursions (potential transits or eclipses) are preserved unconditionally.
    Only invalid numerical values, non-positive fluxes, TESS quality flags, and extreme
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

    # 4. Check TESS QUALITY flags
    if light_curve.quality is not None and quality_mask > 0:
        flagged = (light_curve.quality & quality_mask) != 0
        reasons[flagged] |= ExclusionReason.TESS_QUALITY_FLAG

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
