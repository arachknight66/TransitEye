"""Preprocessing pipeline: quality assessment, gap segmentation, and detrending."""

from transiteye.preprocessing.detrend import (
    DualLightCurve,
    build_in_transit_mask,
    detrend_biweight,
    two_pass_detrend,
)
from transiteye.preprocessing.quality import DEFAULT_QUALITY_MASK, ExclusionReason, assess_quality
from transiteye.preprocessing.representation import (
    PreparedLightCurve,
    compute_cdpp_proxy,
    compute_robust_ptp_scatter,
    compute_segment_slices,
    prepare_light_curve,
)

__all__ = [
    "DEFAULT_QUALITY_MASK",
    "DualLightCurve",
    "ExclusionReason",
    "PreparedLightCurve",
    "assess_quality",
    "build_in_transit_mask",
    "compute_cdpp_proxy",
    "compute_robust_ptp_scatter",
    "compute_segment_slices",
    "detrend_biweight",
    "prepare_light_curve",
    "two_pass_detrend",
]

