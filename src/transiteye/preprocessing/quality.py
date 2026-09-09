"""Documented conservative SPOC quality masking."""

from __future__ import annotations

import numpy as np

# MVP policy: reject only non-zero SPOC QUALITY; preserve bit values for later policy refinement.
QUALITY_POLICY_VERSION = "spoc-nonzero-v1"


def quality_mask(quality: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return accepted mask and immutable explanatory reason codes."""
    accepted = quality == 0
    reasons = np.where(accepted, "accepted", "spoc_quality_nonzero")
    return accepted, reasons
