"""Locate the optional, real TESS sample used by the desktop."""

from __future__ import annotations

import hashlib
from pathlib import Path

from transiteye.config import default_config

SAMPLE_RELATIVE_PATH = Path(
    "data/real_catalog_cohort_v1/"
    "tess2024326142117-s0086-0000000198356533-0283-s_lc.fits"
)
SAMPLE_SHA256 = "1ca8a7b4b42ef1bc89bd955992eea250ef68eda18e19b9c41a84344556eeb576"


def resolve_sample(path: Path | None = None) -> Path:
    """Return an available sample; verify the identity of the built-in choice."""
    candidate = (path if path is not None else default_config().workspace.root / SAMPLE_RELATIVE_PATH)
    candidate = candidate.expanduser().resolve()
    if not candidate.is_file():
        raise FileNotFoundError(
            f"Sample FITS file not found: {candidate}. "
            "Launch with --demo --demo-file PATH to use another local FITS file."
        )
    if path is None:
        with candidate.open("rb") as source:
            checksum = hashlib.file_digest(source, "sha256").hexdigest()
        if checksum != SAMPLE_SHA256:
            raise ValueError(f"Sample FITS checksum does not match the expected input: {candidate}")
    return candidate
