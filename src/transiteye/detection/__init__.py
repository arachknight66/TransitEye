"""Detection module: BLS periodograms, Lomb-Scargle variability, and iterative candidate search."""

from transiteye.detection.bls import BlsPeriodogram, run_bls
from transiteye.detection.peaks import DetectionResult, find_candidates
from transiteye.detection.variability import (
    VariabilitySummary,
    compute_harmonic_distance,
    run_variability_search,
)

__all__ = [
    "BlsPeriodogram",
    "DetectionResult",
    "VariabilitySummary",
    "compute_harmonic_distance",
    "find_candidates",
    "run_bls",
    "run_variability_search",
]
