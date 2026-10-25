"""Characterization module: trapezoid/batman modeling, parameter refinement, and uncertainties."""

from transiteye.characterization.fitting import (
    RefinedCandidate,
    compute_phase_coverage,
    evaluate_trapezoid,
    fit_batman_refinement,
    fit_trapezoid,
    refine_candidate,
)
from transiteye.characterization.uncertainty import (
    compute_empirical_significance,
    compute_parameter_intervals,
    compute_red_noise_snr,
)

__all__ = [
    "RefinedCandidate",
    "compute_empirical_significance",
    "compute_parameter_intervals",
    "compute_phase_coverage",
    "compute_red_noise_snr",
    "evaluate_trapezoid",
    "fit_batman_refinement",
    "fit_trapezoid",
    "refine_candidate",
]

