"""Grouped robustness and domain-shift diagnostics for frozen demo models."""

from transiteye.evaluation.builder import EvaluationRunResult, run_robustness_evaluation
from transiteye.evaluation.final_builder import FinalEvaluationRunResult, run_final_evaluation

__all__ = [
    "EvaluationRunResult",
    "FinalEvaluationRunResult",
    "run_final_evaluation",
    "run_robustness_evaluation",
]
