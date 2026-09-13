"""Deterministic regularized Extreme Learning Machine classifier."""

from __future__ import annotations

import numpy as np


class ExtremeLearningMachine:
    """Single random tanh layer with a closed-form ridge output solution."""

    def __init__(self, hidden_units: int, regularization: float, random_state: int) -> None:
        if hidden_units <= 0 or regularization <= 0:
            raise ValueError("ELM hidden units and regularization must be positive.")
        self.hidden_units = hidden_units
        self.regularization = regularization
        self.random_state = random_state
        self.input_weights: np.ndarray | None = None
        self.biases: np.ndarray | None = None
        self.output_weights: np.ndarray | None = None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> ExtremeLearningMachine:
        if features.ndim != 2 or labels.ndim != 1 or len(features) != len(labels):
            raise ValueError("ELM inputs must be a two-dimensional matrix and aligned labels.")
        if set(np.unique(labels)) != {0, 1}:
            raise ValueError("ELM training requires both binary classes.")
        generator = np.random.default_rng(self.random_state)
        self.input_weights = generator.normal(
            0, 1 / np.sqrt(features.shape[1]), size=(features.shape[1], self.hidden_units)
        )
        self.biases = generator.normal(0, 1, size=self.hidden_units)
        hidden = np.tanh(features @ self.input_weights + self.biases)
        target = labels.astype(float) * 2 - 1
        gram = hidden.T @ hidden + self.regularization * np.eye(self.hidden_units)
        self.output_weights = np.linalg.solve(gram, hidden.T @ target)
        return self

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        if self.input_weights is None or self.biases is None or self.output_weights is None:
            raise ValueError("ELM has not been fitted.")
        hidden = np.tanh(features @ self.input_weights + self.biases)
        return np.asarray(hidden @ self.output_weights, dtype=float)
