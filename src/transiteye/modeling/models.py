"""B042 baseline model construction and score interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.svm import SVC  # type: ignore[import-untyped]

from transiteye.config import ModelingSettings
from transiteye.modeling.elm import ExtremeLearningMachine
from transiteye.provenance import derive_seed

MODEL_ORDER = ("logistic_regression", "random_forest", "svm_rbf", "elm")


@dataclass(frozen=True)
class TrainedClassifier:
    family: str
    estimator: Any
    seed: int
    hyperparameters: dict[str, Any]
    requires_scaling: bool

    def scores(self, features: np.ndarray) -> np.ndarray:
        if self.family in {"logistic_regression", "random_forest"}:
            return np.asarray(self.estimator.predict_proba(features)[:, 1], dtype=float)
        return np.asarray(self.estimator.decision_function(features), dtype=float)


def fit_classifier(
    family: str,
    features: np.ndarray,
    labels: np.ndarray,
    *,
    settings: ModelingSettings,
    master_seed: int,
) -> TrainedClassifier:
    """Fit one predefined baseline on an already train-only transformed matrix."""
    seed = derive_seed(master_seed, f"model:{family}")
    if family == "logistic_regression":
        parameters = {
            "C": settings.logistic_regression.c,
            "max_iter": settings.logistic_regression.max_iter,
            "solver": "lbfgs",
            "random_state": seed,
        }
        estimator: Any = LogisticRegression(**parameters).fit(features, labels)
        scaled = True
    elif family == "random_forest":
        parameters = {
            "n_estimators": settings.random_forest.n_estimators,
            "max_depth": settings.random_forest.max_depth,
            "min_samples_leaf": settings.random_forest.min_samples_leaf,
            "random_state": seed,
            "n_jobs": 1,
        }
        estimator = RandomForestClassifier(**parameters).fit(features, labels)
        scaled = False
    elif family == "svm_rbf":
        parameters = {
            "C": settings.svm.c,
            "gamma": settings.svm.gamma,
            "kernel": "rbf",
        }
        estimator = SVC(**parameters).fit(features, labels)
        scaled = True
    elif family == "elm":
        parameters = {
            "hidden_units": settings.elm.hidden_units,
            "regularization": settings.elm.regularization,
            "activation": settings.elm.activation,
            "random_state": seed,
        }
        estimator = ExtremeLearningMachine(
            settings.elm.hidden_units, settings.elm.regularization, seed
        ).fit(features, labels)
        scaled = True
    else:
        raise ValueError(f"Unsupported model family: {family}")
    return TrainedClassifier(family, estimator, seed, parameters, scaled)


def model_requires_scaling(family: str) -> bool:
    if family not in MODEL_ORDER:
        raise ValueError(f"Unsupported model family: {family}")
    return family != "random_forest"
