"""Train-only deterministic numeric transformation contract."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from transiteye.serialization import content_hash


@dataclass
class TrainOnlyTransformer:
    """Median imputation and optional standardization fitted on training rows only."""

    scale: bool
    input_features: tuple[str, ...] = ()
    retained_features: tuple[str, ...] = ()
    removed_all_null_features: tuple[str, ...] = ()
    medians: np.ndarray | None = None
    means: np.ndarray | None = None
    scales: np.ndarray | None = None

    def fit(self, frame: pd.DataFrame) -> TrainOnlyTransformer:
        if frame.empty:
            raise ValueError("Cannot fit transformations on an empty training table.")
        self.input_features = tuple(str(column) for column in frame.columns)
        numeric = frame.to_numpy(dtype=float)
        all_null = np.isnan(numeric).all(axis=0)
        self.removed_all_null_features = tuple(
            column for column, remove in zip(self.input_features, all_null, strict=True) if remove
        )
        self.retained_features = tuple(
            column
            for column, remove in zip(self.input_features, all_null, strict=True)
            if not remove
        )
        if not self.retained_features:
            raise ValueError("All training features are null.")
        retained = numeric[:, ~all_null]
        self.medians = np.nanmedian(retained, axis=0)
        imputed = np.where(np.isnan(retained), self.medians, retained)
        self.means = np.mean(imputed, axis=0) if self.scale else np.zeros(imputed.shape[1])
        raw_scale = np.std(imputed, axis=0) if self.scale else np.ones(imputed.shape[1])
        self.scales = np.where(raw_scale == 0, 1.0, raw_scale)
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.medians is None or self.means is None or self.scales is None:
            raise ValueError("Transformer has not been fitted.")
        missing = set(self.input_features).difference(frame.columns)
        if missing:
            raise ValueError(f"Inference feature schema is missing columns: {sorted(missing)}")
        retained = frame.loc[:, self.retained_features].to_numpy(dtype=float)
        imputed = np.where(np.isnan(retained), self.medians, retained)
        transformed = (imputed - self.means) / self.scales
        if not np.isfinite(transformed).all():
            raise ValueError("Transformed features contain non-finite values.")
        return np.asarray(transformed, dtype=float)

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        return self.fit(frame).transform(frame)

    def portable_state(self) -> dict[str, object]:
        if self.medians is None or self.means is None or self.scales is None:
            raise ValueError("Transformer has not been fitted.")
        return {
            "policy": "train-median-standardize-v1" if self.scale else "train-median-v1",
            "input_features": list(self.input_features),
            "retained_features": list(self.retained_features),
            "removed_all_null_features": list(self.removed_all_null_features),
            "medians": self.medians.tolist(),
            "means": self.means.tolist(),
            "scales": self.scales.tolist(),
        }

    def state_hash(self) -> str:
        return content_hash(self.portable_state())
