"""Model definitions, metadata serialization, and prediction contracts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from transiteye.domain.models import Classification, MorphologyClass, SourceAssessment
from transiteye.features.registry import FEATURE_MAP, FeatureFamily, validate_feature_vector
from transiteye.provenance import model_fingerprint

# The 4 target morphology classes (UNCERTAIN is reserved for abstentions)
TARGET_MORPHOLOGY_CLASSES: tuple[MorphologyClass, ...] = (
    MorphologyClass.TRANSIT_LIKE,
    MorphologyClass.ECLIPSE_LIKE,
    MorphologyClass.STELLAR_VARIABILITY,
    MorphologyClass.OTHER_ARTIFACT,
)

# Standard features used for morphology classifier (excluding source evidence)
MORPHOLOGY_FEATURE_NAMES: tuple[str, ...] = tuple(
    name
    for name, f_def in FEATURE_MAP.items()
    if f_def.family != FeatureFamily.SOURCE_EVIDENCE
)


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    """Metadata tracking training provenance, hyperparameters, and abstention policies."""

    model_id: str
    estimator_name: str
    feature_names: tuple[str, ...]
    imputation_medians: dict[str, float]
    train_manifest_id: str
    code_version: str
    training_sample_count: int
    synthetic_sample_count: int
    cv_macro_f1: float
    cv_metrics_per_class: dict[str, dict[str, float]]
    tau_conf: float
    tau_margin: float
    feature_min_bounds: dict[str, float]
    feature_max_bounds: dict[str, float]
    schema_version: str = "1"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ModelBundle:
    """A self-contained, reproducible morphology classifier bundle."""

    metadata: ModelMetadata
    estimator: Any
    calibrator: Any | None = None
    artifact_sha256: str | None = None

    def predict(
        self,
        features: Mapping[str, float],
        source_assessment: SourceAssessment = SourceAssessment.NOT_ASSESSED,
    ) -> Classification:
        """Classify a candidate with calibrated confidence and abstention guardrails."""
        # Validate and sanitize features
        sanitized = validate_feature_vector(features)

        # Assemble feature array in the exact trained order
        x_vec = np.zeros(len(self.metadata.feature_names), dtype=np.float64)
        for i, fname in enumerate(self.metadata.feature_names):
            x_vec[i] = sanitized.get(
                fname, self.metadata.imputation_medians.get(fname, 0.0)
            )

        x_2d = x_vec.reshape(1, -1)

        # Check OOD guardrail (extreme values far beyond training distribution)
        is_ood = False
        for fname, val in sanitized.items():
            if fname in self.metadata.feature_min_bounds and fname in self.metadata.feature_max_bounds:
                min_b = self.metadata.feature_min_bounds[fname]
                max_b = self.metadata.feature_max_bounds[fname]
                span = max(1e-5, max_b - min_b)
                if val < min_b - 5.0 * span or val > max_b + 5.0 * span:
                    is_ood = True
                    break

        # Predict calibrated probabilities
        if self.calibrator is not None:
            raw_probs = self.calibrator.predict_proba(x_2d)[0]
        else:
            raw_probs = self.estimator.predict_proba(x_2d)[0]

        # Map probabilities to Target Classes
        classes = list(TARGET_MORPHOLOGY_CLASSES)
        if len(raw_probs) != len(classes):
            # Fall back to uniform if class length mismatches
            raw_probs = np.full(len(classes), 1.0 / len(classes))

        # Normalize probabilities sum to 1.0
        prob_sum = float(np.sum(raw_probs))
        if prob_sum > 0:
            norm_probs = raw_probs / prob_sum
        else:
            norm_probs = np.full(len(classes), 1.0 / len(classes))

        prob_map: dict[MorphologyClass, float] = {
            cls: float(p) for cls, p in zip(classes, norm_probs, strict=True)
        }

        # Abstention decision
        sorted_probs = np.sort(norm_probs)[::-1]
        top_prob = float(sorted_probs[0])
        runner_up_prob = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
        margin = top_prob - runner_up_prob

        abstained = bool(
            is_ood
            or top_prob < self.metadata.tau_conf
            or margin < self.metadata.tau_margin
        )

        if abstained:
            predicted_class = None
        else:
            best_idx = int(np.argmax(norm_probs))
            predicted_class = classes[best_idx]

        return Classification(
            probabilities=prob_map,
            predicted_class=predicted_class,
            source_assessment=source_assessment,
            abstained=abstained,
            model_version=self.metadata.model_id,
        )

    def save(self, bundle_dir: Path) -> Path:
        """Serialize model bundle to directory with JSON metadata and joblib estimator."""
        bundle_dir.mkdir(parents=True, exist_ok=True)
        meta_dict = asdict(self.metadata)

        # Write estimator and calibrator
        est_path = bundle_dir / "estimator.joblib"
        joblib.dump(self.estimator, est_path)

        if self.calibrator is not None:
            cal_path = bundle_dir / "calibrator.joblib"
            joblib.dump(self.calibrator, cal_path)
            meta_dict["has_calibrator"] = True
            meta_dict["calibrator_sha256"] = hashlib.sha256(cal_path.read_bytes()).hexdigest()
        else:
            meta_dict["has_calibrator"] = False

        # Compute estimator checksum
        with open(est_path, "rb") as f:
            meta_dict["estimator_sha256"] = hashlib.sha256(f.read()).hexdigest()

        meta_path = bundle_dir / "model_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_dict, f, indent=2, sort_keys=True)

        return bundle_dir

    @classmethod
    def load(
        cls,
        bundle_dir: Path,
        *,
        expected_calibrator_sha256: str | None = None,
        require_verified_calibrator: bool = False,
    ) -> ModelBundle:
        """Load a trusted local model and verify recorded file checksums before deserialization.

        Checksums establish integrity, not trust: joblib files can execute code and must
        originate from a trusted source. Legacy calibrated bundles need an externally
        recorded calibrator checksum for strict loading.
        """
        meta_path = bundle_dir / "model_metadata.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"Model metadata not found at {meta_path}")

        with open(meta_path, encoding="utf-8") as f:
            meta_dict = json.load(f)

        has_calibrator = meta_dict.pop("has_calibrator", False)
        estimator_sha256 = meta_dict.pop("estimator_sha256", None)
        calibrator_sha256 = meta_dict.pop("calibrator_sha256", None)

        metadata = ModelMetadata(
            model_id=meta_dict["model_id"],
            estimator_name=meta_dict["estimator_name"],
            feature_names=tuple(meta_dict["feature_names"]),
            imputation_medians=meta_dict["imputation_medians"],
            train_manifest_id=meta_dict["train_manifest_id"],
            code_version=meta_dict["code_version"],
            training_sample_count=meta_dict["training_sample_count"],
            synthetic_sample_count=meta_dict["synthetic_sample_count"],
            cv_macro_f1=meta_dict["cv_macro_f1"],
            cv_metrics_per_class=meta_dict["cv_metrics_per_class"],
            tau_conf=meta_dict["tau_conf"],
            tau_margin=meta_dict["tau_margin"],
            feature_min_bounds=meta_dict["feature_min_bounds"],
            feature_max_bounds=meta_dict["feature_max_bounds"],
            schema_version=meta_dict.get("schema_version", "1"),
            extra=meta_dict.get("extra", {}),
        )

        est_path = bundle_dir / "estimator.joblib"
        if not estimator_sha256 or hashlib.sha256(est_path.read_bytes()).hexdigest() != estimator_sha256:
            raise ValueError("Model estimator checksum is missing or does not match.")
        cal_path = bundle_dir / "calibrator.joblib"
        expected_cal = expected_calibrator_sha256 or calibrator_sha256
        if has_calibrator:
            if not cal_path.is_file():
                raise FileNotFoundError(f"Model calibrator not found at {cal_path}")
            if require_verified_calibrator and not expected_cal:
                raise ValueError("A trusted calibrator checksum is required for this legacy model.")
            if expected_cal and hashlib.sha256(cal_path.read_bytes()).hexdigest() != expected_cal:
                raise ValueError("Model calibrator checksum does not match.")
        estimator = joblib.load(est_path)

        calibrator = None
        if has_calibrator:
            calibrator = joblib.load(cal_path)

        return cls(
            metadata=metadata, estimator=estimator, calibrator=calibrator,
            artifact_sha256=model_fingerprint(bundle_dir),
        )
