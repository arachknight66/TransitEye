"""Portable construction and immutable storage of the pre-BLS demo matrix."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from transiteye.acquisition.checksums import sha256_file
from transiteye.config import DemoSettings
from transiteye.demo.schemas import (
    DemoGoldClass,
    DemoVariantRecord,
    SyntheticEventClass,
    SyntheticEventRecord,
)
from transiteye.provenance import derive_seed
from transiteye.serialization import canonical_json, content_hash


class DemoManifestError(ValueError):
    """Raised when the frozen synthetic variant policy is invalid or conflicts."""


@dataclass(frozen=True)
class DemoVariantManifest:
    manifest_id: str
    policy_hash: str
    variant_manifest_hash: str
    variants: pd.DataFrame
    synthetic_events: pd.DataFrame
    qa_summary: dict[str, Any]


def _gold_class(event_class: str) -> DemoGoldClass:
    if event_class == "planet_like":
        return DemoGoldClass.POSITIVE
    if event_class in {"eclipsing_binary_like", "sinusoidal_variability_like"}:
        return DemoGoldClass.NEGATIVE
    return DemoGoldClass.UNLABELED


def build_variant_manifest(
    base_products: pd.DataFrame,
    *,
    settings: DemoSettings,
    master_seed: int,
) -> DemoVariantManifest:
    """Declare every injected variant and its truth before any BLS operation."""
    required = {
        "object_id",
        "source_observation_id",
        "source_product_id",
        "source_raw_checksum",
        "source_preprocessing_hash",
        "source_processed_checksum",
        "time_min",
    }
    missing = sorted(required.difference(base_products.columns))
    if missing:
        raise DemoManifestError(f"Base products missing demo-lineage columns: {missing}")
    if base_products["source_product_id"].duplicated().any():
        raise DemoManifestError("Base product identities must be unique.")
    policy_hash = content_hash(settings.model_dump(mode="json"))
    variants: list[dict[str, Any]] = []
    truth: list[dict[str, Any]] = []
    for base in base_products.sort_values(
        ["object_id", "source_observation_id"], kind="stable"
    ).to_dict(orient="records"):
        for definition in settings.variants:
            seed = derive_seed(
                master_seed,
                f"{settings.seed_component}:{base['source_raw_checksum']}:{definition.name}",
            )
            rng = np.random.default_rng(seed)
            epoch = None
            if definition.period_days is not None:
                # A fixed interior phase range deliberately avoids cadence-grid alignment.
                epoch = float(base["time_min"]) + float(rng.uniform(0.21, 0.79)) * float(
                    definition.period_days
                )
            variant_payload = {
                "base_observation_id": base["source_observation_id"],
                "base_raw_checksum": base["source_raw_checksum"],
                "demo_policy_hash": policy_hash,
                "definition": definition.model_dump(mode="json"),
                "injection_seed": seed,
                "epoch": epoch,
            }
            variant_id = f"demo-var-{content_hash(variant_payload)}"
            event_id = f"demo-event-{content_hash({'variant_id': variant_id})}"
            gold = _gold_class(definition.event_class)
            variant_record = DemoVariantRecord(
                variant_id=variant_id,
                variant_name=definition.name,
                synthetic_event_id=event_id,
                object_id=str(base["object_id"]),
                source_observation_id=str(base["source_observation_id"]),
                source_product_id=str(base["source_product_id"]),
                source_raw_checksum=str(base["source_raw_checksum"]),
                source_preprocessing_hash=str(base["source_preprocessing_hash"]),
                source_processed_checksum=str(base["source_processed_checksum"]),
                synthetic_event_class=SyntheticEventClass(definition.event_class),
                demo_gold_class=gold,
                difficulty=definition.difficulty,
                injection_seed=seed,
                demo_policy_hash=policy_hash,
                generator_version=settings.generator_version,
            )
            event_record = SyntheticEventRecord(
                synthetic_event_id=event_id,
                variant_id=variant_id,
                object_id=str(base["object_id"]),
                synthetic_event_class=SyntheticEventClass(definition.event_class),
                demo_gold_class=gold,
                injected_period_days=definition.period_days,
                injected_epoch=epoch,
                injected_duration_days=definition.duration_days,
                injected_depth_or_amplitude=definition.depth_or_amplitude,
                injected_secondary_depth=definition.secondary_depth,
                injection_seed=seed,
                difficulty=definition.difficulty,
                variant_name=definition.name,
                generator_version=settings.generator_version,
            )
            variants.append(variant_record.model_dump(mode="json"))
            truth.append(event_record.model_dump(mode="json"))
    variant_frame = (
        pd.DataFrame(variants).sort_values("variant_id", kind="stable").reset_index(drop=True)
    )
    truth_frame = (
        pd.DataFrame(truth).sort_values("synthetic_event_id", kind="stable").reset_index(drop=True)
    )
    # Pandas may materialize optional floats as NaN; JSON round-tripping gives
    # canonical nulls required by the project's strict portable serializer.
    portable_rows = json.loads(variant_frame.to_json(orient="records"))
    truth_rows = json.loads(truth_frame.to_json(orient="records"))
    manifest_hash = content_hash({"variants": portable_rows, "truth": truth_rows})
    manifest_id = f"demo-manifest-{manifest_hash}"
    qa = {
        "source_tics": int(variant_frame["object_id"].nunique()),
        "source_light_curves": int(base_products["source_product_id"].nunique()),
        "source_raw_checksums": int(base_products["source_raw_checksum"].nunique()),
        "demo_variants": len(variant_frame),
        "variants_by_class": variant_frame["synthetic_event_class"].value_counts().to_dict(),
        "variants_by_difficulty": variant_frame["difficulty"].value_counts().to_dict(),
    }
    return DemoVariantManifest(
        manifest_id, policy_hash, manifest_hash, variant_frame, truth_frame, qa
    )


def _write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    partial = path.with_name(f".{path.name}.part")
    frame.to_parquet(partial, engine="pyarrow", index=False)
    os.replace(partial, path)


def freeze_variant_manifest(manifest: DemoVariantManifest, *, root: str | Path) -> Path:
    """Freeze the variant/truth decision and reject conflicting replays."""
    directory = Path(root) / manifest.manifest_id
    if directory.exists():
        loaded = load_variant_manifest(directory)
        if loaded.variant_manifest_hash != manifest.variant_manifest_hash:
            raise DemoManifestError("Existing demo manifest conflicts with reconstructed policy.")
        pd.testing.assert_frame_equal(loaded.variants, manifest.variants, check_dtype=False)
        pd.testing.assert_frame_equal(
            loaded.synthetic_events, manifest.synthetic_events, check_dtype=False
        )
        return directory
    directory.mkdir(parents=True, exist_ok=False)
    try:
        variants_path = directory / "variant_manifest.parquet"
        truth_path = directory / "synthetic_events.parquet"
        _write_parquet_atomic(manifest.variants, variants_path)
        _write_parquet_atomic(manifest.synthetic_events, truth_path)
        metadata = {
            "manifest_id": manifest.manifest_id,
            "demo_policy_hash": manifest.policy_hash,
            "variant_manifest_hash": manifest.variant_manifest_hash,
            "dataset_role": "demo",
            "truth_source": "synthetic_injection",
            "qa_summary": manifest.qa_summary,
            "variant_manifest_sha256": sha256_file(variants_path),
            "synthetic_events_sha256": sha256_file(truth_path),
        }
        (directory / "metadata.json").write_text(canonical_json(metadata) + "\n", encoding="utf-8")
    except Exception:
        for path in directory.iterdir():
            path.unlink()
        directory.rmdir()
        raise
    return directory


def load_variant_manifest(directory: str | Path) -> DemoVariantManifest:
    """Load a frozen demo manifest only after checksum validation."""
    root = Path(directory)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    variant_path = root / "variant_manifest.parquet"
    truth_path = root / "synthetic_events.parquet"
    if sha256_file(variant_path) != metadata["variant_manifest_sha256"]:
        raise DemoManifestError("Demo variant manifest checksum mismatch.")
    if sha256_file(truth_path) != metadata["synthetic_events_sha256"]:
        raise DemoManifestError("Demo truth-table checksum mismatch.")
    return DemoVariantManifest(
        str(metadata["manifest_id"]),
        str(metadata["demo_policy_hash"]),
        str(metadata["variant_manifest_hash"]),
        pd.read_parquet(variant_path),
        pd.read_parquet(truth_path),
        dict(metadata["qa_summary"]),
    )
