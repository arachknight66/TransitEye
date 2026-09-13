"""Deterministic B040 TIC-grouped modeling split."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from transiteye.config import ModelingSettings
from transiteye.provenance import derive_seed
from transiteye.serialization import content_hash


@dataclass(frozen=True)
class ModelingSplit:
    manifest: pd.DataFrame
    split_id: str
    seed: int


def _partition_counts(group_count: int, settings: ModelingSettings) -> tuple[int, int, int]:
    if group_count < 3:
        raise ValueError("At least three TIC groups are required for a three-way split.")
    fractions = np.asarray(
        [
            settings.split.train_fraction,
            settings.split.validation_fraction,
            settings.split.test_fraction,
        ]
    )
    raw = fractions * group_count
    counts = np.floor(raw).astype(int)
    counts[counts == 0] = 1
    while counts.sum() < group_count:
        residual = raw - counts
        counts[int(np.argmax(residual))] += 1
    while counts.sum() > group_count:
        eligible = np.where(counts > 1)[0]
        if not eligible.size:
            raise ValueError("Could not construct non-empty split partitions.")
        index = int(eligible[np.argmax(counts[eligible] - raw[eligible])])
        counts[index] -= 1
    return int(counts[0]), int(counts[1]), int(counts[2])


def make_grouped_split(
    candidates: pd.DataFrame,
    labels: pd.DataFrame,
    lineage: pd.DataFrame,
    *,
    settings: ModelingSettings,
    master_seed: int,
    dataset_version: str,
    feature_version: str,
) -> ModelingSplit:
    """Assign every candidate and raw checksum by TIC, independently of labels."""
    merged = candidates[["candidate_id", "object_id", "observation_group_id"]].merge(
        labels[["candidate_id", "gold_candidate_label"]], on="candidate_id", validate="one_to_one"
    )
    group_lineage = lineage[
        ["observation_group_id", "object_id", "source_raw_checksum"]
    ].drop_duplicates()
    merged = merged.merge(
        group_lineage,
        on=["observation_group_id", "object_id"],
        validate="many_to_one",
    )
    if len(merged) != len(candidates):
        raise ValueError("Not every modeling candidate has raw-product lineage.")
    groups = np.asarray(sorted(merged["object_id"].astype(str).unique()), dtype=object)
    train_count, validation_count, _ = _partition_counts(len(groups), settings)
    seed = derive_seed(master_seed, settings.split.seed_component)
    shuffled = groups[np.random.default_rng(seed).permutation(len(groups))]
    assignments = {
        **{str(value): "train" for value in shuffled[:train_count]},
        **{
            str(value): "validation"
            for value in shuffled[train_count : train_count + validation_count]
        },
        **{str(value): "test" for value in shuffled[train_count + validation_count :]},
    }
    merged["split"] = merged["object_id"].astype(str).map(assignments)
    manifest = (
        merged[
            [
                "candidate_id",
                "object_id",
                "observation_group_id",
                "source_raw_checksum",
                "gold_candidate_label",
                "split",
            ]
        ]
        .sort_values("candidate_id", kind="stable")
        .reset_index(drop=True)
    )
    validate_grouped_split(manifest)
    identity = {
        "dataset_version": dataset_version,
        "feature_version": feature_version,
        "seed": seed,
        "assignments": dict(sorted(assignments.items())),
        "policy_version": settings.policy_version,
        "fractions": {
            "train": settings.split.train_fraction,
            "validation": settings.split.validation_fraction,
            "test": settings.split.test_fraction,
        },
    }
    return ModelingSplit(manifest, f"model-split-{content_hash(identity)}", seed)


def validate_grouped_split(manifest: pd.DataFrame) -> None:
    """Reject TIC, checksum, group, or candidate leakage across partitions."""
    required = {
        "candidate_id",
        "object_id",
        "observation_group_id",
        "source_raw_checksum",
        "split",
    }
    if missing := required.difference(manifest.columns):
        raise ValueError(f"Modeling split lacks columns: {sorted(missing)}")
    if manifest["candidate_id"].duplicated().any():
        raise ValueError("Candidate IDs are duplicated in the modeling split.")
    for column in ("object_id", "observation_group_id", "source_raw_checksum"):
        crossing = manifest.groupby(column, dropna=False)["split"].nunique()
        if (crossing > 1).any():
            raise ValueError(f"{column} crosses modeling partitions.")
    if set(manifest["split"]) != {"train", "validation", "test"}:
        raise ValueError("Modeling train, validation, and test partitions must be non-empty.")
