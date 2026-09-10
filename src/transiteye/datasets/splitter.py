"""Deterministic TIC-grouped development and future final splits."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from transiteye.config import DatasetSettings
from transiteye.datasets.schemas import DatasetRole, DatasetSplit
from transiteye.identifiers import normalize_object_id
from transiteye.provenance import derive_seed
from transiteye.serialization import content_hash


class SplitError(ValueError):
    """Raised when a grouped split cannot meet its declared contract."""


def _allocation_counts(size: int, fractions: tuple[float, float, float]) -> tuple[int, int, int]:
    exact = [size * value for value in fractions]
    counts = [math.floor(value) for value in exact]
    remainder = size - sum(counts)
    order = sorted(range(3), key=lambda index: (-(exact[index] - counts[index]), index))
    for index in order[:remainder]:
        counts[index] += 1
    return counts[0], counts[1], counts[2]


def grouped_split(
    object_ids: list[str],
    *,
    settings: DatasetSettings,
    master_seed: int,
    role: DatasetRole,
) -> pd.DataFrame:
    """Assign each normalized TIC once; group integrity has priority over balance."""
    normalized = sorted({normalize_object_id(value) for value in object_ids})
    if not normalized:
        raise SplitError("At least one TIC is required for a split.")
    seed = derive_seed(master_seed, settings.split.seed_component)
    if role is DatasetRole.DEVELOPMENT:
        assignments = [(object_id, DatasetSplit.DEVELOPMENT.value) for object_id in normalized]
    else:
        positive_partitions = sum(
            value > 0
            for value in (
                settings.split.train_fraction,
                settings.split.validation_fraction,
                settings.split.test_fraction,
            )
        )
        if len(normalized) < positive_partitions:
            raise SplitError(
                "Too few TIC groups for the configured final split; use development role."
            )
        rng = np.random.default_rng(seed)
        shuffled = [normalized[index] for index in rng.permutation(len(normalized))]
        counts = _allocation_counts(
            len(shuffled),
            (
                settings.split.train_fraction,
                settings.split.validation_fraction,
                settings.split.test_fraction,
            ),
        )
        labels = (
            [DatasetSplit.TRAIN.value] * counts[0]
            + [DatasetSplit.VALIDATION.value] * counts[1]
            + [DatasetSplit.TEST.value] * counts[2]
        )
        assignments = list(zip(shuffled, labels, strict=True))
    frame = pd.DataFrame(assignments, columns=["object_id", "split"])
    frame["dataset_role"] = role.value
    frame["split_seed"] = seed
    frame["split_config_hash"] = content_hash(settings.split.model_dump(mode="json"))
    return frame.sort_values("object_id", kind="stable").reset_index(drop=True)


def enrich_split_manifest(
    split: pd.DataFrame, candidates: pd.DataFrame, events: pd.DataFrame
) -> pd.DataFrame:
    """Attach deterministic group-level counts without collapsing event truth to a label."""
    result = split.copy(deep=True)
    event_counts = events.groupby("object_id").size()
    candidate_counts = candidates.groupby("object_id").size()
    positives = (
        events.loc[events["internal_event_class"] == "gold_positive"].groupby("object_id").size()
    )
    negatives = (
        events.loc[events["internal_event_class"] == "gold_negative"].groupby("object_id").size()
    )
    result["event_count"] = result["object_id"].map(event_counts).fillna(0).astype(int)
    result["candidate_count"] = result["object_id"].map(candidate_counts).fillna(0).astype(int)
    result["positive_event_count"] = result["object_id"].map(positives).fillna(0).astype(int)
    result["negative_event_count"] = result["object_id"].map(negatives).fillna(0).astype(int)
    result["mixed_gold_event_classes"] = (result["positive_event_count"] > 0) & (
        result["negative_event_count"] > 0
    )
    return result
