"""Portable, content-derived identifiers for data lineage."""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from transiteye.serialization import content_hash

_SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")
_SHORT_HASH_PATTERN = re.compile(r"^[a-f0-9]{20,64}$")


def normalize_object_id(value: object) -> str:
    """Normalize a TIC identifier as a portable `tic-<integer>` object ID."""
    if isinstance(value, bool):
        raise ValueError("Object identifiers cannot be booleans.")
    if isinstance(value, int):
        numeric = value
    elif isinstance(value, str):
        normalized = value.strip().lower().replace(" ", "")
        if normalized.startswith("tic-"):
            normalized = normalized[4:]
        elif normalized.startswith("tic"):
            normalized = normalized[3:]
        if not normalized.isdecimal():
            raise ValueError("Object identifier must be a TIC integer or TIC-prefixed integer.")
        numeric = int(normalized)
    else:
        raise TypeError("Object identifier must be a string or integer.")

    if numeric <= 0:
        raise ValueError("Object identifier must be positive.")
    return f"tic-{numeric}"


def validate_sha256(value: str) -> str:
    """Validate and normalize a SHA-256 digest."""
    normalized = value.lower()
    if not _SHA256_PATTERN.fullmatch(normalized):
        raise ValueError("Expected a 64-character hexadecimal SHA-256 digest.")
    return normalized


def validate_short_hash(value: str) -> str:
    """Validate a canonical truncated/full SHA-256 digest."""
    normalized = value.lower()
    if not _SHORT_HASH_PATTERN.fullmatch(normalized):
        raise ValueError("Expected a 20 to 64 character hexadecimal digest.")
    return normalized


def _identifier(prefix: str, payload: dict[str, Any]) -> str:
    return f"{prefix}-{content_hash(payload)}"


def make_observation_id(
    *,
    object_id: str | int,
    sector: int,
    author: str,
    cadence_seconds: float,
    product_id: str,
    checksum_sha256: str,
) -> str:
    """Identify one scientific product independently of local cache location."""
    if sector <= 0:
        raise ValueError("Sector must be positive.")
    if cadence_seconds <= 0:
        raise ValueError("Cadence must be positive.")
    if not author.strip() or not product_id.strip():
        raise ValueError("Author and product ID must be non-empty.")
    return _identifier(
        "obs",
        {
            "author": author.strip(),
            "cadence_seconds": cadence_seconds,
            "checksum_sha256": validate_sha256(checksum_sha256),
            "object_id": normalize_object_id(object_id),
            "product_id": product_id.strip(),
            "sector": sector,
        },
    )


def make_observation_group_id(observation_ids: Sequence[str]) -> str:
    """Identify an ordered, non-empty collection of observations."""
    ordered_ids = list(observation_ids)
    if not ordered_ids or any(not item.strip() for item in ordered_ids):
        raise ValueError("An observation group requires non-empty observation IDs.")
    return _identifier("og", {"observation_ids": ordered_ids})


def make_candidate_id(*, observation_group_id: str, bls_config_hash: str, rank: int) -> str:
    """Identify a ranked BLS candidate from a group and BLS configuration."""
    if not observation_group_id.strip():
        raise ValueError("Observation group ID must be non-empty.")
    if rank < 1:
        raise ValueError("Candidate rank must be at least one.")
    return _identifier(
        "cand",
        {
            "bls_config_hash": validate_short_hash(bls_config_hash),
            "observation_group_id": observation_group_id.strip(),
            "rank": rank,
        },
    )


def make_dataset_version(
    *,
    catalog_snapshot_hash: str,
    raw_manifest_hash: str,
    preprocessing_config_hash: str,
    detection_config_hash: str,
    label_policy_hash: str,
) -> str:
    """Identify the immutable scientific inputs used to build a dataset."""
    return _identifier(
        "dataset",
        {
            "catalog_snapshot_hash": validate_short_hash(catalog_snapshot_hash),
            "detection_config_hash": validate_short_hash(detection_config_hash),
            "label_policy_hash": validate_short_hash(label_policy_hash),
            "preprocessing_config_hash": validate_short_hash(preprocessing_config_hash),
            "raw_manifest_hash": validate_short_hash(raw_manifest_hash),
        },
    )


def make_run_fingerprint(*, config_hash: str, source_revision: str | None, master_seed: int) -> str:
    """Identify equivalent scientific runs without including wall-clock time."""
    if master_seed < 0:
        raise ValueError("Master seed must be non-negative.")
    return _identifier(
        "run",
        {
            "config_hash": validate_short_hash(config_hash),
            "master_seed": master_seed,
            "source_revision": source_revision or "unavailable",
        },
    )


def make_experiment_id(*, run_fingerprint: str, timestamp_utc: datetime) -> str:
    """Identify one execution while retaining the deterministic run fingerprint."""
    if timestamp_utc.tzinfo is None:
        raise ValueError("Experiment timestamps must be timezone-aware.")
    compact_time = timestamp_utc.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"exp-{compact_time}-{run_fingerprint.removeprefix('run-')}"
