"""Portable artifact-lineage construction for dataset assembly."""

from __future__ import annotations

import pandas as pd

from transiteye.identifiers import make_candidate_id, make_observation_id, validate_sha256
from transiteye.serialization import content_hash


class LineageError(ValueError):
    """Raised when an artifact edge cannot be established unambiguously."""


def build_artifact_lineage(
    products: pd.DataFrame,
    receipts: pd.DataFrame,
    processed: pd.DataFrame,
    detections: pd.DataFrame,
) -> pd.DataFrame:
    """Join product, raw, processed, and BLS identities without local paths."""
    product_columns = {
        "object_id",
        "mast_obs_id",
        "sector",
        "author",
        "exposure_seconds",
        "product_filename",
        "data_uri",
    }
    receipt_columns = {"object_id", "data_uri", "sha256"}
    processed_columns = {
        "source_raw_checksum",
        "processed_checksum",
        "preprocessing_config_hash",
    }
    detection_columns = {"source_raw_checksum", "observation_group_id", "bls_config_hash", "status"}
    for name, frame, required in (
        ("products", products, product_columns),
        ("receipts", receipts, receipt_columns),
        ("processed", processed, processed_columns),
        ("detections", detections, detection_columns),
    ):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise LineageError(f"{name} missing lineage columns: {missing}")

    joined = receipts[list(receipt_columns)].merge(
        products[list(product_columns)],
        on=["object_id", "data_uri"],
        how="left",
        validate="one_to_one",
    )
    if joined["mast_obs_id"].isna().any():
        raise LineageError("A downloaded product has no unique frozen product record.")
    joined = joined.merge(
        processed,
        left_on="sha256",
        right_on="source_raw_checksum",
        how="left",
        validate="one_to_one",
    ).merge(
        detections,
        left_on="sha256",
        right_on="source_raw_checksum",
        how="left",
        validate="one_to_one",
        suffixes=("", "_detection"),
    )
    if joined[["processed_checksum", "observation_group_id", "bls_config_hash"]].isna().any().any():
        raise LineageError(
            "Downloaded pilot lineage is incomplete through processed/BLS artifacts."
        )
    if not joined["source_raw_checksum"].eq(joined["sha256"]).all():
        raise LineageError("Processed input checksum does not match its raw product.")
    if not joined["source_raw_checksum_detection"].eq(joined["sha256"]).all():
        raise LineageError("Detection input checksum does not match its raw product.")

    rows: list[dict[str, object]] = []
    for row in joined.to_dict(orient="records"):
        checksum = validate_sha256(str(row["sha256"]))
        observation_id = make_observation_id(
            object_id=str(row["object_id"]),
            sector=int(row["sector"]),
            author=str(row["author"]),
            cadence_seconds=float(row["exposure_seconds"]),
            product_id=str(row["data_uri"]),
            checksum_sha256=checksum,
        )
        rows.append(
            {
                "object_id": row["object_id"],
                "observation_id": observation_id,
                "observation_group_id": row["observation_group_id"],
                "mast_obs_id": str(row["mast_obs_id"]),
                "sector": int(row["sector"]),
                "author": row["author"],
                "exposure_seconds": float(row["exposure_seconds"]),
                "source_product_id": row["data_uri"],
                "product_filename": row["product_filename"],
                "source_raw_checksum": checksum,
                "processed_checksum": row["processed_checksum"],
                "processed_artifact_id": "processed-"
                + content_hash(
                    {
                        "input": checksum,
                        "output": row["processed_checksum"],
                        "preprocessing_config_hash": row["preprocessing_config_hash"],
                    }
                ),
                "preprocessing_config_hash": row["preprocessing_config_hash"],
                "bls_config_hash": row["bls_config_hash"],
                "detection_status": row["status"],
            }
        )
    return pd.DataFrame(rows).sort_values("object_id", kind="stable").reset_index(drop=True)


def validate_frozen_candidate_ids(candidates: pd.DataFrame, lineage: pd.DataFrame) -> None:
    """Prove each frozen candidate identity against its group, config, and rank."""
    merged = candidates.merge(
        lineage[["observation_group_id", "bls_config_hash"]],
        on="observation_group_id",
        how="left",
        validate="many_to_one",
    )
    if merged["bls_config_hash"].isna().any():
        raise LineageError("Candidate refers to an unknown observation group.")
    for row in merged.to_dict(orient="records"):
        expected = make_candidate_id(
            observation_group_id=str(row["observation_group_id"]),
            bls_config_hash=str(row["bls_config_hash"]),
            rank=int(row["rank"]),
        )
        if row["candidate_id"] != expected:
            raise LineageError("Frozen candidate ID does not match its lineage inputs.")
