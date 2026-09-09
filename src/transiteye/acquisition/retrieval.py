"""Resumable frozen-observation MAST product retrieval."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from transiteye.acquisition.checksums import sha256_file
from transiteye.acquisition.mast import AstroqueryMastClient, normalize_mast_products
from transiteye.serialization import content_hash


def retrieve_pending(observations: pd.DataFrame, root: str | Path) -> pd.DataFrame:
    """Fetch one frozen observation at a time, atomically checkpointing every terminal state."""
    directory = Path(root)
    directory.mkdir(parents=True, exist_ok=True)
    status_path = directory / "coverage.parquet"
    status = (
        pd.read_parquet(status_path)
        if status_path.exists()
        else pd.DataFrame(
            {
                "mast_obs_id": observations.mast_obs_id.astype(str),
                "status": "pending",
                "error": None,
                "retrieved_at_utc": None,
            }
        )
    )
    client = AstroqueryMastClient()
    for _, observation in observations.iterrows():
        obsid = str(observation.mast_obs_id)
        index = status.index[status.mast_obs_id == obsid][0]
        if status.loc[index, "status"] != "pending":
            continue
        try:
            # Query only the frozen observation identifier; source metadata remains authoritative.
            from astroquery.mast import Observations  # type: ignore[import-untyped]

            raw = Observations.query_criteria(obsid=obsid)
            products = client.get_product_list(raw)
            normalized = normalize_mast_products(
                products, observations.loc[observations.mast_obs_id.astype(str) == obsid]
            )
            normalized.to_parquet(directory / f"products-{obsid}.parquet", index=False)
            status.loc[index, ["status", "error"]] = [
                "success" if len(normalized) else "zero_products",
                None,
            ]
        except Exception as exc:
            status.loc[index, ["status", "error"]] = ["failed", f"{type(exc).__name__}: {exc}"]
        status.loc[index, "retrieved_at_utc"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        status.to_parquet(status_path, index=False)
    return status


def freeze_products(root: str | Path) -> Path:
    """Freeze complete product discovery only after every observation has terminal coverage."""
    directory = Path(root)
    coverage = pd.read_parquet(directory / "coverage.parquet")
    if (coverage.status == "pending").any():
        raise ValueError("Product snapshot has pending observations.")
    products = (
        pd.concat(
            [pd.read_parquet(p) for p in sorted(directory.glob("products-*.parquet"))],
            ignore_index=True,
        )
        if list(directory.glob("products-*.parquet"))
        else pd.DataFrame()
    )
    products = products.drop_duplicates(subset=["mast_obs_id", "data_uri"]).reset_index(drop=True)
    products.to_parquet(directory / "normalized_products.parquet", index=False)
    products_sha256 = sha256_file(directory / "normalized_products.parquet")
    identity = {
        "coverage": coverage[["mast_obs_id", "status"]].to_dict(orient="records"),
        "products_sha256": products_sha256,
    }
    snapshot_id = f"mast-products-{content_hash(identity)}"
    (directory / "metadata.json").write_text(
        json.dumps(
            {
                "snapshot_id": snapshot_id,
                "observation_count": len(coverage),
                "coverage_counts": coverage.status.value_counts().to_dict(),
                "product_rows": len(products),
                "products_sha256": products_sha256,
            },
            sort_keys=True,
        )
        + "\n"
    )
    return directory
