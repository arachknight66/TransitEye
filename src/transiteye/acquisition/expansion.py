"""Frozen, label-independent product selection for real-cohort expansion."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from transiteye.acquisition.checksums import sha256_file
from transiteye.config import ExpansionSettings
from transiteye.serialization import canonical_json, content_hash


class ExpansionManifestError(ValueError):
    """Raised when an expansion selection cannot be frozen safely."""


@dataclass(frozen=True)
class ExpansionManifest:
    """Selected and excluded frozen product rows plus portable policy provenance."""

    products: pd.DataFrame
    selected_products: pd.DataFrame
    manifest_id: str
    policy_hash: str
    qa_summary: dict[str, Any]


_REQUIRED_COLUMNS = {
    "object_id",
    "mast_obs_id",
    "sector",
    "author",
    "exposure_seconds",
    "product_filename",
    "data_uri",
    "product_size_bytes",
}


def build_expansion_manifest(
    products: pd.DataFrame,
    *,
    settings: ExpansionSettings,
    product_snapshot_sha256: str,
) -> ExpansionManifest:
    """Select earliest homogeneous products without accepting labels or ephemerides."""
    missing = sorted(_REQUIRED_COLUMNS.difference(products.columns))
    if missing:
        raise ExpansionManifestError(f"Frozen products missing required columns: {missing}")
    policy_hash = content_hash(settings.model_dump(mode="json"))
    annotated = products.copy(deep=True)
    annotated["expansion_policy_hash"] = policy_hash
    annotated["selection_status"] = "excluded"
    annotated["selection_reason"] = "not_evaluated"
    filename = annotated["product_filename"].fillna("").astype(str).str.lower()
    author = annotated["author"].fillna("").astype(str).str.upper()
    is_lightcurve = filename.str.endswith(settings.required_product_suffix)
    is_author = author.eq(settings.required_author.upper())
    is_exposure = annotated["exposure_seconds"].eq(settings.preferred_exposure_seconds)
    annotated.loc[~is_lightcurve, "selection_reason"] = "not_lightcurve_product"
    annotated.loc[is_lightcurve & ~is_author, "selection_reason"] = "nonpreferred_author"
    annotated.loc[is_lightcurve & is_author & ~is_exposure, "selection_reason"] = (
        "nonpreferred_exposure"
    )
    eligible = annotated.loc[is_lightcurve & is_author & is_exposure].copy()
    eligible = eligible.sort_values(
        ["object_id", "sector", "data_release", "product_filename", "data_uri"],
        ascending=[True, True, False, True, True],
        kind="stable",
    ).drop_duplicates(["object_id", "sector"], keep="first")
    selected_index = eligible.groupby("object_id", sort=True).head(settings.sectors_per_tic).index
    annotated.loc[eligible.index, "selection_reason"] = "sector_bound"
    annotated.loc[selected_index, "selection_status"] = "selected"
    annotated.loc[selected_index, "selection_reason"] = "earliest_qualifying_sector"
    annotated["download_selection_reason"] = ""
    annotated.loc[annotated["selection_status"].eq("selected"), "download_selection_reason"] = (
        annotated.loc[annotated["selection_status"].eq("selected"), "selection_reason"]
    )
    annotated = annotated.sort_values(
        ["object_id", "sector", "product_filename", "data_uri"], kind="stable"
    ).reset_index(drop=True)
    selected = (
        annotated.loc[annotated["selection_status"].eq("selected")].copy().reset_index(drop=True)
    )
    identity = {
        "policy": settings.model_dump(mode="json"),
        "product_snapshot_sha256": product_snapshot_sha256,
        "selected": selected[
            ["object_id", "mast_obs_id", "sector", "exposure_seconds", "data_uri"]
        ].to_dict(orient="records"),
    }
    manifest_id = f"expansion-{content_hash(identity)}"
    qa = {
        "product_snapshot_id": settings.product_snapshot_id,
        "preferred_exposure_seconds": settings.preferred_exposure_seconds,
        "available_tics_at_preferred_exposure": int(eligible["object_id"].nunique()),
        "selected_tics": int(selected["object_id"].nunique()),
        "selected_products": len(selected),
        "estimated_download_bytes": int(pd.to_numeric(selected["product_size_bytes"]).sum()),
        "selected_by_sector": {
            str(key): int(value)
            for key, value in selected["sector"].value_counts().sort_index().items()
        },
        "excluded_by_reason": {
            str(key): int(value)
            for key, value in annotated.loc[
                annotated["selection_status"] == "excluded", "selection_reason"
            ]
            .value_counts()
            .sort_index()
            .items()
        },
    }
    return ExpansionManifest(annotated, selected, manifest_id, policy_hash, qa)


def write_expansion_manifest(manifest: ExpansionManifest, *, root: str | Path) -> Path:
    """Freeze the pre-download policy decision; later execution does not alter it."""
    directory = Path(root) / manifest.manifest_id
    if directory.exists():
        return load_expansion_manifest(directory)[0]
    directory.mkdir(parents=True, exist_ok=False)
    try:
        products_path = directory / "eligible_and_excluded_products.parquet"
        selected_path = directory / "selected_products.parquet"
        manifest.products.to_parquet(products_path, engine="pyarrow", index=False)
        manifest.selected_products.to_parquet(selected_path, engine="pyarrow", index=False)
        metadata = {
            "manifest_id": manifest.manifest_id,
            "expansion_policy_hash": manifest.policy_hash,
            "qa_summary": manifest.qa_summary,
            "eligible_and_excluded_products_sha256": sha256_file(products_path),
            "selected_products_sha256": sha256_file(selected_path),
        }
        (directory / "metadata.json").write_text(canonical_json(metadata) + "\n", encoding="utf-8")
    except Exception:
        for item in directory.iterdir():
            item.unlink()
        directory.rmdir()
        raise
    return directory


def load_expansion_manifest(directory: str | Path) -> tuple[Path, ExpansionManifest]:
    """Reload a frozen expansion decision only if its selected rows remain intact."""
    root = Path(directory)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    products_path = root / "eligible_and_excluded_products.parquet"
    selected_path = root / "selected_products.parquet"
    if sha256_file(products_path) != metadata["eligible_and_excluded_products_sha256"]:
        raise ExpansionManifestError("Expansion product table checksum mismatch.")
    if sha256_file(selected_path) != metadata["selected_products_sha256"]:
        raise ExpansionManifestError("Expansion selected table checksum mismatch.")
    manifest = ExpansionManifest(
        pd.read_parquet(products_path),
        pd.read_parquet(selected_path),
        str(metadata["manifest_id"]),
        str(metadata["expansion_policy_hash"]),
        dict(metadata["qa_summary"]),
    )
    return root, manifest
