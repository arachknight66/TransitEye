"""Deterministic product selection, acquisition planning, and portable manifests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from transiteye.acquisition.checksums import sha256_file
from transiteye.config import AcquisitionSettings
from transiteye.serialization import canonical_json, content_hash


class AcquisitionManifestError(ValueError):
    """Raised when a product manifest cannot be safely constructed or written."""


@dataclass(frozen=True)
class AcquisitionPlan:
    """All available candidates, selected core products, and auditable QA."""

    available_products: pd.DataFrame
    selected_products: pd.DataFrame
    qa_summary: dict[str, Any]
    plan_id: str


def _counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in frame:
        return {}
    values = frame[column].fillna("<missing>").astype(str).value_counts().sort_index()
    return {str(key): int(value) for key, value in values.items()}


def _is_lightcurve(row: pd.Series) -> bool:
    filename = str(row.get("product_filename") or "").lower()
    subgroup = str(row.get("product_subgroup") or "").strip().lower()
    return filename.endswith(".fits") and (
        subgroup == "light curves" or filename.endswith("_lc.fits")
    )


def apply_product_policy(products: pd.DataFrame, settings: AcquisitionSettings) -> pd.DataFrame:
    """Annotate every MAST result; only explicit preferred products become core selections."""
    required = {"object_id", "product_filename", "data_uri", "author", "sector"}
    missing = sorted(required.difference(products.columns))
    if missing:
        raise AcquisitionManifestError(f"Search results missing required fields: {missing}")
    available = products.copy(deep=True)
    available["selection_status"] = "excluded"
    available["selection_reason"] = "not_evaluated"
    available["is_selected_core"] = False
    for index, row in available.iterrows():
        if not _is_lightcurve(row):
            available.loc[index, "selection_reason"] = "not_lightcurve_fits"
        elif str(row["author"] or "").strip().upper() != settings.preferred_author.upper():
            available.loc[index, "selection_reason"] = "nonpreferred_author"
        elif (
            settings.preferred_exposure_seconds is not None
            and row.get("exposure_seconds") != settings.preferred_exposure_seconds
        ):
            available.loc[index, "selection_reason"] = "nonpreferred_exposure"
        else:
            available.loc[index, "selection_status"] = "core_candidate"
            available.loc[index, "selection_reason"] = "preferred_product"
    candidates = available.loc[available["selection_status"] == "core_candidate"].copy()
    if not candidates.empty:
        candidates["_release_sort"] = candidates["data_release"].fillna("").astype(str)
        candidates = candidates.sort_values(
            [
                "object_id",
                "sector",
                "author",
                "exposure_seconds",
                "_release_sort",
                "product_filename",
                "data_uri",
            ],
            ascending=[True, True, True, True, False, True, True],
            kind="stable",
        )
        keys = ["object_id", "sector", "author", "exposure_seconds"]
        winners = candidates.drop_duplicates(subset=keys, keep="first").index
        available.loc[winners, "selection_status"] = "selected"
        available.loc[winners, "selection_reason"] = "preferred_product"
        redundant = candidates.index.difference(winners)
        available.loc[redundant, "selection_status"] = "excluded"
        available.loc[redundant, "selection_reason"] = "redundant_product"
    preferred_by_tic = available.loc[
        (available["selection_status"] == "selected"), "object_id"
    ].astype(str)
    for object_id in sorted(set(available["object_id"].astype(str)).difference(preferred_by_tic)):
        rows = available["object_id"].astype(str) == object_id
        available.loc[
            rows & (available["selection_reason"] == "nonpreferred_author"), "selection_reason"
        ] = "preferred_author_unavailable"
    available["is_selected_core"] = available["selection_status"] == "selected"
    return available.sort_values(
        ["object_id", "sector", "product_filename", "data_uri"], kind="stable"
    ).reset_index(drop=True)


def build_acquisition_plan(
    products: pd.DataFrame, *, settings: AcquisitionSettings, requested_object_ids: list[str]
) -> AcquisitionPlan:
    """Create a dry-run plan; this function never downloads product data."""
    annotated = apply_product_policy(products, settings)
    selected = annotated.loc[annotated["is_selected_core"]].copy().reset_index(drop=True)
    requested = sorted(set(requested_object_ids))
    objects_with_results = set(annotated["object_id"].astype(str))
    selected_sizes = pd.to_numeric(selected["product_size_bytes"], errors="coerce").fillna(0)
    selected_by_tic = (
        selected.groupby("object_id")["product_size_bytes"].sum().to_dict()
        if not selected.empty
        else {}
    )
    qa = {
        "pilot_tic_count": len(requested),
        "tics_with_mast_results": len(objects_with_results.intersection(requested)),
        "tics_with_no_mast_results": sorted(set(requested).difference(objects_with_results)),
        "tics_without_preferred_products": sorted(
            set(requested).difference(set(selected["object_id"]))
        ),
        "total_returned_products": len(annotated),
        "selected_products": len(selected),
        "products_excluded_by_policy": int((annotated["selection_status"] == "excluded").sum()),
        "selected_by_author": _counts(selected, "author"),
        "selected_by_sector": _counts(selected, "sector"),
        "selected_by_exposure_seconds": _counts(selected, "exposure_seconds"),
        "estimated_selected_download_bytes": int(selected_sizes.sum()),
        "estimated_bytes_per_tic": {str(key): int(value) for key, value in selected_by_tic.items()},
        "duplicate_or_redundant_product_count": int(
            (annotated["selection_reason"] == "redundant_product").sum()
        ),
        "unresolved_selection_ambiguities": [],
    }
    identity = {
        "settings": settings.model_dump(mode="json"),
        "products": annotated.to_dict(orient="records"),
        "requested": requested,
    }
    plan_id = f"acq-{content_hash(identity)}"
    return AcquisitionPlan(annotated, selected, qa, plan_id)


def select_download_pilot(plan: AcquisitionPlan, settings: AcquisitionSettings) -> pd.DataFrame:
    """Choose a small deterministic subset after review, keeping one earliest sector per TIC."""
    selected = plan.selected_products.sort_values(
        ["object_id", "sector", "product_filename", "data_uri"], kind="stable"
    )
    per_target = selected.groupby("object_id", sort=True).head(
        settings.download_pilot_sectors_per_target
    )
    result = per_target.groupby("object_id", sort=True).head(
        settings.download_pilot_sectors_per_target
    )
    target_ids = sorted(result["object_id"].unique())[: settings.download_pilot_target_limit]
    result = result.loc[result["object_id"].isin(target_ids)].copy().reset_index(drop=True)
    result["download_selection_reason"] = "small_development_pilot: earliest selected sector"
    return result


def write_acquisition_plan(plan: AcquisitionPlan, *, root: str | Path = "data/manifests") -> Path:
    """Write an immutable, replayable search and selected-product plan."""
    directory = Path(root) / plan.plan_id
    metadata = {
        "plan_id": plan.plan_id,
        "available_products_sha256": None,
        "selected_products_sha256": None,
        "qa_summary": plan.qa_summary,
    }
    if directory.exists():
        metadata_path = directory / "metadata.json"
        if metadata_path.exists():
            current = json.loads(metadata_path.read_text(encoding="utf-8"))
            if current["plan_id"] == plan.plan_id:
                return directory
        raise AcquisitionManifestError(f"Immutable acquisition manifest exists: {directory}")
    directory.mkdir(parents=True, exist_ok=False)
    try:
        available_path = directory / "mast_search.parquet"
        selected_path = directory / "selected_products.parquet"
        plan.available_products.to_parquet(available_path, engine="pyarrow", index=False)
        plan.selected_products.to_parquet(selected_path, engine="pyarrow", index=False)
        metadata["available_products_sha256"] = sha256_file(available_path)
        metadata["selected_products_sha256"] = sha256_file(selected_path)
        (directory / "metadata.json").write_text(canonical_json(metadata) + "\n", encoding="utf-8")
    except Exception:
        for path in directory.iterdir():
            path.unlink()
        directory.rmdir()
        raise
    return directory


def load_acquisition_plan(directory: str | Path) -> AcquisitionPlan:
    """Load a plan only when its recorded Parquet checksums still match."""
    root = Path(directory)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    available_path = root / "mast_search.parquet"
    selected_path = root / "selected_products.parquet"
    if sha256_file(available_path) != metadata["available_products_sha256"]:
        raise AcquisitionManifestError("Search manifest checksum mismatch.")
    if sha256_file(selected_path) != metadata["selected_products_sha256"]:
        raise AcquisitionManifestError("Selected-product manifest checksum mismatch.")
    return AcquisitionPlan(
        pd.read_parquet(available_path, engine="pyarrow"),
        pd.read_parquet(selected_path, engine="pyarrow"),
        metadata["qa_summary"],
        metadata["plan_id"],
    )
