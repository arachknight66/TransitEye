"""Auditable canonical numeric-TIC TESS-SPOC MAST search."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

import pandas as pd

from transiteye.identifiers import normalize_object_id
from transiteye.serialization import canonical_json

MAST_QUERY_SCHEMA_VERSION = "mast-product-search-v2"
MAST_API_ENDPOINT = "https://mast.stsci.edu/api/v0.1/"
OBSERVATION_COLUMNS = (
    "object_id",
    "mast_target_name",
    "mast_obs_id",
    "mast_observation_identifier",
    "sector",
    "author",
    "exposure_seconds",
    "data_release",
    "search_timestamp_utc",
    "query_parameters_json",
    "mast_api_endpoint",
    "source_observation_json",
)
PRODUCT_COLUMNS = (
    "source_row_index",
    *OBSERVATION_COLUMNS,
    "product_filename",
    "data_uri",
    "product_size_bytes",
    "product_type",
    "product_subgroup",
    "description",
    "product_version",
    "source_product_json",
)


class MastSearchError(ValueError):
    """Raised when MAST returns a response unsuitable for provenance."""


class MastClient(Protocol):
    def query_observations(self, **criteria: Any) -> Any: ...
    def get_product_list(self, observations: Any) -> Any: ...


class AstroqueryMastClient:
    def query_observations(self, **criteria: Any) -> Any:
        from astroquery.mast import Observations  # type: ignore[import-untyped]

        return Observations.query_criteria(**criteria)

    def get_product_list(self, observations: Any) -> Any:
        from astroquery.mast import Observations

        return Observations.get_product_list(observations)


@dataclass(frozen=True)
class MastSearchResult:
    observations: pd.DataFrame
    products: pd.DataFrame
    query_parameters: dict[str, Any]


def mast_query_parameters(
    object_ids: Iterable[str], *, provenance_name: str = "TESS-SPOC"
) -> dict[str, Any]:
    """Use MAST's canonical TESS-SPOC numeric target-name convention."""
    values = [object_ids] if isinstance(object_ids, str) else object_ids
    targets = sorted({normalize_object_id(item).removeprefix("tic-") for item in values})
    if not targets or not provenance_name.strip():
        raise MastSearchError("Non-empty TIC identifiers and provenance are required.")
    return {"provenance_name": provenance_name.strip(), "target_name": targets}


def _records(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, pd.DataFrame):
        return [{str(k): v for k, v in row.items()} for row in value.to_dict(orient="records")]
    if hasattr(value, "colnames"):
        return [{str(name): row[name] for name in value.colnames} for row in value]
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, Mapping)):
        return [{str(k): v for k, v in dict(row).items()} for row in value]
    raise MastSearchError("MAST response is not a supported tabular response.")


def _value(row: Mapping[str, Any], *names: str) -> Any:
    values = {str(k).lower(): v for k, v in row.items()}
    for name in names:
        value = values.get(name.lower())
        if value is not None:
            scalar = value.item() if hasattr(value, "item") else value
            if str(scalar) not in {"--", "masked"} and not pd.isna(scalar):
                return scalar
    return None


def _json_safe(row: Mapping[str, Any]) -> dict[str, Any]:
    """Convert Astropy/Numpy scalar values into portable JSON scalars."""
    result: dict[str, Any] = {}
    for key, value in row.items():
        scalar = value.item() if hasattr(value, "item") else value
        result[str(key)] = None if pd.isna(scalar) else scalar
    return result


def _integer(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _number(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def normalize_mast_observations(
    observations: Any,
    *,
    object_ids: Iterable[str],
    searched_at_utc: datetime,
    query: Mapping[str, Any],
) -> pd.DataFrame:
    """Freeze every observation before requesting or filtering product rows."""
    if searched_at_utc.tzinfo is None:
        raise MastSearchError("Search timestamps must be timezone-aware.")
    mapping = {
        normalize_object_id(x).removeprefix("tic-"): normalize_object_id(x) for x in object_ids
    }
    query_json = canonical_json(dict(query))
    rows: list[dict[str, Any]] = []
    for row in _records(observations):
        target = str(_value(row, "target_name") or "").strip()
        if target not in mapping:
            raise MastSearchError(f"Unrequested MAST target_name: {target!r}")
        obsid = _value(row, "obsid", "obsID")
        rows.append(
            {
                "object_id": mapping[target],
                "mast_target_name": target,
                "mast_obs_id": str(obsid) if obsid is not None else None,
                "mast_observation_identifier": _value(row, "obs_id"),
                "sector": _integer(_value(row, "sequence_number")),
                "author": _value(row, "provenance_name"),
                "exposure_seconds": _number(_value(row, "t_exptime")),
                "data_release": _value(row, "dataReleaseDate", "t_obs_release"),
                "search_timestamp_utc": searched_at_utc.astimezone(UTC)
                .isoformat()
                .replace("+00:00", "Z"),
                "query_parameters_json": query_json,
                "mast_api_endpoint": MAST_API_ENDPOINT,
                "source_observation_json": canonical_json(_json_safe(row)),
            }
        )
    return pd.DataFrame.from_records(rows, columns=OBSERVATION_COLUMNS)


def normalize_mast_products(products: Any, observations: pd.DataFrame) -> pd.DataFrame:
    """Join all raw product rows to their frozen parent observations."""
    by_obsid = {
        str(row.mast_obs_id): row
        for row in observations.itertuples(index=False)
        if row.mast_obs_id is not None
    }
    rows: list[dict[str, Any]] = []
    for index, product in enumerate(_records(products)):
        obsid = _value(product, "obsID", "obsid", "obs_id")
        parent = by_obsid.get(str(obsid))
        filename = _value(product, "productFilename")
        uri = _value(product, "dataURI")
        if parent is None:
            raise MastSearchError(f"Product cannot be linked to observation: {obsid!r}")
        if filename is None or uri is None:
            raise MastSearchError("MAST product response is missing productFilename or dataURI.")
        row = {column: getattr(parent, column) for column in OBSERVATION_COLUMNS}
        row.update(
            {
                "source_row_index": index,
                "product_filename": str(filename),
                "data_uri": str(uri),
                "product_size_bytes": _integer(_value(product, "size")),
                "product_type": _value(product, "productType"),
                "product_subgroup": _value(product, "productSubGroupDescription"),
                "description": _value(product, "description"),
                "product_version": _value(product, "productVersion"),
                "source_product_json": canonical_json(_json_safe(product)),
            }
        )
        rows.append(row)
    return pd.DataFrame.from_records(rows, columns=PRODUCT_COLUMNS)


def normalize_mast_product_search(
    *,
    object_id: str,
    observations: Any,
    products: Any,
    searched_at_utc: datetime,
    query: Mapping[str, Any],
) -> pd.DataFrame:
    """Compatibility helper for one-TIC tests and callers."""
    normalized = normalize_mast_observations(
        observations, object_ids=[object_id], searched_at_utc=searched_at_utc, query=query
    )
    return normalize_mast_products(products, normalized)


def search_tess_products(
    object_ids: Iterable[str],
    *,
    provenance_name: str = "TESS-SPOC",
    client: MastClient | None = None,
) -> MastSearchResult:
    """Issue one canonical batched TIC query and retain observations before products."""
    requested = sorted({normalize_object_id(x) for x in object_ids})
    query = mast_query_parameters(requested, provenance_name=provenance_name)
    active = client or AstroqueryMastClient()
    raw_observations = active.query_observations(**query)
    observations = normalize_mast_observations(
        raw_observations, object_ids=requested, searched_at_utc=datetime.now(UTC), query=query
    )
    # MAST accepts table slices; bounded batches avoid one large product-list request.
    product_rows: list[dict[str, Any]] = []
    for start in range(0, len(raw_observations), 10):
        product_rows.extend(_records(active.get_product_list(raw_observations[start : start + 10])))
    raw_products = product_rows if not observations.empty else []
    return MastSearchResult(
        observations, normalize_mast_products(raw_products, observations), query
    )
