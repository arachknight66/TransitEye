"""NASA Exoplanet Archive TOI TAP query and normalization support."""

from __future__ import annotations

import csv
import io
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final, cast
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from transiteye.catalogs.labels import map_disposition
from transiteye.identifiers import normalize_object_id

TAP_SYNC_ENDPOINT: Final = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
TOI_TABLE: Final = "toi"
TOI_FIELDS: Final[tuple[str, ...]] = (
    "tid",
    "toi",
    "tfopwg_disp",
    "pl_orbper",
    "pl_tranmid",
    "pl_trandurh",
    "pl_trandep",
    "toi_created",
    "rowupdate",
)
NORMALIZATION_SCHEMA_VERSION: Final = "toi-normalized-v1"


class CatalogQueryError(RuntimeError):
    """Raised when the archive query cannot be completed."""


class CatalogResponseError(ValueError):
    """Raised when an archive response cannot be safely normalized."""


@dataclass(frozen=True)
class CatalogResponse:
    """The exact archive payload and its retrieval provenance."""

    endpoint: str
    request_url: str
    query: str
    retrieved_at_utc: datetime
    raw_bytes: bytes


Fetcher = Callable[[str, float], bytes]


def selected_toi_fields(fields: Sequence[str] = TOI_FIELDS) -> tuple[str, ...]:
    """Validate requested fields and return them in the canonical source order."""
    requested = tuple(fields)
    if not requested:
        raise ValueError("At least one TOI field must be requested.")
    if len(set(requested)) != len(requested):
        raise ValueError("TOI fields must not contain duplicates.")
    unknown = set(requested).difference(TOI_FIELDS)
    if unknown:
        raise ValueError(f"Unsupported TOI fields: {sorted(unknown)}")
    return tuple(field for field in TOI_FIELDS if field in requested)


def build_toi_query(fields: Sequence[str] = TOI_FIELDS) -> str:
    """Build a fixed-order ADQL query for the documented TOI fields only."""
    selected = selected_toi_fields(fields)
    return f"SELECT {','.join(selected)} FROM {TOI_TABLE} ORDER BY toi"


def build_tap_request_url(endpoint: str, query: str) -> str:
    """Build the portable TAP sync URL used for a CSV retrieval."""
    if not endpoint.startswith("https://"):
        raise ValueError("The catalog endpoint must use HTTPS.")
    return f"{endpoint}?{urlencode({'query': query, 'format': 'csv'})}"


def _default_fetcher(request_url: str, timeout_seconds: float) -> bytes:
    request = Request(request_url, headers={"User-Agent": "TransitEye/0.1 catalog snapshot"})
    try:
        with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
            return cast(bytes, response.read())
    except OSError as exc:
        raise CatalogQueryError("NASA Exoplanet Archive TAP request failed.") from exc


def fetch_toi_catalog(
    *,
    endpoint: str = TAP_SYNC_ENDPOINT,
    fields: Sequence[str] = TOI_FIELDS,
    timeout_seconds: float = 60.0,
    fetcher: Fetcher | None = None,
    retrieved_at_utc: datetime | None = None,
) -> CatalogResponse:
    """Retrieve a TOI CSV response, with injection support for offline tests."""
    if timeout_seconds <= 0:
        raise ValueError("Timeout must be positive.")
    query = build_toi_query(fields)
    request_url = build_tap_request_url(endpoint, query)
    raw_bytes = (fetcher or _default_fetcher)(request_url, timeout_seconds)
    if not raw_bytes.strip():
        raise CatalogResponseError("Archive response is empty.")
    timestamp = retrieved_at_utc or datetime.now(UTC)
    if timestamp.tzinfo is None:
        raise ValueError("Retrieval timestamp must be timezone-aware.")
    return CatalogResponse(
        endpoint=endpoint,
        request_url=request_url,
        query=query,
        retrieved_at_utc=timestamp.astimezone(UTC).replace(microsecond=0),
        raw_bytes=raw_bytes,
    )


def _optional_text(value: str) -> str | None:
    cleaned = value.strip()
    return cleaned or None


def _optional_float(value: str, *, field_name: str, row_index: int) -> float | None:
    text = _optional_text(value)
    if text is None or text.lower() in {"nan", "null", "none"}:
        return None
    try:
        numeric = float(text)
    except ValueError as exc:
        raise CatalogResponseError(
            f"Invalid numeric value for {field_name!r} in source row {row_index}: {text!r}"
        ) from exc
    if not math.isfinite(numeric):
        raise CatalogResponseError(
            f"Non-finite numeric value for {field_name!r} in source row {row_index}: {text!r}"
        )
    return numeric


def _normalize_toi_id(value: str, *, row_index: int) -> str:
    text = _optional_text(value)
    if text is None:
        raise CatalogResponseError(f"Missing required TOI identifier in source row {row_index}.")
    normalized = text.lower().replace(" ", "")
    if normalized.startswith("toi-"):
        normalized = normalized[4:]
    elif normalized.startswith("toi"):
        normalized = normalized[3:]
    if not normalized.replace(".", "", 1).isdecimal() or normalized.count(".") > 1:
        raise CatalogResponseError(f"Malformed TOI identifier in source row {row_index}: {text!r}")
    return f"toi-{normalized.lstrip('0') or '0'}"


def _validate_response_columns(fieldnames: Sequence[str] | None) -> tuple[str, ...]:
    if fieldnames is None:
        raise CatalogResponseError("Archive CSV response does not contain a header row.")
    headers = tuple(fieldnames)
    if len(set(headers)) != len(headers):
        raise CatalogResponseError("Archive CSV response contains duplicate header names.")
    missing = [field for field in TOI_FIELDS if field not in headers]
    if missing:
        raise CatalogResponseError(f"Archive CSV response is missing expected fields: {missing}")
    return headers


def normalize_toi_response(raw_bytes: bytes) -> pd.DataFrame:
    """Normalize an exact TOI CSV response without discarding source values."""
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CatalogResponseError("Archive CSV response is not UTF-8 text.") from exc
    try:
        reader = csv.DictReader(io.StringIO(text))
        _validate_response_columns(reader.fieldnames)
        source_rows = list(reader)
    except csv.Error as exc:
        raise CatalogResponseError("Archive response is not valid CSV.") from exc
    if not source_rows:
        raise CatalogResponseError("Archive CSV response contains no data rows.")

    normalized_rows: list[dict[str, object]] = []
    for row_index, source in enumerate(source_rows):
        source_values = {field: _optional_text(source.get(field, "")) for field in TOI_FIELDS}
        tid = source_values["tid"]
        try:
            object_id = normalize_object_id(tid) if tid is not None else None
            tic_status = "valid" if object_id is not None else "missing"
        except (TypeError, ValueError):
            object_id = None
            tic_status = "invalid"

        disposition = map_disposition(source_values["tfopwg_disp"])
        normalized_rows.append(
            {
                "source_row_index": row_index,
                "source_tid": source_values["tid"],
                "source_toi": source_values["toi"],
                "source_tfopwg_disp": source_values["tfopwg_disp"],
                "source_pl_orbper": source_values["pl_orbper"],
                "source_pl_tranmid": source_values["pl_tranmid"],
                "source_pl_trandurh": source_values["pl_trandurh"],
                "source_pl_trandep": source_values["pl_trandep"],
                "source_toi_created": source_values["toi_created"],
                "source_rowupdate": source_values["rowupdate"],
                "object_id": object_id,
                "tic_status": tic_status,
                "toi_id": _normalize_toi_id(source["toi"], row_index=row_index),
                "source_disposition": disposition.original_value,
                "mapped_label": disposition.internal_label.value,
                "disposition_subgroup": disposition.subgroup,
                "disposition_known": disposition.is_known,
                "period_days": _optional_float(
                    source["pl_orbper"], field_name="pl_orbper", row_index=row_index
                ),
                "transit_epoch_bjd": _optional_float(
                    source["pl_tranmid"], field_name="pl_tranmid", row_index=row_index
                ),
                "transit_duration_hours": _optional_float(
                    source["pl_trandurh"], field_name="pl_trandurh", row_index=row_index
                ),
                "transit_depth_ppm": _optional_float(
                    source["pl_trandep"], field_name="pl_trandep", row_index=row_index
                ),
                "toi_created": source_values["toi_created"],
                "rowupdate": source_values["rowupdate"],
            }
        )

    frame = pd.DataFrame(normalized_rows)
    frame["duplicate_toi"] = frame.duplicated(subset=["toi_id"], keep=False)
    return frame
