"""Create a versioned TOI disposition proxy-label sidecar for the 50,000-target corpus."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import platform
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

LOGGER = logging.getLogger(__name__)
DEFAULT_DATA_ROOT = Path(".transiteye-workspace/data/mast_unlabeled_50000_v1")
TOI_SOURCE_URL = "https://exofop.ipac.caltech.edu/tess/json/tois.csv"
CLASS_MAP = {
    "PC": "transit_like",
    "KP": "transit_like",
    "CP": "transit_like",
    "EB": "eclipse_like",
    "V": "stellar_variability",
    "IS": "other_artifact",
}
PLANET_DISPOSITIONS = {"PC", "KP", "CP"}
KEEP_TOI_COLUMNS = (
    "TOI",
    "TESS Disposition",
    "TFOPWG Disposition",
    "Sectors",
    "Source",
    "Detection",
    "Period (days)",
    "Epoch (BJD)",
    "Duration (hours)",
    "Depth (ppm)",
    "Comments",
    "Date Modified",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_tic(value: Any) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if not text.isdigit():
        raise ValueError(f"Invalid TIC identifier in catalog: {value!r}")
    return text


def _parse_sectors(value: str) -> set[int]:
    """Parse only comma-separated integer sector IDs; do not infer malformed tokens."""
    sectors: set[int] = set()
    for token in str(value).split(","):
        token = token.strip()
        if re.fullmatch(r"\d{1,3}", token):
            sector = int(token)
            if 1 <= sector <= 200:
                sectors.add(sector)
    return sectors


def _mapped_label(tess_disposition: str, tfopwg_disposition: str) -> tuple[str | None, str]:
    tess = tess_disposition.strip().upper()
    tfopwg = tfopwg_disposition.strip().upper()
    label = CLASS_MAP.get(tess)
    if label is None:
        return None, "unsupported_tess_disposition"
    if tfopwg in {"APC", "FP", "FA"}:
        return None, "ambiguous_or_false_positive_followup"
    if tfopwg and label == "transit_like" and tfopwg not in PLANET_DISPOSITIONS:
        return None, "conflicting_followup_disposition"
    if tfopwg and label != "transit_like" and tfopwg in PLANET_DISPOSITIONS:
        return None, "conflicting_followup_disposition"
    return label, "clear_same_event_catalog_proxy"


def build_sidecar(data_root: Path, toi_csv: Path) -> dict[str, Any]:
    manifest_path = data_root / "products.v1.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if not toi_csv.is_file():
        raise FileNotFoundError(toi_csv)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries: list[dict[str, Any]] = manifest["entries"]
    if len(entries) != manifest["target_count"]:
        raise ValueError("Frozen manifest entry count does not match target_count")
    product_ids = [item["product_id"] for item in entries]
    target_ids = [item["target_id"] for item in entries]
    if len(set(product_ids)) != len(product_ids) or len(set(target_ids)) != len(target_ids):
        raise ValueError("Frozen manifest contains duplicate products or TIC targets")

    toi_by_tic: dict[str, list[dict[str, str]]] = defaultdict(list)
    required_columns = {"TIC ID", "TOI", "TESS Disposition", "TFOPWG Disposition", "Sectors"}
    with toi_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or ())
        missing = required_columns - columns
        if missing:
            raise ValueError(f"TOI catalog is missing required columns: {sorted(missing)}")
        for row in reader:
            tic = _normalize_tic(row.get("TIC ID", ""))
            toi_by_tic[tic].append(row)

    label_rows: list[dict[str, Any]] = []
    match_rows: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()

    for item in entries:
        tic = _normalize_tic(item["target_id"])
        sector = int(item["sector"])
        target_matches = toi_by_tic.get(tic, [])
        same_sector: list[dict[str, str]] = []
        for match in target_matches:
            toi_sectors = _parse_sectors(match.get("Sectors", ""))
            event = {
                "product_id": item["product_id"],
                "target_id": tic,
                "product_sector": sector,
                "toi_id": match.get("TOI", "").strip(),
                "tess_disposition": match.get("TESS Disposition", "").strip().upper(),
                "tfopwg_disposition": match.get("TFOPWG Disposition", "").strip().upper(),
                "toi_sectors": match.get("Sectors", "").strip(),
                "toi_sector_match": sector in toi_sectors,
                "catalog_source": "ExoFOP-TESS TOI",
            }
            for column in KEEP_TOI_COLUMNS:
                if column in match:
                    event[f"toi_{column.lower().replace(' ', '_').replace('(', '').replace(')', '')}"] = (
                        match[column].strip()
                    )
            match_rows.append(event)
            if sector in toi_sectors:
                same_sector.append(match)

        label: str | None = None
        if not target_matches:
            status = "unlabeled_no_toi_match"
        elif not same_sector:
            status = "unlabeled_toi_sector_not_listed"
        else:
            event_labels: list[str | None] = []
            mapping_statuses: list[str] = []
            for match in same_sector:
                mapped, map_status = _mapped_label(
                    match.get("TESS Disposition", ""),
                    match.get("TFOPWG Disposition", ""),
                )
                event_labels.append(mapped)
                mapping_statuses.append(map_status)
            distinct_labels = {value for value in event_labels if value is not None}
            if all(value is not None for value in event_labels) and len(distinct_labels) == 1:
                label = distinct_labels.pop()
                status = "catalog_proxy_labeled"
                label_counts[label] += 1
            elif len(distinct_labels) > 1:
                status = "unlabeled_conflicting_toi_classes"
            elif any(value is None for value in event_labels):
                status = "unlabeled_ambiguous_or_unsupported_disposition"
            else:
                status = "unlabeled_no_usable_toi_disposition"

        status_counts[status] += 1
        label_rows.append(
            {
                "product_id": item["product_id"],
                "target_id": tic,
                "source_group_id": item["source_group_id"],
                "sector": sector,
                "label": label,
                "label_status": status,
                "label_scope": "same-sector TOI disposition catalog proxy",
                "catalog_match_count": len(target_matches),
                "same_sector_toi_count": len(same_sector),
                "toi_ids_same_sector_json": json.dumps(
                    sorted(match.get("TOI", "").strip() for match in same_sector)
                ),
                "tess_dispositions_same_sector_json": json.dumps(
                    sorted(match.get("TESS Disposition", "").strip().upper() for match in same_sector)
                ),
                "tfopwg_dispositions_same_sector_json": json.dumps(
                    sorted(match.get("TFOPWG Disposition", "").strip().upper() for match in same_sector)
                ),
            }
        )

    labels_path = data_root / "labels_toi_proxy.v1.parquet"
    matches_path = data_root / "toi_matches.v1.jsonl"
    report_path = data_root / "toi_crossmatch_report.v1.json"
    label_schema = pa.schema(
        [
            ("product_id", pa.string()),
            ("target_id", pa.string()),
            ("source_group_id", pa.string()),
            ("sector", pa.int16()),
            ("label", pa.string()),
            ("label_status", pa.string()),
            ("label_scope", pa.string()),
            ("catalog_match_count", pa.int16()),
            ("same_sector_toi_count", pa.int16()),
            ("toi_ids_same_sector_json", pa.string()),
            ("tess_dispositions_same_sector_json", pa.string()),
            ("tfopwg_dispositions_same_sector_json", pa.string()),
        ]
    )
    table = pa.Table.from_pylist(label_rows, schema=label_schema)
    pq.write_table(table, labels_path, compression="zstd")
    with matches_path.open("w", encoding="utf-8") as stream:
        for event in match_rows:
            stream.write(json.dumps(event, sort_keys=True, ensure_ascii=False) + "\n")

    report = {
        "schema_version": "1",
        "dataset_id": manifest["dataset_id"],
        "product_manifest_id": manifest["manifest_id"],
        "label_snapshot_id": hashlib.sha256(
            f"{manifest['manifest_id']}:{_sha256(toi_csv)}:toi-proxy-policy-v1".encode()
        ).hexdigest()[:16],
        "retrieved_at": datetime.now(UTC).isoformat(),
        "catalog_name": "ExoFOP-TESS current public TOI table",
        "catalog_url": TOI_SOURCE_URL,
        "catalog_file": toi_csv.name,
        "catalog_sha256": _sha256(toi_csv),
        "catalog_latest_date_modified": max(
            (row.get("Date Modified", "").strip() for rows in toi_by_tic.values() for row in rows),
            default="",
        ),
        "product_manifest_sha256": _sha256(manifest_path),
        "crossmatch_script_sha256": _sha256(Path(__file__).resolve()),
        "python_version": platform.python_version(),
        "pyarrow_version": pa.__version__,
        "catalog_row_count": sum(len(rows) for rows in toi_by_tic.values()),
        "catalog_unique_tic_count": len(toi_by_tic),
        "product_count": len(label_rows),
        "exact_tic_matched_product_count": sum(row["catalog_match_count"] > 0 for row in label_rows),
        "products_with_same_sector_toi": sum(row["same_sector_toi_count"] > 0 for row in label_rows),
        "same_sector_toi_event_count": sum(row["same_sector_toi_count"] for row in label_rows),
        "label_counts": dict(sorted(label_counts.items())),
        "label_status_counts": dict(sorted(status_counts.items())),
        "label_policy": {
            "join": "exact TIC ID; no coordinate-radius matching",
            "sector_gate": "TOI sector list must include the light-curve product sector",
            "tess_disposition_mapping": CLASS_MAP,
            "followup_gate": "APC/FP/FA and conflicting follow-up dispositions remain unlabeled",
            "unmatched_policy": "unmatched, off-sector, ambiguous, and unsupported entries stay unlabeled",
            "feature_leakage_policy": "TOI IDs, ephemerides, catalog dispositions, and TIC parameters are not model features",
            "scope": "catalog proxy for a cataloged event; not independent source-local morphology truth or planet confirmation",
        },
        "labels_parquet": labels_path.name,
        "toi_matches_jsonl": matches_path.name,
        "labels_parquet_sha256": _sha256(labels_path),
        "toi_matches_jsonl_sha256": _sha256(matches_path),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    LOGGER.info(
        "Created TOI proxy labels for %d/%d products (%s)",
        sum(label_counts.values()),
        len(label_rows),
        dict(label_counts),
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--toi-csv",
        type=Path,
        default=DEFAULT_DATA_ROOT / "catalogs" / "tois.exofop_current.csv",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    report = build_sidecar(args.data_root, args.toi_csv)
    LOGGER.info(
        "TOI cross-match report: %d exact matches; labels=%s; statuses=%s",
        report["exact_tic_matched_product_count"],
        report["label_counts"],
        report["label_status_counts"],
    )


if __name__ == "__main__":
    main()
