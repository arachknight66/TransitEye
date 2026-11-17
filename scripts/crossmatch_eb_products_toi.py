"""Create same-sector ExoFOP-TESS disposition proxies for the EB light-curve cohort."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import platform
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

LOGGER = logging.getLogger(__name__)
DEFAULT_DATA_ROOT = Path(".transiteye-workspace/data/mast_eclipsing_binary_catalogs_v1")
DEFAULT_TOI_CSV = Path(
    ".transiteye-workspace/data/mast_unlabeled_50000_v1/catalogs/tois.exofop_current.csv"
)
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
    "TOI", "TESS Disposition", "TFOPWG Disposition", "Sectors", "Source", "Detection",
    "Period (days)", "Epoch (BJD)", "Duration (hours)", "Depth (ppm)", "Comments", "Date Modified",
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
        raise ValueError(f"Invalid TIC identifier in TOI table or product manifest: {value!r}")
    return text


def _parse_sectors(value: Any) -> set[int]:
    sectors: set[int] = set()
    for token in str(value or "").split(","):
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


def _atomic_write(path: Path, writer: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, filename = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(filename)
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_sidecars(data_root: Path, toi_csv: Path) -> dict[str, Any]:
    manifest_path = data_root / "products.v1.json"
    eb_labels_path = data_root / "labels_eb_catalog_proxy.v1.parquet"
    if not manifest_path.is_file() or not toi_csv.is_file() or not eb_labels_path.is_file():
        raise FileNotFoundError("Expected frozen product manifest, TOI CSV, and EB label sidecar")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries: list[dict[str, Any]] = manifest["entries"]
    if len(entries) != manifest.get("available_mast_spoc_product_count"):
        raise ValueError("Frozen product manifest count does not match available product count")
    product_ids = [item["product_id"] for item in entries]
    if len(set(product_ids)) != len(product_ids):
        raise ValueError("Frozen product manifest contains duplicate product IDs")

    required_columns = {"TIC ID", "TOI", "TESS Disposition", "TFOPWG Disposition", "Sectors"}
    toi_by_tic: dict[str, list[dict[str, str]]] = defaultdict(list)
    with toi_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = required_columns - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"TOI snapshot is missing required columns: {sorted(missing)}")
        for row in reader:
            toi_by_tic[_normalize_tic(row.get("TIC ID", ""))].append(row)

    snapshot_path = data_root / "catalogs" / toi_csv.name
    source_hash = _sha256(toi_csv)
    if toi_csv.resolve() != snapshot_path.resolve():
        def copy_snapshot(temporary: Path) -> None:
            shutil.copyfile(toi_csv, temporary)
        _atomic_write(snapshot_path, copy_snapshot)
    if _sha256(snapshot_path) != source_hash:
        raise ValueError("Copied TOI catalog snapshot checksum does not match source")

    eb_labels = pq.read_table(eb_labels_path, columns=["product_id", "label", "label_status"]).to_pylist()
    eb_by_product = {row["product_id"]: row for row in eb_labels}
    if len(eb_by_product) != len(entries) or set(eb_by_product) != set(product_ids):
        raise ValueError("EB label sidecar product IDs do not match the frozen manifest")

    label_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    overlap = Counter()

    for item in entries:
        tic = _normalize_tic(item["target_id"])
        sector = int(item["sector"])
        product_id = str(item["product_id"])
        target_matches = toi_by_tic.get(tic, [])
        same_sector: list[dict[str, str]] = []
        for match in target_matches:
            toi_sectors = _parse_sectors(match.get("Sectors", ""))
            mapped_label, mapping_status = _mapped_label(
                match.get("TESS Disposition", ""), match.get("TFOPWG Disposition", "")
            )
            event = {
                "product_id": product_id,
                "target_id": tic,
                "source_group_id": item["source_group_id"],
                "product_sector": sector,
                "toi_id": match.get("TOI", "").strip(),
                "tess_disposition": match.get("TESS Disposition", "").strip().upper(),
                "tfopwg_disposition": match.get("TFOPWG Disposition", "").strip().upper(),
                "toi_sectors": match.get("Sectors", "").strip(),
                "toi_sector_match": sector in toi_sectors,
                "mapped_label": mapped_label,
                "mapping_status": mapping_status,
                "catalog_source": "ExoFOP-TESS TOI",
            }
            for column in KEEP_TOI_COLUMNS:
                if column in match:
                    event[f"toi_{column.lower().replace(' ', '_').replace('(', '').replace(')', '')}"] = (
                        match[column].strip()
                    )
            event_rows.append(event)
            if sector in toi_sectors:
                same_sector.append(match)

        mapped = [
            _mapped_label(match.get("TESS Disposition", ""), match.get("TFOPWG Disposition", ""))
            for match in same_sector
        ]
        non_null_classes = {label for label, _ in mapped if label is not None}
        if not target_matches:
            label, status = None, "unlabeled_no_toi_match"
        elif not same_sector:
            label, status = None, "unlabeled_toi_sector_not_listed"
        elif len(non_null_classes) > 1:
            label, status = None, "unlabeled_conflicting_toi_classes"
        elif any(mapped_label is None for mapped_label, _ in mapped):
            label, status = None, "unlabeled_ambiguous_or_unsupported_disposition"
        elif len(non_null_classes) == 1:
            label, status = non_null_classes.pop(), "catalog_proxy_labeled"
        else:
            label, status = None, "unlabeled_no_usable_toi_disposition"

        eb_row = eb_by_product[product_id]
        eb_label = eb_row.get("label")
        if eb_label is not None and label is not None:
            overlap["both_catalogs_labeled"] += 1
            if eb_label == label:
                overlap["both_labeled_same_class"] += 1
            else:
                overlap["both_labeled_conflicting_classes"] += 1
        elif eb_label is not None:
            overlap["eb_only_labeled"] += 1
        elif label is not None:
            overlap["toi_only_labeled"] += 1

        if label is not None:
            label_counts[label] += 1
        status_counts[status] += 1
        label_rows.append({
            "product_id": product_id,
            "target_id": tic,
            "source_group_id": str(item["source_group_id"]),
            "sector": sector,
            "label": label,
            "label_status": status,
            "label_scope": "same-sector ExoFOP-TESS disposition catalog proxy",
            "catalog_match_count": len(target_matches),
            "same_sector_toi_count": len(same_sector),
            "toi_ids_same_sector_json": json.dumps(sorted(match.get("TOI", "").strip() for match in same_sector)),
            "tess_dispositions_same_sector_json": json.dumps(sorted(
                match.get("TESS Disposition", "").strip().upper() for match in same_sector
            )),
            "tfopwg_dispositions_same_sector_json": json.dumps(sorted(
                match.get("TFOPWG Disposition", "").strip().upper() for match in same_sector
            )),
            "eb_catalog_label": eb_label,
            "eb_catalog_label_status": eb_row["label_status"],
            "cross_catalog_class_conflict": bool(
                eb_label is not None and label is not None and eb_label != label
            ),
        })

    labels_path = data_root / "labels_toi_proxy.v1.parquet"
    combined_path = data_root / "labels_eb_toi_catalog_proxy.v1.parquet"
    matches_path = data_root / "toi_matches.v1.jsonl"
    report_path = data_root / "toi_crossmatch_report.v1.json"
    schema = pa.schema([
        ("product_id", pa.string()), ("target_id", pa.string()),
        ("source_group_id", pa.string()), ("sector", pa.int16()), ("label", pa.string()),
        ("label_status", pa.string()), ("label_scope", pa.string()),
        ("catalog_match_count", pa.int16()), ("same_sector_toi_count", pa.int16()),
        ("toi_ids_same_sector_json", pa.string()), ("tess_dispositions_same_sector_json", pa.string()),
        ("tfopwg_dispositions_same_sector_json", pa.string()), ("eb_catalog_label", pa.string()),
        ("eb_catalog_label_status", pa.string()), ("cross_catalog_class_conflict", pa.bool_()),
    ])
    table = pa.Table.from_pylist(label_rows, schema=schema)
    _atomic_write(labels_path, lambda temporary: pq.write_table(table, temporary, compression="zstd"))

    combined_rows: list[dict[str, Any]] = []
    combined_label_counts: Counter[str] = Counter()
    combined_status_counts: Counter[str] = Counter()
    for row in label_rows:
        eb_label = row["eb_catalog_label"]
        toi_label = row["label"]
        if eb_label is not None and toi_label is not None and eb_label != toi_label:
            combined_label = None
            combined_status = "unlabeled_conflicting_eb_and_toi_proxy_classes"
        elif eb_label is not None and toi_label is not None:
            combined_label = eb_label
            combined_status = "both_catalog_overlays_same_class"
        elif eb_label is not None:
            combined_label = eb_label
            combined_status = "eb_catalog_proxy_only"
        elif toi_label is not None:
            combined_label = toi_label
            combined_status = "toi_catalog_proxy_only"
        else:
            combined_label = None
            combined_status = "unlabeled_in_both_catalog_overlays"
        if combined_label is not None:
            combined_label_counts[combined_label] += 1
        combined_status_counts[combined_status] += 1
        combined_rows.append({
            "product_id": row["product_id"],
            "target_id": row["target_id"],
            "source_group_id": row["source_group_id"],
            "sector": row["sector"],
            "label": combined_label,
            "label_status": combined_status,
            "eb_catalog_label": eb_label,
            "eb_catalog_label_status": row["eb_catalog_label_status"],
            "toi_catalog_label": toi_label,
            "toi_catalog_label_status": row["label_status"],
            "cross_catalog_class_conflict": row["cross_catalog_class_conflict"],
        })
    combined_schema = pa.schema([
        ("product_id", pa.string()), ("target_id", pa.string()),
        ("source_group_id", pa.string()), ("sector", pa.int16()), ("label", pa.string()),
        ("label_status", pa.string()), ("eb_catalog_label", pa.string()),
        ("eb_catalog_label_status", pa.string()), ("toi_catalog_label", pa.string()),
        ("toi_catalog_label_status", pa.string()), ("cross_catalog_class_conflict", pa.bool_()),
    ])
    combined_table = pa.Table.from_pylist(combined_rows, schema=combined_schema)
    _atomic_write(combined_path, lambda temporary: pq.write_table(combined_table, temporary, compression="zstd"))

    def write_matches(temporary: Path) -> None:
        with temporary.open("w", encoding="utf-8") as stream:
            for event in event_rows:
                stream.write(json.dumps(event, sort_keys=True, ensure_ascii=False) + "\n")
    _atomic_write(matches_path, write_matches)

    report = {
        "schema_version": "1",
        "dataset_id": manifest["dataset_id"],
        "label_policy_id": "toi_same_sector_disposition_proxy.v1",
        "label_snapshot_id": hashlib.sha256(
            f"{manifest['manifest_id']}:{source_hash}:toi-proxy-policy-v1".encode()
        ).hexdigest()[:16],
        "retrieved_at_utc": datetime.now(UTC).isoformat(),
        "catalog_name": "ExoFOP-TESS TOI table snapshot reused from the 50,000-target overlay",
        "catalog_url": TOI_SOURCE_URL,
        "catalog_snapshot_path": str(snapshot_path.relative_to(data_root)),
        "catalog_sha256": source_hash,
        "catalog_latest_date_modified": max(
            (row.get("Date Modified", "").strip() for rows in toi_by_tic.values() for row in rows),
            default="",
        ),
        "catalog_row_count": sum(len(rows) for rows in toi_by_tic.values()),
        "catalog_unique_tic_count": len(toi_by_tic),
        "product_manifest_id": manifest["manifest_id"],
        "product_manifest_sha256": _sha256(manifest_path),
        "eb_label_sidecar_sha256": _sha256(eb_labels_path),
        "crossmatch_script_sha256": _sha256(Path(__file__).resolve()),
        "python_version": platform.python_version(),
        "pyarrow_version": pa.__version__,
        "product_count": len(label_rows),
        "distinct_tic_count": len({row["target_id"] for row in label_rows}),
        "exact_tic_matched_product_count": sum(row["catalog_match_count"] > 0 for row in label_rows),
        "products_with_same_sector_toi": sum(row["same_sector_toi_count"] > 0 for row in label_rows),
        "exact_tic_toi_event_match_count": len(event_rows),
        "same_sector_toi_event_match_count": sum(
            1 for row in event_rows if row["toi_sector_match"]
        ),
        "label_counts": dict(sorted(label_counts.items())),
        "label_status_counts": dict(sorted(status_counts.items())),
        "cross_catalog_overlap_counts": dict(sorted(overlap.items())),
        "combined_product_label_counts": dict(sorted(combined_label_counts.items())),
        "combined_product_label_status_counts": dict(sorted(combined_status_counts.items())),
        "label_policy": {
            "join": "exact TIC ID; no coordinate-radius matching",
            "sector_gate": "TOI sector list must include the light-curve product sector",
            "tess_disposition_mapping": CLASS_MAP,
            "followup_gate": "APC/FP/FA and conflicting follow-up dispositions remain unlabeled",
            "unmatched_policy": "unmatched, off-sector, ambiguous, and unsupported entries stay unlabeled",
            "feature_leakage_policy": "TOI IDs, ephemerides, catalog dispositions, and TIC parameters are not model features",
            "scope": "catalog proxy for a same-sector cataloged event; not independent source-local morphology truth or planet confirmation",
        },
        "outputs": {
            "labels_parquet": labels_path.name,
            "combined_labels_parquet": combined_path.name,
            "toi_matches_jsonl": matches_path.name,
        },
        "labels_parquet_sha256": _sha256(labels_path),
        "combined_labels_parquet_sha256": _sha256(combined_path),
        "toi_matches_jsonl_sha256": _sha256(matches_path),
        "limitations": [
            "This is a catalog disposition proxy, not independent light-curve morphology adjudication.",
            "Same-sector cataloging does not prove that the signal is on target or present in the delivered SPOC flux.",
            "A product may have separate TOI and EB catalog event proxies; preserve event identity and match detections to ephemerides before candidate-level training.",
            "Unmatched, off-sector, ambiguous, conflicting, and unsupported records remain unlabeled; absence from the TOI table is not a negative.",
            "All sectors from the same TIC share one source_group_id and must remain in one future partition.",
        ],
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    LOGGER.info("TOI proxy label counts=%s statuses=%s overlaps=%s", label_counts, status_counts, overlap)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--toi-csv", type=Path, default=DEFAULT_TOI_CSV)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    report = build_sidecars(args.data_root, args.toi_csv)
    print(json.dumps({
        "product_count": report["product_count"],
        "label_counts": report["label_counts"],
        "label_status_counts": report["label_status_counts"],
        "cross_catalog_overlap_counts": report["cross_catalog_overlap_counts"],
        "combined_product_label_counts": report["combined_product_label_counts"],
        "report": str(args.data_root / "toi_crossmatch_report.v1.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
