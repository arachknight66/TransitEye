"""Freeze a source-grouped candidate sequence cohort from catalog event sidecars.

Preparation reads manifests and catalog sidecars only. It never opens FITS flux data,
so the fresh holdout assignment is fixed before any scientific observation is read.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

CLASSES = ("transit_like", "eclipse_like")
JULIAN_BJD_REFERENCE = 2_457_000.0
SEED = 42
SPLIT_FRACTIONS = {
    "train": 0.70,
    "calibration": 0.10,
    "selection": 0.10,
    "fresh_holdout": 0.10,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _normalise_class(value: Any) -> str | None:
    return str(value) if value in CLASSES else None


def _valid_event(
    *,
    catalog: str,
    record_id: str,
    period: Any,
    epoch_bjd: Any,
    duration_hours: Any,
    epoch_is_btjd: bool = False,
) -> dict[str, Any] | None:
    try:
        parsed_period = float(period)
        parsed_epoch = float(epoch_bjd)
        parsed_duration = float(duration_hours) if duration_hours not in (None, "") else None
    except (TypeError, ValueError):
        return None
    if parsed_period <= 0 or not math.isfinite(parsed_period) or not math.isfinite(parsed_epoch):
        return None
    if parsed_duration is not None:
        if not math.isfinite(parsed_duration) or parsed_duration <= 0 or parsed_duration > 48:
            parsed_duration = None
    epoch_btjd = parsed_epoch if epoch_is_btjd else parsed_epoch - JULIAN_BJD_REFERENCE
    return {
        "catalog": catalog,
        "record_id": str(record_id),
        "period_days": parsed_period,
        "epoch_days_bjd_minus_2457000": epoch_btjd,
        "duration_hours": parsed_duration,
    }


def _manifest_entries(data_root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    manifest_path = data_root / "products.v1.json"
    acquisition_path = data_root / "acquisition_report.v1.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    acquisition = json.loads(acquisition_path.read_text(encoding="utf-8"))
    acquisition_results = acquisition.get("results", {})
    entries: dict[str, dict[str, Any]] = {}
    for entry in manifest["entries"]:
        product_id = str(entry["product_id"])
        if product_id in entries:
            raise ValueError(f"Duplicate product ID in frozen manifest: {product_id}")
        result = acquisition_results.get(product_id, {})
        product_sha = (
            entry.get("expected_sha256")
            or entry.get("sha256")
            or result.get("sha256")
        )
        entries[product_id] = {
            "product_id": product_id,
            "target_id": str(entry["target_id"]).removesuffix(".0"),
            "source_group_id": f"tic-{str(entry['target_id']).removesuffix('.0')}",
            "sector": int(entry["sector"]),
            "local_filename": str(entry["local_filename"]),
            "expected_sha256": product_sha,
            "data_root": str(data_root),
            "manifest_id": manifest.get("manifest_id"),
        }
    if len(entries) != len(manifest["entries"]):
        raise ValueError(f"Manifest entry count mismatch: {manifest_path}")
    return entries, {
        "products_manifest_sha256": sha256_file(manifest_path),
        "acquisition_report_sha256": sha256_file(acquisition_path),
    }


def _add_event(
    event_map: dict[str, list[dict[str, Any]]],
    product_id: str,
    event: dict[str, Any] | None,
) -> None:
    if event is not None:
        event_map[product_id].append(event)


def _load_cohort(
    *,
    cohort_name: str,
    data_root: Path,
    labels_file: str,
    label_column: str,
    event_sources: list[tuple[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    products, hashes = _manifest_entries(data_root)
    label_path = data_root / labels_file
    hashes["label_sidecar_sha256"] = sha256_file(label_path)
    label_rows = pq.read_table(label_path).to_pylist()
    events_by_product: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event_file, event_builder in event_sources:
        event_path = data_root / event_file
        hashes[f"{event_file}_sha256"] = sha256_file(event_path)
        for row in _read_jsonl(event_path):
            event_builder(row, events_by_product)

    cohort_records: list[dict[str, Any]] = []
    for row in label_rows:
        label = _normalise_class(row.get(label_column))
        if label is None:
            continue
        product_id = str(row["product_id"])
        entry = products.get(product_id)
        if entry is None:
            raise ValueError(f"Label row absent from frozen product manifest: {product_id}")
        target_id = str(row["target_id"]).removesuffix(".0")
        if target_id != entry["target_id"]:
            raise ValueError(f"TIC mismatch for {product_id}: label={target_id}, product={entry['target_id']}")
        label_group = str(row.get("source_group_id") or entry["source_group_id"])
        if label_group not in {entry["source_group_id"], f"TIC {target_id}", target_id}:
            raise ValueError(f"Source-group mismatch for {product_id}: {label_group}")
        refs = events_by_product.get(product_id, [])
        matching_refs = [item for item in refs if item["class_label"] == label]
        cohort_records.append({
            **entry,
            "cohort": cohort_name,
            "label": label,
            "label_status": row.get("label_status"),
            "label_sidecar_scope": row.get("label_scope"),
            "event_refs": [item["event"] for item in matching_refs],
        })
    return cohort_records, hashes


def _transit_event_builder(
    source: str,
    product_id_key: str,
    period_key: str,
    epoch_key: str,
    duration_key: str,
    *,
    epoch_is_btjd: bool = False,
):
    def build(row: dict[str, Any], output: dict[str, list[dict[str, Any]]]) -> None:
        if row.get("event_match_status") != "transit_window_overlaps_product_interval":
            return
        event = _valid_event(
            catalog=source,
            record_id=row.get("planet_name") or row.get("toi_id") or "unknown",
            period=row.get(period_key),
            epoch_bjd=row.get(epoch_key),
            duration_hours=row.get(duration_key),
            epoch_is_btjd=epoch_is_btjd,
        )
        if event:
            _add_event(output, str(row[product_id_key]), {"class_label": "transit_like", "event": event})
    return build


def _eb_event_builder(row: dict[str, Any], output: dict[str, list[dict[str, Any]]]) -> None:
    if row.get("event_match_status") != "primary_eclipse_ephemeris_overlaps_product_interval":
        return
    event = _valid_event(
        catalog=f"eb:{row.get('catalog', 'catalog')}",
        record_id=row.get("catalog_record_id") or "unknown",
        period=row.get("catalog_period_days"),
        epoch_bjd=row.get("catalog_primary_epoch_bjd_minus_2457000"),
        duration_hours=row.get("catalog_primary_duration_hours"),
        epoch_is_btjd=True,
    )
    if event:
        _add_event(output, str(row["product_id"]), {"class_label": "eclipse_like", "event": event})


def _toi_event_builder(row: dict[str, Any], output: dict[str, list[dict[str, Any]]]) -> None:
    mapped_label = row.get("mapped_label")
    if mapped_label is None:
        tess = str(row.get("tess_disposition") or "").strip().upper()
        followup = str(row.get("tfopwg_disposition") or "").strip().upper()
        planet_dispositions = {"PC", "KP", "CP"}
        if tess in planet_dispositions and (not followup or followup in planet_dispositions):
            mapped_label = "transit_like"
    if not row.get("toi_sector_match") or mapped_label != "transit_like":
        return
    event = _valid_event(
        catalog="ExoFOP-TESS TOI",
        record_id=row.get("toi_id") or "unknown",
        period=row.get("toi_period_days"),
        epoch_bjd=row.get("toi_epoch_bjd"),
        duration_hours=row.get("toi_duration_hours"),
    )
    if event:
        _add_event(output, str(row["product_id"]), {"class_label": "transit_like", "event": event})


def _prior_partitions(repo_root: Path) -> tuple[dict[str, set[str]], list[dict[str, str]]]:
    assignments: dict[str, set[str]] = defaultdict(set)
    source_files: list[dict[str, str]] = []
    manifest_dir = repo_root / "manifests"
    for path in sorted(manifest_dir.glob("*split*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        for row in raw.get("assignments", []):
            group = str(row["source_group_id"]).removeprefix("tic-")
            assignments[group].add(str(row["split"]))
        source_files.append({"path": str(path), "sha256": sha256_file(path)})
    remediation = repo_root / ".transiteye-workspace" / "remediation"
    if remediation.exists():
        for path in sorted(remediation.rglob("*.json")):
            name = path.name.lower()
            if not any(token in name for token in ("manifest", "partition", "split", "selection")):
                continue
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                continue
            rows = raw.get("assignments") or raw.get("records") or []
            found = False
            for row in rows:
                group = row.get("source_group_id") or row.get("group") or row.get("tic_id")
                part = row.get("split") or row.get("partition")
                if group and part:
                    assignments[str(group).removeprefix("tic-")].add(str(part))
                    found = True
            if found:
                source_files.append({"path": str(path), "sha256": sha256_file(path)})
    return assignments, source_files


def _stable_partition(groups: list[str], label: str) -> dict[str, str]:
    ordered = sorted(groups, key=lambda group: hashlib.sha256(
        f"transiteye-labeled-sequence-v1:{SEED}:{label}:{group}".encode()
    ).hexdigest())
    n = len(ordered)
    ideal = {name: n * fraction for name, fraction in SPLIT_FRACTIONS.items()}
    counts = {name: int(value) for name, value in ideal.items()}
    remaining = n - sum(counts.values())
    remainder_order = sorted(ideal, key=lambda name: (-(ideal[name] - counts[name]), name))
    for name in remainder_order[:remaining]:
        counts[name] += 1
    output: dict[str, str] = {}
    cursor = 0
    for name in SPLIT_FRACTIONS:
        for group in ordered[cursor:cursor + counts[name]]:
            output[group] = name
        cursor += counts[name]
    return output


def prepare_cohort(repo_root: Path, output_dir: Path) -> dict[str, Any]:
    """Audit sidecars and freeze assignments without opening FITS observations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "cohort_manifest.json"
    if output_path.exists():
        raise FileExistsError(f"Refusing to replace frozen cohort manifest: {output_path}")
    data_root = repo_root / ".transiteye-workspace" / "data"
    specifications = [
        {
            "name": "confirmed_planet_hosts",
            "root": data_root / "mast_confirmed_exoplanet_tess_v1",
            "labels": "labels_confirmed_transit_proxy.v1.parquet",
            "label_column": "label",
            "event_sources": [(
                "confirmed_transit_event_matches.v1.jsonl",
                _transit_event_builder(
                    "NASA Exoplanet Archive pscomppars", "product_id",
                    "period_days", "transit_epoch_bjd", "transit_duration_hours"
                ),
            )],
        },
        {
            "name": "eb_catalog_toi_combined",
            "root": data_root / "mast_eclipsing_binary_catalogs_v1",
            "labels": "labels_eb_toi_catalog_proxy.v1.parquet",
            "label_column": "label",
            "event_sources": [
                ("eb_catalog_event_matches.v1.jsonl", _eb_event_builder),
                ("toi_matches.v1.jsonl", _toi_event_builder),
            ],
        },
        {
            "name": "fifty_thousand_toi_overlay",
            "root": data_root / "mast_unlabeled_50000_v1",
            "labels": "labels_toi_proxy.v1.parquet",
            "label_column": "label",
            "event_sources": [("toi_matches.v1.jsonl", _toi_event_builder)],
        },
    ]
    all_records: list[dict[str, Any]] = []
    input_hashes: dict[str, Any] = {}
    cohort_counts: dict[str, Any] = {}
    for spec in specifications:
        required_paths = [
            spec["root"] / name
            for name in ("products.v1.json", "acquisition_report.v1.json", spec["labels"])
        ] + [spec["root"] / name for name, _ in spec["event_sources"]]
        if not all(path.is_file() for path in required_paths):
            raise FileNotFoundError(f"Incomplete labeled cohort inputs: {spec['name']}")
        records, hashes = _load_cohort(
            cohort_name=spec["name"],
            data_root=spec["root"],
            labels_file=spec["labels"],
            label_column=spec["label_column"],
            event_sources=spec["event_sources"],
        )
        all_records.extend(records)
        input_hashes[spec["name"]] = hashes
        cohort_counts[spec["name"]] = {
            "labeled_product_rows": len(records),
            "class_counts": dict(Counter(row["label"] for row in records)),
            "products_with_ephemeris_records": sum(bool(row["event_refs"]) for row in records),
            "missing_ephemeris_support": sum(not row["event_refs"] for row in records),
        }

    product_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_records:
        product_rows[row["product_id"]].append(row)
    product_conflicts: list[dict[str, Any]] = []
    merged_products: list[dict[str, Any]] = []
    for product_id, copies in sorted(product_rows.items()):
        labels = {row["label"] for row in copies}
        tic_ids = {row["target_id"] for row in copies}
        hashes = {row["expected_sha256"] for row in copies if row["expected_sha256"]}
        if len(labels) != 1 or len(tic_ids) != 1 or len(hashes) > 1:
            product_conflicts.append({
                "product_id": product_id,
                "reason": "label_conflict" if len(labels) > 1 else "identity_or_checksum_conflict",
                "labels": sorted(labels),
                "target_ids": sorted(tic_ids),
                "hashes": sorted(hashes),
            })
            continue
        primary = sorted(copies, key=lambda row: row["cohort"])[0].copy()
        refs: dict[str, dict[str, Any]] = {}
        for row in copies:
            for event in row["event_refs"]:
                key = json.dumps(event, sort_keys=True, separators=(",", ":"))
                refs[key] = event
        primary["cohorts"] = sorted({row["cohort"] for row in copies})
        primary["event_refs"] = [refs[key] for key in sorted(refs)]
        if not primary["expected_sha256"]:
            product_conflicts.append({"product_id": product_id, "reason": "missing_product_sha256"})
            continue
        if not primary["event_refs"]:
            product_conflicts.append({"product_id": product_id, "reason": "no_valid_event_ephemeris"})
            continue
        merged_products.append(primary)

    tic_labels: dict[str, set[str]] = defaultdict(set)
    for row in merged_products:
        tic_labels[row["target_id"]].add(row["label"])
    conflicting_tics = {
        tic: labels for tic, labels in tic_labels.items() if len(labels) != 1
    }
    source_conflicts = [
        {"target_id": tic, "labels": sorted(labels), "product_ids": sorted(
            row["product_id"] for row in merged_products if row["target_id"] == tic
        )}
        for tic, labels in sorted(conflicting_tics.items())
    ]
    conflict_tics = set(conflicting_tics)
    merged_products = [row for row in merged_products if row["target_id"] not in conflict_tics]

    prior, prior_files = _prior_partitions(repo_root)
    groups_by_class: dict[str, set[str]] = {label: set() for label in CLASSES}
    for row in merged_products:
        groups_by_class[row["label"]].add(row["source_group_id"])
    excluded_prior: dict[str, list[str]] = {}
    partition_by_group: dict[str, str] = {}
    new_groups_by_class: dict[str, list[str]] = {label: [] for label in CLASSES}
    prior_train_groups: set[str] = set()
    for label, groups in groups_by_class.items():
        for group in sorted(groups):
            history = prior.get(group.removeprefix("tic-"), set())
            if history and history <= {"train"}:
                partition_by_group[group] = "train"
                prior_train_groups.add(group)
            elif history:
                excluded_prior[group] = sorted(history)
            else:
                new_groups_by_class[label].append(group)
        partition_by_group.update(_stable_partition(new_groups_by_class[label], label))

    retained_groups = {group for group in partition_by_group}
    merged_products = [row for row in merged_products if row["source_group_id"] in retained_groups]
    for row in merged_products:
        row["split"] = partition_by_group[row["source_group_id"]]
        row["source_group_id"] = f"tic-{row['target_id']}"
        row["relative_path"] = str(Path(row.pop("data_root")) / "curves" / row.pop("local_filename"))

    partition_counts = {
        name: {
            "source_groups": sum(part == name for part in partition_by_group.values()),
            "products": sum(row["split"] == name for row in merged_products),
            "classes": dict(Counter(row["label"] for row in merged_products if row["split"] == name)),
        }
        for name in (*SPLIT_FRACTIONS.keys(),)
    }
    label_by_group = {row["source_group_id"]: row["label"] for row in merged_products}
    retained_class_groups = {
        label: sum(value == label for value in label_by_group.values())
        for label in CLASSES
    }
    expected_group_count = len({row["source_group_id"] for row in merged_products})
    if expected_group_count != sum(value["source_groups"] for value in partition_counts.values()):
        raise RuntimeError("Frozen partition counts do not reconcile to retained source groups")

    manifest = {
        "schema_version": 1,
        "protocol_id": "labeled_catalog_event_candidate_sequence_binary_v1",
        "seed": SEED,
        "class_order": list(CLASSES),
        "feature_leakage_policy": "Catalog IDs, dispositions, ephemerides, label source, and split metadata are not model inputs; ephemerides are used only for candidate-match selection and sequence alignment.",
        "test_access_policy": "fresh_holdout FITS observations remain unopened through fitting and model selection",
        "input_hashes": input_hashes,
        "prior_split_manifests": prior_files,
        "prior_nontraining_tic_count": sum(bool(parts - {"train"}) for parts in prior.values()),
        "excluded_prior_partition_group_count": len(excluded_prior),
        "excluded_prior_partition_groups": excluded_prior,
        "excluded_product_conflict_count": len(product_conflicts),
        "excluded_product_conflicts": product_conflicts,
        "excluded_source_conflict_count": len(source_conflicts),
        "excluded_source_conflicts": source_conflicts,
        "prior_training_group_count": len(prior_train_groups),
        "cohort_counts": cohort_counts,
        "partition_counts": partition_counts,
        "retained_labeled_product_count": len(merged_products),
        "retained_labeled_source_count": expected_group_count,
        "retained_source_class_counts": retained_class_groups,
        "candidate_match_required": True,
        "unlabeled_and_unmatched_are_negatives": False,
        "products": merged_products,
    }
    temporary = output_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    manifest["manifest_sha256"] = sha256_file(output_path)
    report_path = output_dir / "cohort_audit.json"
    report_path.write_text(json.dumps({
        key: value for key, value in manifest.items() if key != "products"
    } | {"cohort_manifest_sha256": manifest["manifest_sha256"]}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
