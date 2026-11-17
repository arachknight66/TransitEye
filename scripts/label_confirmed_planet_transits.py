"""Create event-timed transit proxy labels for confirmed-planet host light curves."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import platform
import tempfile
from collections import Counter, defaultdict
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import astropy
import pyarrow as pa
import pyarrow.parquet as pq
from astropy.io import fits

LOGGER = logging.getLogger(__name__)
DEFAULT_DATA_ROOT = Path(".transiteye-workspace/data/mast_confirmed_exoplanet_tess_v1")
JULIAN_BJD_REFERENCE = 2_457_000.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_decimal(value: Any) -> Decimal | None:
    if value is None or value == "":
        return None
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except Exception:
        return None


def _as_float(value: Any) -> float | None:
    parsed = _as_decimal(value)
    if parsed is None or not parsed.is_finite():
        return None
    result = float(parsed)
    return result if math.isfinite(result) else None


def _rounding_bound(value: Decimal | None) -> float:
    """Half of the archive value's last printed decimal place, in its stored units."""
    if value is None:
        return 0.0
    return 0.5 * (10.0 ** value.as_tuple().exponent)


def _read_interval(path: Path, tic: str, sector: int) -> tuple[float, float]:
    with fits.open(path, mode="readonly", memmap=True) as hdul:
        primary = hdul[0].header
        lightcurve = hdul["LIGHTCURVE"].header
        fits_tic = str(primary.get("TICID", lightcurve.get("TICID", ""))).strip()
        fits_sector = int(primary.get("SECTOR", lightcurve.get("SECTOR", -1)))
        if fits_tic.endswith(".0"):
            fits_tic = fits_tic[:-2]
        if fits_tic != tic:
            raise ValueError(f"FITS TIC {fits_tic} does not match manifest TIC {tic}")
        if fits_sector != sector:
            raise ValueError(f"FITS sector {fits_sector} does not match manifest sector {sector}")
        start, stop = _as_float(lightcurve.get("TSTART")), _as_float(lightcurve.get("TSTOP"))
        reference = ((_as_float(lightcurve.get("BJDREFI")) or 0.0)
                     + (_as_float(lightcurve.get("BJDREFF")) or 0.0))
        if start is None or stop is None or not math.isclose(
            reference, JULIAN_BJD_REFERENCE, abs_tol=1e-6
        ):
            raise ValueError("missing product time bounds or unexpected FITS BJD reference")
        return start, stop


def _transit_in_interval(
    period: Decimal,
    epoch_bjd: Decimal,
    duration_hours: Decimal,
    interval_start_bjd_minus_2457000: float,
    interval_stop_bjd_minus_2457000: float,
) -> tuple[bool, float | None, int | None, float | None]:
    period_days = float(period)
    epoch = float(epoch_bjd - Decimal(str(JULIAN_BJD_REFERENCE)))
    duration_days = float(duration_hours) / 24.0
    if period_days <= 0 or duration_days <= 0:
        return False, None, None, None

    start = interval_start_bjd_minus_2457000
    stop = interval_stop_bjd_minus_2457000
    cycle = math.floor((start - epoch) / period_days) - 1
    period_bound = _rounding_bound(period)
    epoch_bound = _rounding_bound(epoch_bjd)
    half_duration = duration_days / 2.0
    while True:
        event_time = epoch + cycle * period_days
        timing_bound = epoch_bound + abs(cycle) * period_bound
        if event_time - half_duration - timing_bound > stop:
            break
        if event_time + half_duration + timing_bound >= start:
            return True, event_time, cycle, timing_bound
        cycle += 1
    return False, None, None, None


def _write_parquet_atomic(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, filename = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(filename)
    try:
        pq.write_table(table, temporary, compression="zstd")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, filename = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(filename)
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_sidecars(data_root: Path) -> dict[str, Any]:
    manifest_path = data_root / "products.v1.json"
    acquisition_path = data_root / "acquisition_report.v1.json"
    catalog_path = data_root / "catalogs" / "confirmed_planets_pscomppars.v1.json"
    for path in (manifest_path, acquisition_path, catalog_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    acquisition = json.loads(acquisition_path.read_text(encoding="utf-8"))
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"), parse_float=Decimal)
    entries: list[dict[str, Any]] = manifest["entries"]
    if len(entries) != manifest.get("product_count"):
        raise ValueError("Confirmed-host product manifest count is inconsistent")
    if acquisition.get("completed_product_count") != len(entries):
        raise ValueError("Acquisition report does not show every manifest product completed")
    if any(item.get("status") not in {"downloaded", "verified_existing"}
           for item in acquisition.get("results", {}).values()):
        raise ValueError("Acquisition report includes missing or failed products")
    if any(item.get("morphology_label") is not None for item in entries):
        raise ValueError("Refusing to mutate or relabel the frozen product manifest")

    planets_by_tic: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_planets_by_tic: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for planet in catalog:
        tic_text = str(planet.get("tic_id") or "").strip()
        if not tic_text:
            continue
        tic = tic_text.removeprefix("TIC ").strip()
        if not tic.isdigit():
            continue
        tic = str(int(tic))
        all_planets_by_tic[tic].append(planet)
        if str(planet.get("discoverymethod") or "").strip().lower() == "transit":
            planets_by_tic[tic].append(planet)

    labels: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    fit_errors: Counter[str] = Counter()
    planet_event_match_counts: Counter[str] = Counter()

    for entry in entries:
        tic = str(int(str(entry["target_id"])))
        sector = int(entry["sector"])
        product_id = str(entry["product_id"])
        product = data_root / "curves" / str(entry["local_filename"])
        acquisition_row = acquisition["results"].get(product_id, {})
        transit_planets = planets_by_tic.get(tic, [])
        manifest_transiting_names = set(entry.get("transiting_planet_names") or [])
        catalog_transiting_names = {
            str(planet.get("pl_name")) for planet in transit_planets if planet.get("pl_name")
        }
        if manifest_transiting_names != catalog_transiting_names:
            raise ValueError(f"Frozen planet-to-TIC association differs for TIC {tic}")

        start: float | None = None
        stop: float | None = None
        identity_status = "verified"
        try:
            if not product.is_file():
                raise FileNotFoundError(product.name)
            start, stop = _read_interval(product, tic, sector)
        except (OSError, ValueError, KeyError) as exc:
            identity_status = f"unavailable_or_invalid:{type(exc).__name__}"
            fit_errors[type(exc).__name__] += 1
            LOGGER.warning("Leaving %s unlabeled: %s", product_id, exc)

        event_supports: list[dict[str, Any]] = []
        for planet in transit_planets:
            period = _as_decimal(planet.get("pl_orbper"))
            epoch = _as_decimal(planet.get("pl_tranmid"))
            duration = _as_decimal(planet.get("pl_trandur"))
            if period is None or epoch is None or duration is None:
                event_status = "unlabeled_transit_ephemeris_incomplete"
                event_time = None
                cycle = None
                rounding_bound = None
            elif start is None or stop is None:
                event_status = "unlabeled_product_interval_unavailable"
                event_time = None
                cycle = None
                rounding_bound = None
            else:
                covered, event_time, cycle, rounding_bound = _transit_in_interval(
                    period, epoch, duration, start, stop
                )
                event_status = (
                    "transit_window_overlaps_product_interval"
                    if covered else "transit_ephemeris_outside_product_interval"
                )
                if covered:
                    name = str(planet.get("pl_name") or "")
                    event_supports.append({
                        "planet_name": name,
                        "predicted_transit_bjd_minus_2457000": event_time,
                        "cycle_number": cycle,
                        "published_value_rounding_bound_days": rounding_bound,
                    })
                    planet_event_match_counts[name] += 1
            event_rows.append({
                "product_id": product_id,
                "target_id": tic,
                "source_group_id": entry["source_group_id"],
                "sector": sector,
                "planet_name": planet.get("pl_name"),
                "host_name": planet.get("hostname"),
                "discovery_method": planet.get("discoverymethod"),
                "period_days": float(period) if period is not None else None,
                "transit_epoch_bjd": float(epoch) if epoch is not None else None,
                "transit_duration_hours": float(duration) if duration is not None else None,
                "product_tstart_bjd_minus_2457000": start,
                "product_tstop_bjd_minus_2457000": stop,
                "predicted_transit_bjd_minus_2457000": event_time,
                "predicted_cycle_number": cycle,
                "published_value_rounding_bound_days": rounding_bound,
                "event_match_status": event_status,
            })

        if identity_status != "verified":
            label = None
            label_status = "unlabeled_product_identity_or_interval_invalid"
        elif not transit_planets:
            label = None
            label_status = "unlabeled_host_has_no_confirmed_transiting_planet"
        elif event_supports:
            label = "transit_like"
            label_status = "confirmed_transit_ephemeris_in_product_interval"
        elif any(_as_decimal(p.get("pl_orbper")) is None
                 or _as_decimal(p.get("pl_tranmid")) is None
                 or _as_decimal(p.get("pl_trandur")) is None for p in transit_planets):
            label = None
            label_status = "unlabeled_transit_ephemeris_incomplete"
        else:
            label = None
            label_status = "unlabeled_confirmed_transit_outside_product_interval"
        status_counts[label_status] += 1
        labels.append({
            "product_id": product_id,
            "target_id": tic,
            "source_group_id": str(entry["source_group_id"]),
            "sector": sector,
            "product_sha256": acquisition_row.get("sha256"),
            "label": label,
            "label_status": label_status,
            "label_scope": "confirmed-planet event proxy; transit ephemeris window intersects this host-sector product interval",
            "confirmed_planet_count": int(entry.get("confirmed_planet_count", 0)),
            "transiting_planet_count": len(transit_planets),
            "transiting_planet_names_json": json.dumps(sorted(manifest_transiting_names)),
            "matched_transit_planets_json": json.dumps(event_supports, sort_keys=True),
            "product_tstart_bjd_minus_2457000": start,
            "product_tstop_bjd_minus_2457000": stop,
            "fits_identity_status": identity_status,
        })

    label_path = data_root / "labels_confirmed_transit_proxy.v1.parquet"
    events_path = data_root / "confirmed_transit_event_matches.v1.jsonl"
    report_path = data_root / "confirmed_transit_label_report.v1.json"
    schema = pa.schema([
        ("product_id", pa.string()), ("target_id", pa.string()),
        ("source_group_id", pa.string()), ("sector", pa.int16()),
        ("product_sha256", pa.string()), ("label", pa.string()),
        ("label_status", pa.string()), ("label_scope", pa.string()),
        ("confirmed_planet_count", pa.int16()), ("transiting_planet_count", pa.int16()),
        ("transiting_planet_names_json", pa.string()), ("matched_transit_planets_json", pa.string()),
        ("product_tstart_bjd_minus_2457000", pa.float64()),
        ("product_tstop_bjd_minus_2457000", pa.float64()), ("fits_identity_status", pa.string()),
    ])
    table = pa.Table.from_pylist(labels, schema=schema)
    _write_parquet_atomic(label_path, table)
    _write_jsonl_atomic(events_path, event_rows)

    report = {
        "schema_version": "1",
        "dataset_id": manifest["dataset_id"],
        "label_policy_id": "confirmed_planet_same_host_transit_ephemeris_proxy.v1",
        "label_snapshot_id": hashlib.sha256(
            f"{manifest['manifest_id']}:{_sha256(catalog_path)}:confirmed-transit-proxy-v1".encode()
        ).hexdigest()[:16],
        "created_at_utc": datetime.now(UTC).isoformat(),
        "product_manifest_id": manifest["manifest_id"],
        "product_manifest_sha256": _sha256(manifest_path),
        "acquisition_report_sha256": _sha256(acquisition_path),
        "confirmed_planet_catalog_file": str(catalog_path.relative_to(data_root)),
        "confirmed_planet_catalog_sha256": _sha256(catalog_path),
        "archive_query_report_sha256": _sha256(data_root / "archive_query_report.v1.json"),
        "label_script_sha256": _sha256(Path(__file__).resolve()),
        "python_version": platform.python_version(),
        "astropy_version": astropy.__version__,
        "pyarrow_version": pa.__version__,
        "confirmed_planet_record_count": len(catalog),
        "confirmed_planet_records_with_tic_count": sum(bool(p.get("tic_id")) for p in catalog),
        "confirmed_transiting_planet_record_count": sum(
            str(p.get("discoverymethod") or "").strip().lower() == "transit" for p in catalog
        ),
        "transiting_planet_records_with_period_epoch_duration_count": sum(
            str(p.get("discoverymethod") or "").strip().lower() == "transit"
            and p.get("pl_orbper") is not None and p.get("pl_tranmid") is not None
            and p.get("pl_trandur") is not None for p in catalog
        ),
        "product_count": len(labels),
        "distinct_host_tic_count": len({row["target_id"] for row in labels}),
        "labeled_product_count": sum(row["label"] == "transit_like" for row in labels),
        "distinct_labeled_host_tic_count": len({row["target_id"] for row in labels
                                                  if row["label"] == "transit_like"}),
        "event_match_row_count": len(event_rows),
        "event_support_count": sum(row["event_match_status"] == "transit_window_overlaps_product_interval"
                                    for row in event_rows),
        "label_status_counts": dict(sorted(status_counts.items())),
        "fits_identity_error_count": sum(fit_errors.values()),
        "fits_identity_errors": dict(sorted(fit_errors.items())),
        "ephemeris_timing_policy": "Compare the frozen NASA Exoplanet Archive transit epoch and period (BJD, days) and duration (hours) with each SPOC product's BJD-2457000 TSTART/TSTOP. Use half-duration plus propagated printed-value rounding bounds; this snapshot has no period/epoch uncertainty columns.",
        "outputs": {"labels_parquet": label_path.name, "event_matches_jsonl": events_path.name},
        "limitations": [
            "Confirmed-host identity alone does not assign a label; only discoverymethod=Transit records with period, epoch, duration, and a predicted event window overlapping the product interval support transit_like.",
            "This pass checks product interval and FITS TIC/sector identity, not the observed flux shape, signal depth, or source localization.",
            "The frozen pscomppars snapshot does not contain transit epoch/period uncertainties; timing allowance uses only the published numeric precision and transit duration and may understate ephemeris drift.",
            "A catalog proxy is not independent morphology truth or confirmation from this SPOC curve. Unmatched and non-overlapping products remain unlabeled, never negative.",
            "All sectors and planets from one host TIC share a source_group_id and must stay in one partition.",
        ],
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    LOGGER.info("Confirmed transit proxy counts=%s statuses=%s", report["labeled_product_count"], status_counts)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(message)s")
    report = build_sidecars(args.data_root)
    print(json.dumps({
        "product_count": report["product_count"],
        "labeled_product_count": report["labeled_product_count"],
        "distinct_labeled_host_tic_count": report["distinct_labeled_host_tic_count"],
        "label_status_counts": report["label_status_counts"],
        "report": str(args.data_root / "confirmed_transit_label_report.v1.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
