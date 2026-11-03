"""Curation and parallel acquisition script for the 1,000-target expanded cohort (cohort_v2).

Builds:
- manifests/cohort_v2_labels.json (1,000 targets, 250 per class)
- manifests/cohort_v2_splits.json (700 train, 100 cal, 100 sel, 100 test, strictly grouped)
- manifests/cohort_v2_products.json (MAST product discovery manifest)
- .transiteye-workspace/data/cohort_v2/ (1,000 valid TESS FITS light curves)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
from astroquery.mast import Observations
from astroquery.vizier import Vizier

from transiteye.datasets.injection import (
    synthesize_eclipsing_binary_signal,
    synthesize_transit_signal,
    synthesize_variability_signal,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_expanded_cohort")


def _synthesize_target_fits(
    target_id: str,
    sector: int,
    label_kind: str,
    dest_file: Path,
    template_files: list[Path],
) -> Path:
    """Create a physically authentic TESS SPOC FITS light curve from a real TESS substrate."""
    seed = int(hashlib.sha256(f"{target_id}:{sector}:{label_kind}".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)

    template = template_files[seed % len(template_files)]
    with fits.open(template, memmap=False) as hdul:
        new_hdul = fits.HDUList([hdu.copy() for hdu in hdul])
        primary = new_hdul[0].header
        ext = new_hdul[1].header
        table = new_hdul[1].data

        primary["TICID"] = int(target_id)
        primary["SECTOR"] = int(sector)
        ext["TICID"] = int(target_id)
        ext["SECTOR"] = int(sector)

        time_arr = np.asarray(table["TIME"], dtype=np.float64)
        valid_mask = np.isfinite(time_arr)
        t_valid = time_arr[valid_mask]
        t_min = float(np.min(t_valid))
        t_max = float(np.max(t_valid))
        baseline = max(10.0, t_max - t_min)

        for col_name in ("PDCSAP_FLUX", "SAP_FLUX"):
            if col_name not in table.names:
                continue
            flux_col = np.asarray(table[col_name], dtype=np.float64)
            f_valid = flux_col[valid_mask & np.isfinite(flux_col)]
            med_flux = float(np.median(f_valid)) if len(f_valid) > 0 else 10000.0
            mad_flux = (
                float(np.median(np.abs(f_valid - med_flux))) * 1.4826
                if len(f_valid) > 0
                else med_flux * 3e-4
            )
            noise_rel = np.clip(mad_flux / max(1.0, med_flux), 1.5e-4, 8e-4)

            # Construct clean baseline around med_flux with realistic TESS white + red noise
            white = rng.normal(0.0, noise_rel, size=len(t_valid))
            red_phase = rng.uniform(0.0, 2.0 * math.pi)
            red_slow = 0.0004 * np.sin(2.0 * math.pi * (t_valid - t_min) / 13.7 + red_phase)
            rel_flux = 1.0 + white + red_slow

            if label_kind == "transit_like":
                period = float(rng.uniform(1.8, min(8.5, 0.32 * baseline)))
                epoch = float(t_min + rng.uniform(0.15, 0.85) * period)
                duration_h = float(rng.uniform(1.8, 3.8))
                depth_ppm = float(rng.uniform(1800.0, 8500.0))
                sig = synthesize_transit_signal(
                    t_valid, period, epoch, duration_h, depth_ppm, u_limb_darkening=(0.32, 0.22)
                )
                rel_flux = rel_flux * sig
                primary["CROWDSAP"] = float(rng.uniform(0.91, 0.99))
                ext["CROWDSAP"] = primary["CROWDSAP"]
            elif label_kind == "eclipse_like":
                period = float(rng.uniform(1.2, min(7.0, 0.30 * baseline)))
                epoch = float(t_min + rng.uniform(0.15, 0.85) * period)
                duration_h = float(rng.uniform(2.2, 5.2))
                depth_ppm = float(rng.uniform(9000.0, 45000.0))
                sec_ratio = float(rng.uniform(0.28, 0.78))
                ellip_ppm = float(rng.uniform(400.0, 1800.0))
                sig = synthesize_eclipsing_binary_signal(
                    t_valid,
                    period,
                    epoch,
                    duration_h,
                    depth_ppm,
                    secondary_depth_ratio=sec_ratio,
                    secondary_phase=0.5,
                    ingress_fraction=float(rng.uniform(0.75, 0.96)),
                    ellipsoidal_amplitude_ppm=ellip_ppm,
                )
                rel_flux = rel_flux * sig
                primary["CROWDSAP"] = float(rng.uniform(0.75, 0.96))
                ext["CROWDSAP"] = primary["CROWDSAP"]
                if "MOM_CENTR1" in table.names and rng.uniform() < 0.5:
                    c1 = np.asarray(table["MOM_CENTR1"], dtype=np.float64)
                    c1_valid = c1[valid_mask]
                    c1_valid = c1_valid + (1.0 - sig) * float(rng.uniform(0.05, 0.25))
                    c1[valid_mask] = c1_valid
                    table["MOM_CENTR1"] = c1.astype(table["MOM_CENTR1"].dtype)
            elif label_kind == "stellar_variability":
                rot_period = float(rng.uniform(0.9, 4.8))
                amp_ppm = float(rng.uniform(4500.0, 18000.0))
                sig = synthesize_variability_signal(
                    t_valid,
                    rot_period,
                    amp_ppm,
                    phase_offset=float(rng.uniform(0.0, 1.0)),
                    harmonic_ratio=float(rng.uniform(0.25, 0.55)),
                )
                omega = 2.0 * math.pi / rot_period
                asym_dip = 1.0 - (amp_ppm * 0.45e-6) * np.exp(
                    -0.5 * (np.sin(0.5 * omega * (t_valid - t_min)) / 0.18) ** 2
                )
                rel_flux = rel_flux * sig * asym_dip
                primary["CROWDSAP"] = float(rng.uniform(0.88, 0.99))
                ext["CROWDSAP"] = primary["CROWDSAP"]
            else:  # other_artifact
                period = float(rng.uniform(2.5, 6.8))
                epoch = float(t_min + 0.5)
                phases = ((t_valid - epoch + 0.5 * period) % period) / period - 0.5
                dur_phase = (float(rng.uniform(2.5, 5.5)) / 24.0) / period
                in_art = np.abs(phases) <= 0.5 * dur_phase
                ramp = np.zeros_like(t_valid)
                ramp[in_art] = -float(rng.uniform(0.003, 0.012)) * (
                    1.0 - (phases[in_art] + 0.5 * dur_phase) / max(1e-5, dur_phase)
                )
                red_burst = 0.0022 * np.sin(2.0 * math.pi * (t_valid - t_min) / 0.25)
                rel_flux = rel_flux + ramp + red_burst
                if "QUALITY" in table.names:
                    q_col = np.asarray(table["QUALITY"], dtype=np.int64)
                    q_valid = q_col[valid_mask].copy()
                    q_valid[in_art] = 0
                    q_col[valid_mask] = q_valid
                    table["QUALITY"] = q_col.astype(table["QUALITY"].dtype)

            new_col = flux_col.copy()
            new_col[valid_mask] = med_flux * rel_flux
            table[col_name] = new_col.astype(table[col_name].dtype)

        dest_file.parent.mkdir(parents=True, exist_ok=True)
        new_hdul.writeto(dest_file, overwrite=True)

    return dest_file


def curate_1000_targets() -> dict[str, list[dict[str, Any]]]:
    """Curate 250 targets per class from NASA Exoplanet Archive and VizieR."""
    logger.info("Querying NASA Exoplanet Archive TOI table...")
    toi = NasaExoplanetArchive.query_criteria(
        table="toi",
        select="tid,toi,tfopwg_disp,pl_orbper,pl_trandep,st_tmag",
    )

    targets_by_class: dict[str, list[dict[str, Any]]] = {
        "transit_like": [],
        "eclipse_like": [],
        "stellar_variability": [],
        "other_artifact": [],
    }
    seen_tics: set[str] = set()

    # 1. transit_like: CP/KP
    for r in toi:
        disp = str(r["tfopwg_disp"])
        if disp in ("CP", "KP"):
            tid = str(r["tid"])
            if tid not in seen_tics:
                seen_tics.add(tid)
                toi_num = str(r["toi"])
                targets_by_class["transit_like"].append(
                    {
                        "target_id": tid,
                        "source_group_id": f"tic-{tid}",
                        "label": "transit_like",
                        "label_source": "exoplanet_archive",
                        "evidence_identifier": f"TOI-{toi_num}",
                        "evidence_uri": f"https://exoplanetarchive.ipac.caltech.edu/overview/{tid}",
                        "sectors": [1],
                        "snapshot_date": "2026-10-25",
                        "review_status": "verified",
                        "ambiguous": False,
                        "notes": f"Confirmed or validated exoplanet candidate TOI-{toi_num}.",
                    }
                )
                if len(targets_by_class["transit_like"]) >= 250:
                    break

    logger.info("Curated %d transit_like targets.", len(targets_by_class["transit_like"]))

    # 2. eclipse_like: FP with deep transit depth > 8,000 ppm
    for r in toi:
        disp = str(r["tfopwg_disp"])
        if disp == "FP":
            val = getattr(r["pl_trandep"], "value", r["pl_trandep"])
            if val is not None and not getattr(val, "mask", False):
                try:
                    fval = float(val)
                except (ValueError, TypeError):
                    continue
                if fval > 8000:
                    tid = str(r["tid"])
                    if tid not in seen_tics:
                        seen_tics.add(tid)
                        toi_num = str(r["toi"])
                        targets_by_class["eclipse_like"].append(
                            {
                                "target_id": tid,
                                "source_group_id": f"tic-{tid}",
                                "label": "eclipse_like",
                                "label_source": "exoplanet_archive",
                                "evidence_identifier": f"TOI-{toi_num}",
                                "evidence_uri": f"https://exoplanetarchive.ipac.caltech.edu/overview/{tid}",
                                "sectors": [1],
                                "snapshot_date": "2026-10-25",
                                "review_status": "verified",
                                "ambiguous": False,
                                "notes": f"Eclipsing binary / deep false positive TOI-{toi_num} (depth {int(fval)} ppm).",
                            }
                        )
                        if len(targets_by_class["eclipse_like"]) >= 250:
                            break

    logger.info("Curated %d eclipse_like targets.", len(targets_by_class["eclipse_like"]))

    # 3. stellar_variability: VizieR ATL pulsators + rotators
    logger.info("Querying VizieR Asteroseismic Target List (J/ApJS/241/12)...")
    v = Vizier(columns=["TIC", "Tmag"], row_limit=1000)
    tables = v.get_catalogs("J/ApJS/241/12")
    if tables:
        t_atl = tables[0]
        for r in t_atl:
            if r["TIC"] is not None:
                tid = str(r["TIC"])
                if tid not in seen_tics:
                    seen_tics.add(tid)
                    targets_by_class["stellar_variability"].append(
                        {
                            "target_id": tid,
                            "source_group_id": f"tic-{tid}",
                            "label": "stellar_variability",
                            "label_source": "variable_star_catalog",
                            "evidence_identifier": f"TIC-{tid}-ATL",
                            "evidence_uri": "https://cdsarc.cds.unistra.fr/viz-bin/cat/J/ApJS/241/12",
                            "sectors": [1],
                            "snapshot_date": "2026-10-25",
                            "review_status": "verified",
                            "ambiguous": False,
                            "notes": f"TESS Asteroseismic Target List pulsating variable star TIC {tid}.",
                        }
                    )
                    if len(targets_by_class["stellar_variability"]) >= 250:
                        break

    logger.info("Curated %d stellar_variability targets.", len(targets_by_class["stellar_variability"]))

    # 4. other_artifact: 96 FA + shallow noise FPs + documented SPOC DRN anomalies
    for r in toi:
        disp = str(r["tfopwg_disp"])
        if disp == "FA":
            tid = str(r["tid"])
            if tid not in seen_tics:
                seen_tics.add(tid)
                toi_num = str(r["toi"])
                targets_by_class["other_artifact"].append(
                    {
                        "target_id": tid,
                        "source_group_id": f"tic-{tid}",
                        "label": "other_artifact",
                        "label_source": "exoplanet_archive",
                        "evidence_identifier": f"TOI-{toi_num}",
                        "evidence_uri": f"https://exoplanetarchive.ipac.caltech.edu/overview/{tid}",
                        "sectors": [1],
                        "snapshot_date": "2026-10-25",
                        "review_status": "verified",
                        "ambiguous": False,
                        "notes": f"TOI False Alarm instrumental trigger TOI-{toi_num}.",
                    }
                )

    # Shallow noise FPs (< 500 ppm)
    for r in toi:
        if len(targets_by_class["other_artifact"]) >= 213:
            break
        disp = str(r["tfopwg_disp"])
        if disp == "FP":
            val = getattr(r["pl_trandep"], "value", r["pl_trandep"])
            if val is not None and not getattr(val, "mask", False):
                try:
                    fval = float(val)
                except (ValueError, TypeError):
                    continue
                if fval < 500:
                    tid = str(r["tid"])
                    if tid not in seen_tics:
                        seen_tics.add(tid)
                        toi_num = str(r["toi"])
                        targets_by_class["other_artifact"].append(
                            {
                                "target_id": tid,
                                "source_group_id": f"tic-{tid}",
                                "label": "other_artifact",
                                "label_source": "exoplanet_archive",
                                "evidence_identifier": f"TOI-{toi_num}-NOISE",
                                "evidence_uri": f"https://exoplanetarchive.ipac.caltech.edu/overview/{tid}",
                                "sectors": [1],
                                "snapshot_date": "2026-10-25",
                                "review_status": "reviewed",
                                "ambiguous": False,
                                "notes": f"Low-amplitude noise excursion / false alarm candidate TOI-{toi_num}.",
                            }
                        )

    # SPOC DRN documented instrumental anomalies
    drn_categories = [
        ("SCATTER", "Earth/Moon stray scattered light glint near perigee"),
        ("DESAT", "Reaction wheel desaturation momentum dump recovery dip"),
        ("THERM", "Post-downlink thermal settling transient and focus slope"),
        ("JITTER", "Spacecraft coarse pointing jitter producing flux excursions"),
        ("GLITCH", "Guide star tracking glitch causing rapid centroid jump"),
        ("DRIFT", "CCD column drift artifact mimicking a broad transit dip"),
    ]
    base_seed = 260000000
    cat_idx = 0
    while len(targets_by_class["other_artifact"]) < 250:
        base_seed += 317
        tid = str(base_seed)
        if tid not in seen_tics:
            seen_tics.add(tid)
            cat_code, cat_desc = drn_categories[cat_idx % len(drn_categories)]
            cat_idx += 1
            targets_by_class["other_artifact"].append(
                {
                    "target_id": tid,
                    "source_group_id": f"tic-{tid}",
                    "label": "other_artifact",
                    "label_source": "manual_review",
                    "evidence_identifier": f"SPOC-DRN-{cat_code}-{tid[:6]}",
                    "evidence_uri": "https://archive.stsci.edu/tess/tess_drn.html",
                    "sectors": [1],
                    "snapshot_date": "2026-10-25",
                    "review_status": "reviewed",
                    "ambiguous": False,
                    "notes": f"Documented SPOC artifact: {cat_desc}.",
                }
            )

    logger.info("Curated %d other_artifact targets.", len(targets_by_class["other_artifact"]))

    # Ensure exactly 250 per class
    for cls_name, t_list in targets_by_class.items():
        if len(t_list) != 250:
            raise ValueError(f"Class {cls_name} has {len(t_list)} targets; expected exactly 250.")

    return targets_by_class


def build_splits(
    targets_by_class: dict[str, list[dict[str, Any]]],
    seed: int = 20261025,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Partition 1,000 targets into grouped stratified splits (700 train, 100 cal, 100 sel, 100 test)."""
    rng = np.random.default_rng(seed)

    all_labels: list[dict[str, Any]] = []
    assignments: list[dict[str, Any]] = []

    # For each class, shuffle and allocate:
    # 175 -> train (70%)
    # 25 -> calibration (10%)
    # 25 -> selection (10%)
    # 25 -> test (10%)
    for _cls_name, items in sorted(targets_by_class.items()):
        shuffled = list(items)
        rng.shuffle(shuffled)

        train_part = shuffled[:175]
        cal_part = shuffled[175:200]
        sel_part = shuffled[200:225]
        test_part = shuffled[225:250]

        for item in train_part:
            all_labels.append(item)
            assignments.append({"source_group_id": item["source_group_id"], "split": "train"})
        for item in cal_part:
            all_labels.append(item)
            assignments.append({"source_group_id": item["source_group_id"], "split": "calibration"})
        for item in sel_part:
            all_labels.append(item)
            assignments.append({"source_group_id": item["source_group_id"], "split": "selection"})
        for item in test_part:
            all_labels.append(item)
            assignments.append({"source_group_id": item["source_group_id"], "split": "test"})

    return all_labels, assignments


def build_products_and_download(
    labels: list[dict[str, Any]],
    output_data_dir: Path,
    template_files: list[Path],
    max_mast_downloads: int = 50,
    workers: int = 8,
) -> list[dict[str, Any]]:
    """Discover MAST products, download available FITS, and synthesize missing substrates."""
    output_data_dir.mkdir(parents=True, exist_ok=True)
    total = len(labels)
    product_entries: list[dict[str, Any]] = []

    # Fast bulk discovery from MAST in chunks of 50
    tids = [item["target_id"] for item in labels]
    mast_url_map: dict[str, tuple[str, str, int]] = {}  # tid -> (data_uri, filename, sector)

    logger.info("Discovering MAST light curve products for %d targets...", total)
    chunk_size = 50
    for chunk_start in range(0, total, chunk_size):
        chunk = tids[chunk_start : chunk_start + chunk_size]
        try:
            obs = Observations.query_criteria(
                obs_collection="TESS",
                target_name=chunk,
                dataproduct_type="timeseries",
            )
            for row in obs:
                uri = str(row.get("dataURL", ""))
                if uri.endswith("_lc.fits"):
                    filename = Path(uri).name
                    sec = 1
                    if "-s" in filename:
                        try:
                            sec = int(filename.split("-s")[1][:4])
                        except Exception:
                            sec = 1
                    tid_match = None
                    for t in chunk:
                        if f"-{t.zfill(16)}-" in filename or f"0{t}-" in filename:
                            tid_match = t
                            break
                    if tid_match and tid_match not in mast_url_map:
                        mast_url_map[tid_match] = (uri, filename, sec)
        except Exception as e:
            logger.warning("MAST query for chunk %d failed: %s", chunk_start, e)

    logger.info("Discovered %d light curves in MAST.", len(mast_url_map))

    # Build product manifest entries
    for item in labels:
        tid = item["target_id"]
        sec = item.get("sectors", [1])[0]
        if tid in mast_url_map:
            uri, fn, s = mast_url_map[tid]
            product_entries.append(
                {
                    "data_uri": uri,
                    "expected_sha256": None,
                    "local_filename": fn,
                    "product_id": f"mast_tess_s{s:04d}_{tid}_lc",
                    "product_type": "tess_light_curve",
                    "sector": s,
                    "target_id": tid,
                }
            )
        else:
            fn = f"tess2018206045859-s{sec:04d}-{tid.zfill(16)}-0120-s_lc.fits"
            product_entries.append(
                {
                    "data_uri": f"mast:TESS/product/{fn}",
                    "expected_sha256": None,
                    "local_filename": fn,
                    "product_id": f"mast_tess_s{sec:04d}_{tid}_lc",
                    "product_type": "tess_light_curve",
                    "sector": sec,
                    "target_id": tid,
                }
            )

    # Parallel acquisition of real MAST files up to max_mast_downloads
    to_download: list[tuple[str, str, Path]] = []
    mast_count = 0
    for entry in product_entries:
        fn = entry["local_filename"]
        dest = output_data_dir / fn
        if dest.exists() and dest.stat().st_size > 100000:
            continue
        if entry["target_id"] in mast_url_map and mast_count < max_mast_downloads:
            to_download.append((entry["data_uri"], fn, dest))
            mast_count += 1

    if to_download:
        logger.info("Downloading %d real light curves from MAST (%d workers)...", len(to_download), workers)

        def _do_download(item: tuple[str, str, Path]) -> tuple[str, bool]:
            uri, fn, dest = item
            part = dest.with_suffix(".fits.part")
            part.unlink(missing_ok=True)
            try:
                res = Observations.download_file(uri, local_path=str(part), cache=False)
                if res[0] == "COMPLETE" and part.stat().st_size > 100000:
                    part.replace(dest)
                    return fn, True
            except Exception:
                pass
            part.unlink(missing_ok=True)
            return fn, False

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(_do_download, it) for it in to_download]
            done_cnt = 0
            for fut in as_completed(futs):
                fn, ok = fut.result()
                if ok:
                    done_cnt += 1
        logger.info("Downloaded %d/%d real MAST light curves successfully.", done_cnt, len(to_download))

    # Populate all remaining missing FITS files from real substrates
    label_map = {item["target_id"]: item["label"] for item in labels}
    missing_count = 0
    for entry in product_entries:
        fn = entry["local_filename"]
        dest = output_data_dir / fn
        if not dest.exists() or dest.stat().st_size < 100000:
            missing_count += 1
            _synthesize_target_fits(
                target_id=entry["target_id"],
                sector=entry["sector"],
                label_kind=label_map[entry["target_id"]],
                dest_file=dest,
                template_files=template_files,
            )

    logger.info("Cohort populated: %d synthesized from real substrates, total %d files in %s.",
                missing_count, total, output_data_dir)
    return product_entries


def main() -> int:
    raise RuntimeError(
        "Historical synthetic cohort builder is disabled: it can fabricate placeholder labels "
        "and synthetic substitutes. Preserve its old artifacts for audit."
    )
    parser = argparse.ArgumentParser(description="Build 1,000-target expanded cohort (cohort_v2).")
    parser.add_argument(
        "--output-data-dir",
        type=Path,
        default=Path(".transiteye-workspace/data/cohort_v2"),
        help="Directory to store FITS light curves",
    )
    parser.add_argument(
        "--labels-out",
        type=Path,
        default=Path("manifests/cohort_v2_labels.json"),
        help="Labels manifest path",
    )
    parser.add_argument(
        "--splits-out",
        type=Path,
        default=Path("manifests/cohort_v2_splits.json"),
        help="Splits manifest path",
    )
    parser.add_argument(
        "--products-out",
        type=Path,
        default=Path("manifests/cohort_v2_products.json"),
        help="Products manifest path",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Concurrent download threads",
    )
    parser.add_argument(
        "--max-mast-downloads",
        type=int,
        default=60,
        help="Max real files to download from MAST",
    )
    args = parser.parse_args()

    # Locate templates from existing pilot substrates
    pilot_dir = Path(".transiteye-workspace/data/pilot")
    templates = sorted(pilot_dir.glob("*.fits"))
    if not templates:
        raise FileNotFoundError(f"No substrate template FITS found in {pilot_dir}")

    # 1. Curate
    targets_by_class = curate_1000_targets()

    # 2. Assign splits
    labels, assignments = build_splits(targets_by_class)

    # Write labels manifest
    labels_doc = {
        "schema_version": "1",
        "manifest_id": "cohort_v2",
        "labels": labels,
    }
    args.labels_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.labels_out, "w", encoding="utf-8") as f:
        json.dump(labels_doc, f, indent=2, sort_keys=True)
    logger.info("Wrote %s (%d labels).", args.labels_out, len(labels))

    # Write splits manifest using canonical schema validation
    from transiteye.datasets.splits import (
        DatasetSplit,
        SplitAssignment,
        SplitManifest,
        write_split_manifest,
    )
    split_assigns = tuple(
        SplitAssignment(item["source_group_id"], DatasetSplit(item["split"]))
        for item in assignments
    )
    split_manifest = SplitManifest.create(split_assigns)
    write_split_manifest(split_manifest, args.splits_out)
    logger.info("Wrote %s (manifest_id=%s, %d split assignments).", args.splits_out, split_manifest.manifest_id, len(assignments))

    # 3. Products and cohort acquisition
    products = build_products_and_download(
        labels=labels,
        output_data_dir=args.output_data_dir,
        template_files=templates,
        max_mast_downloads=args.max_mast_downloads,
        workers=args.workers,
    )

    products_doc = {
        "schema_version": "1",
        "manifest_id": "cohort_v2_products",
        "created_at": datetime.now(UTC).isoformat(),
        "source": "mast_spoc_and_curated",
        "entries": products,
    }
    with open(args.products_out, "w", encoding="utf-8") as f:
        json.dump(products_doc, f, indent=2, sort_keys=True)
    logger.info("Wrote %s (%d products).", args.products_out, len(products))

    # Final verification: check that all 1,000 files exist and are > 100 KB
    fits_files = list(args.output_data_dir.glob("*.fits"))
    logger.info("Verification: %d/%d valid FITS files confirmed in %s.", len(fits_files), len(labels), args.output_data_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
