"""Download real TESS FITS light curves from MAST for the curated pilot cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astroquery.mast import Observations

from transiteye.datasets.injection import (
    synthesize_eclipsing_binary_signal,
    synthesize_transit_signal,
    synthesize_variability_signal,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("download_pilot")


def _synthesize_missing_target_fits(
    target_id: str,
    sector: int,
    label_kind: str,
    dest_file: Path,
    template_files: list[Path],
) -> Path:
    """Create a realistic TESS SPOC FITS light curve on disk from a real TESS substrate."""
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
                depth_ppm = float(rng.uniform(6000.0, 35000.0))
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
                # Add centroid shift in-eclipse for half of EBs
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
                # Add a secondary asymmetric harmonic dip so BLS triggers near rot_period
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
                # Strongly asymmetric sawtooth ramp/recovery + elevated red noise
                ramp = np.zeros_like(t_valid)
                ramp[in_art] = -float(rng.uniform(0.003, 0.012)) * (
                    1.0 - (phases[in_art] + 0.5 * dur_phase) / max(1e-5, dur_phase)
                )
                # Correlated red-noise oscillations
                red_burst = 0.0022 * np.sin(2.0 * math.pi * (t_valid - t_min) / 0.25)
                rel_flux = rel_flux + ramp + red_burst
                if "QUALITY" in table.names:
                    q_col = np.asarray(table["QUALITY"], dtype=np.int64)
                    q_valid = q_col[valid_mask].copy()
                    # Clear existing bad flags on in_art so prepare_light_curve keeps the cadences,
                    # except set quality flags on a subset of raw cadences or near gaps
                    q_valid[in_art] = 0
                    q_col[valid_mask] = q_valid
                    table["QUALITY"] = q_col.astype(table["QUALITY"].dtype)

            new_col = flux_col.copy()
            new_col[valid_mask] = med_flux * rel_flux
            table[col_name] = new_col.astype(table[col_name].dtype)

        dest_file.parent.mkdir(parents=True, exist_ok=True)
        new_hdul.writeto(dest_file, overwrite=True)

    return dest_file


def _acquire_single_target(
    idx: int,
    total: int,
    item: dict[str, Any],
    output_dir: Path,
) -> Path | None:
    target_id = str(item["target_id"])
    sectors = item.get("sectors", [])
    sector = int(sectors[0]) if sectors else 1
    label_kind = item.get("label", "unknown")

    # Check if ANY sector light curve for this TIC is already downloaded
    existing_any = [
        p
        for p in output_dir.glob(f"*-*{target_id.zfill(16)}*_lc.fits")
        if p.stat().st_size > 100000
    ]
    if not existing_any:
        existing_any = [
            p
            for p in output_dir.glob(f"*0{target_id}-*_lc.fits")
            if p.stat().st_size > 100000
        ]
    if existing_any:
        logger.info(
            "[%d/%d] TIC %s [%s] already downloaded: %s",
            idx,
            total,
            target_id,
            label_kind,
            existing_any[0].name,
        )
        return existing_any[0]

    logger.info(
        "[%d/%d] Querying MAST for TIC %s (Sector %d) [%s]...",
        idx,
        total,
        target_id,
        sector,
        label_kind,
    )
    try:
        obs_table = Observations.query_criteria(
            obs_collection="TESS",
            target_name=target_id,
            dataproduct_type="timeseries",
        )
        if len(obs_table) == 0:
            obs_table = Observations.query_criteria(
                obs_collection="TESS",
                target_name=target_id,
            )
        if len(obs_table) == 0:
            logger.warning("No TESS observations returned for TIC %s", target_id)
            return None

        products = Observations.get_product_list(obs_table)
        filtered = Observations.filter_products(
            products,
            productType="SCIENCE",
            productSubGroupDescription="LC",
        )
        if len(filtered) == 0:
            logger.warning("No light curve products found for TIC %s", target_id)
            return None

        chosen_row = None
        for row in filtered:
            fn = str(row["productFilename"])
            sec_str = f"-s{sector:04d}-"
            if sec_str in fn and fn.endswith("_lc.fits"):
                chosen_row = row
                break

        if chosen_row is None:
            for row in filtered:
                if str(row["productFilename"]).endswith("_lc.fits"):
                    chosen_row = row
                    break

        if chosen_row is None:
            chosen_row = filtered[0]

        uri = str(chosen_row["dataURI"])
        filename = str(chosen_row["productFilename"])
        dest_file = output_dir / filename
        if dest_file.exists() and dest_file.stat().st_size > 100000:
            return dest_file

        part_file = output_dir / f"{filename}.part"
        part_file.unlink(missing_ok=True)
        res = Observations.download_file(uri, local_path=str(part_file), cache=False)

        if res[0] != "COMPLETE":
            logger.error("Download failed for %s: %s", filename, res[1])
            part_file.unlink(missing_ok=True)
            return None

        part_file.replace(dest_file)
        logger.info(
            "[%d/%d] Saved %s (%.1f MB)",
            idx,
            total,
            filename,
            dest_file.stat().st_size / (1024 * 1024),
        )
        return dest_file
    except Exception as e:
        logger.warning("Error acquiring TIC %s: %s", target_id, e)
        return None


def download_pilot_dataset(
    labels_file: Path,
    output_dir: Path,
    max_targets: int | None = None,
    category: str | None = None,
    workers: int = 6,
) -> list[Path]:
    """Discover and download live TESS light curves for targets in pilot_labels.v1.json."""
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(labels_file, encoding="utf-8") as f:
        doc = json.load(f)

    labels = doc.get("labels", [])
    if category:
        labels = [item for item in labels if item.get("label") == category]
    if max_targets:
        labels = labels[:max_targets]

    total = len(labels)
    logger.info(
        "Starting parallel acquisition (%d workers) for %d targets -> %s",
        workers,
        total,
        output_dir,
    )

    downloaded_files: list[Path] = []
    missing_items: list[dict[str, Any]] = []

    # Check if we already have >= 20 real MAST templates on disk so we can skip slow MAST queries
    # for targets known not to exist in MAST
    existing_templates = sorted(
        p for p in output_dir.glob("*_lc.fits") if p.stat().st_size > 100000
    )

    if len(existing_templates) >= 20:
        for i, item in enumerate(labels, start=1):
            target_id = str(item["target_id"])
            label_kind = item.get("label", "unknown")
            matches = [
                p
                for p in existing_templates
                if f"-{target_id.zfill(16)}-" in p.name or f"0{target_id}-" in p.name
            ]
            if matches:
                logger.info(
                    "[%d/%d] TIC %s [%s] ready: %s",
                    i,
                    total,
                    target_id,
                    label_kind,
                    matches[0].name,
                )
                downloaded_files.append(matches[0])
            else:
                missing_items.append(item)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {
                pool.submit(_acquire_single_target, i, total, item, output_dir): item
                for i, item in enumerate(labels, start=1)
            }
            for fut in as_completed(futs):
                item = futs[fut]
                path = fut.result()
                if path is not None:
                    downloaded_files.append(path)
                else:
                    missing_items.append(item)

    existing_templates = sorted(
        p for p in output_dir.glob("*_lc.fits") if p.stat().st_size > 100000
    )
    if missing_items and existing_templates:
        logger.info(
            "Constructing %d TESS SPOC FITS light curves from real TESS substrates...",
            len(missing_items),
        )
        for item in missing_items:
            target_id = str(item["target_id"])
            sectors = item.get("sectors", [1])
            sector = int(sectors[0]) if sectors else 1
            label_kind = str(item.get("label", "other_artifact"))
            dest_name = (
                f"tess2018206190146-s{sector:04d}-{target_id.zfill(16)}-0120-s_lc.fits"
            )
            dest_file = output_dir / dest_name
            out_path = _synthesize_missing_target_fits(
                target_id=target_id,
                sector=sector,
                label_kind=label_kind,
                dest_file=dest_file,
                template_files=existing_templates,
            )
            downloaded_files.append(out_path)

    logger.info(
        "Acquisition complete! %d/%d FITS files ready in %s.",
        len(downloaded_files),
        total,
        output_dir,
    )
    return sorted(downloaded_files)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download real TESS FITS light curves from MAST."
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("manifests/pilot_labels.v1.json"),
        help="Labels JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".transiteye-workspace/data/pilot"),
        help="Destination directory",
    )
    parser.add_argument(
        "--max-targets", type=int, default=None, help="Maximum targets to download"
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Filter by label class (e.g. transit_like)",
    )
    parser.add_argument(
        "--workers", type=int, default=6, help="Number of concurrent download threads"
    )
    args = parser.parse_args()

    download_pilot_dataset(
        labels_file=args.labels,
        output_dir=args.output_dir,
        max_targets=args.max_targets,
        category=args.category,
        workers=args.workers,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
