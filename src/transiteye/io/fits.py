"""Validated readers for SPOC and TESS-SPOC light-curve FITS products."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

from transiteye.domain import LightCurve, Provenance


class FitsValidationError(ValueError):
    """Raised when a local FITS product cannot support TransitEye analysis."""


def validate_day_time_axis(curve: LightCurve) -> None:
    """Require explicit TDB/day coordinates and a matching BJD reference."""
    if curve.provenance.time_system != "TDB":
        raise FitsValidationError("Light curve requires TDB time coordinates.")
    try:
        reference = float(curve.provenance.time_reference or "")
    except ValueError as error:
        raise FitsValidationError("Light curve has no valid BJD reference.") from error
    if not np.isfinite(reference) or not 2_400_000 <= reference <= 2_600_000:
        raise FitsValidationError("Light curve has no valid BJD reference.")
    units = (curve.metadata.get("time_unit"), curve.metadata.get("time_column_unit"))
    if not any(unit is not None for unit in units):
        raise FitsValidationError("Light curve has no explicit TIME unit.")
    for unit in units:
        if unit is None:
            continue
        text = str(unit).strip().lower()
        if text in ("d", "day", "days"):
            continue
        match = re.fullmatch(r"bjd\s*-\s*(\d+(?:\.\d+)?)", text)
        if match is None or abs(float(match.group(1)) - reference) >= 1e-7:
            raise FitsValidationError(f"Unsupported or conflicting TIME unit: {unit}")


def _string(value: Any) -> str | None:
    return None if value is None else str(value).strip() or None


def _column(table: fits.FITS_rec, name: str, dtype: np.dtype[Any]) -> np.ndarray | None:
    if name not in table.names:
        return None
    return np.asarray(table[name], dtype=dtype)


def _checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _time_reference(primary: fits.Header, extension: fits.Header) -> str | None:
    integer = extension.get("BJDREFI", primary.get("BJDREFI"))
    fraction = extension.get("BJDREFF", primary.get("BJDREFF"))
    if integer is None and fraction is None:
        return None
    return str(float(integer or 0.0) + float(fraction or 0.0))


def read_tess_light_curve(path: Path | str, flux_preference: str = "PDCSAP_FLUX") -> LightCurve:
    """Read one delivered TESS light curve without discarding any cadence.

    The caller owns subsequent quality masking and preprocessing. The reader keeps NaNs, quality
    flags, and optional centroid series so later stages can report exactly what they excluded.
    """

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FitsValidationError(f"FITS file does not exist: {source}")
    try:
        with fits.open(source, memmap=False) as hdul:
            if len(hdul) < 2 or not isinstance(hdul[1], fits.BinTableHDU):
                raise FitsValidationError(
                    "Expected a light-curve binary table in FITS extension 1."
                )
            primary = hdul[0].header
            extension = hdul[1].header
            for keyword in ("MISSION", "TELESCOP"):
                declared = _string(primary.get(keyword, extension.get(keyword)))
                if declared is not None and declared.casefold() != "tess":
                    raise FitsValidationError(f"Expected TESS FITS, found {keyword}={declared}.")
            table = hdul[1].data
            if table is None or table.names is None:
                raise FitsValidationError("FITS light-curve table is empty.")
            if "TIME" not in table.names:
                raise FitsValidationError("FITS light-curve table does not contain TIME.")
            available_fluxes = [
                name for name in (flux_preference, "PDCSAP_FLUX", "SAP_FLUX") if name in table.names
            ]
            if not available_fluxes:
                raise FitsValidationError(
                    "FITS light-curve table has neither PDCSAP_FLUX nor SAP_FLUX."
                )
            selected_flux = available_fluxes[0]
            flux_error_name = selected_flux.replace("FLUX", "FLUX_ERR")
            target_id = _string(primary.get("TICID", extension.get("TICID")))
            sector_value = primary.get("SECTOR", extension.get("SECTOR"))
            provenance = Provenance(
                source_uri=f"file://{source}",
                source_path=str(source),
                checksum_sha256=_checksum(source),
                product_type="tess_light_curve",
                target_id=target_id,
                sector=int(sector_value) if sector_value is not None else None,
                time_system=_string(extension.get("TIMESYS", primary.get("TIMESYS"))),
                time_reference=_time_reference(primary, extension),
                metadata={
                    "mission": _string(primary.get("MISSION")) or "TESS",
                    "creator": _string(primary.get("CREATOR")) or "unknown",
                    "origin": _string(primary.get("ORIGIN")) or "unknown",
                    "selected_flux": selected_flux,
                    **{
                        key: str(value)
                        for key in ("TIMEDEL", "NUM_FRM", "FRAMETIM", "INT_TIME", "READTIME", "EXPOSURE", "LIVETIME")
                        if (value := extension.get(key, primary.get(key))) is not None
                    },
                },
            )
            return LightCurve(
                time_days=np.asarray(table["TIME"], dtype=np.float64),
                flux=np.asarray(table[selected_flux], dtype=np.float64),
                flux_error=_column(table, flux_error_name, np.dtype(np.float64)),
                quality=_column(table, "QUALITY", np.dtype(np.int64)),
                cadence_number=_column(table, "CADENCENO", np.dtype(np.int64)),
                centroid_column=_column(table, "MOM_CENTR1", np.dtype(np.float64)),
                centroid_row=_column(table, "MOM_CENTR2", np.dtype(np.float64)),
                flux_kind=selected_flux,
                flux_unit=_string(extension.get(f"TUNIT{table.names.index(selected_flux) + 1}")),
                provenance=provenance,
                metadata={
                    "filename": source.name,
                    "time_unit": _string(extension.get("TIMEUNIT", primary.get("TIMEUNIT"))),
                    "time_column_unit": _string(extension.get(f"TUNIT{table.names.index('TIME') + 1}")),
                    "crowdsap": _string(extension.get("CROWDSAP", primary.get("CROWDSAP"))),
                    "flfrcsap": _string(extension.get("FLFRCSAP", primary.get("FLFRCSAP"))),
                    "teff": _string(primary.get("TEFF", extension.get("TEFF"))),
                    "logg": _string(primary.get("LOGG", extension.get("LOGG"))),
                    # Preserve frame metadata as supplied; TIMEDEL is sampling, and
                    # EXPOSURE/LIVETIME are total time on source, not one exposure.
                    "time_resolution": _string(extension.get("TIMEDEL", primary.get("TIMEDEL"))),
                    "num_frames": _string(extension.get("NUM_FRM", primary.get("NUM_FRM"))),
                    "frame_time_seconds": _string(extension.get("FRAMETIM", primary.get("FRAMETIM"))),
                    "integration_time_seconds": _string(extension.get("INT_TIME", primary.get("INT_TIME"))),
                    "readout_time_seconds": _string(extension.get("READTIME", primary.get("READTIME"))),
                },
            )
    except OSError as error:
        raise FitsValidationError(f"Could not open FITS file {source}: {error}") from error


def read_kepler_light_curve(path: Path | str, flux_preference: str = "PDCSAP_FLUX") -> LightCurve:
    """Read an official Kepler light curve without applying a quality mask.

    Kepler's native TIME values use the FITS BJD reference (normally BKJD), which
    is retained rather than silently interpreted as TESS BTJD.
    """
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FitsValidationError(f"FITS file does not exist: {source}")
    try:
        with fits.open(source, memmap=False) as hdul:
            if len(hdul) < 2 or not isinstance(hdul[1], fits.BinTableHDU):
                raise FitsValidationError("Expected a Kepler light-curve binary table.")
            primary, extension = hdul[0].header, hdul[1].header
            declared_mission = _string(primary.get("MISSION", extension.get("MISSION")))
            if declared_mission is not None and declared_mission.casefold() != "kepler":
                raise FitsValidationError(f"Expected Kepler FITS, found mission {declared_mission}.")
            telescope = _string(primary.get("TELESCOP", extension.get("TELESCOP")))
            if telescope is not None and telescope.casefold() != "kepler":
                raise FitsValidationError(f"Expected Kepler FITS, found TELESCOP={telescope}.")
            table = hdul[1].data
            if table is None or table.names is None or "TIME" not in table.names:
                raise FitsValidationError("Kepler FITS light-curve table lacks TIME.")
            selected_flux = next(
                (name for name in (flux_preference, "PDCSAP_FLUX", "SAP_FLUX") if name in table.names),
                None,
            )
            if selected_flux is None:
                raise FitsValidationError("Kepler FITS lacks PDCSAP_FLUX and SAP_FLUX.")
            target = primary.get("KEPLERID", extension.get("KEPLERID"))
            if target is None:
                raise FitsValidationError("Kepler FITS lacks KEPLERID.")
            quarter = primary.get("QUARTER", extension.get("QUARTER"))
            provenance = Provenance(
                source_uri=f"file://{source}",
                source_path=str(source),
                checksum_sha256=_checksum(source),
                product_type="kepler_light_curve",
                target_id=str(int(target)),
                sector=None,
                time_system=_string(extension.get("TIMESYS", primary.get("TIMESYS"))),
                time_reference=_time_reference(primary, extension),
                metadata={
                    "mission": "Kepler",
                    "quarter": "" if quarter is None else str(int(quarter)),
                    "selected_flux": selected_flux,
                    **{
                        key: str(value)
                        for key in ("TIMEDEL", "NUM_FRM", "FRAMETIM", "INT_TIME", "READTIME", "EXPOSURE", "LIVETIME")
                        if (value := extension.get(key, primary.get(key))) is not None
                    },
                },
            )
            return LightCurve(
                time_days=np.asarray(table["TIME"], dtype=np.float64),
                flux=np.asarray(table[selected_flux], dtype=np.float64),
                flux_error=_column(table, selected_flux + "_ERR", np.dtype(np.float64)),
                quality=_column(table, "SAP_QUALITY", np.dtype(np.int64)),
                cadence_number=_column(table, "CADENCENO", np.dtype(np.int64)),
                centroid_column=_column(table, "MOM_CENTR1", np.dtype(np.float64)),
                centroid_row=_column(table, "MOM_CENTR2", np.dtype(np.float64)),
                flux_kind=selected_flux,
                flux_unit=_string(extension.get(f"TUNIT{table.names.index(selected_flux) + 1}")),
                provenance=provenance,
                metadata={
                    "time_unit": _string(extension.get("TIMEUNIT", primary.get("TIMEUNIT"))),
                    "time_column_unit": _string(extension.get(f"TUNIT{table.names.index('TIME') + 1}")),
                },
            )
    except OSError as error:
        raise FitsValidationError(f"Could not open FITS file {source}: {error}") from error


def stitch_light_curves(light_curves: Sequence[LightCurve]) -> LightCurve:
    """Stitch multiple sector light curves for the same target into a single time-ordered series.

    Preserves quality flags, centroids, and provenance across sectors, sorting cadences
    chronologically and validating matching target identifiers.
    """
    if not light_curves:
        raise ValueError("Cannot stitch an empty sequence of light curves.")
    if len(light_curves) == 1:
        return light_curves[0]

    # Validate target_id consistency
    target_ids = {lc.provenance.target_id for lc in light_curves if lc.provenance.target_id}
    if len(target_ids) > 1:
        raise ValueError(f"Cannot stitch light curves from different targets: {sorted(target_ids)}")

    # Sort light curves chronologically by first cadence time
    sorted_lcs = sorted(light_curves, key=lambda lc: lc.time_days[0] if len(lc.time_days) else 0.0)

    # Combine arrays
    all_time = np.concatenate([lc.time_days for lc in sorted_lcs])
    all_flux = np.concatenate([lc.flux for lc in sorted_lcs])

    # Handle optional arrays
    has_err = all(lc.flux_error is not None for lc in sorted_lcs)
    all_err = np.concatenate([lc.flux_error for lc in sorted_lcs]) if has_err else None

    has_qual = all(lc.quality is not None for lc in sorted_lcs)
    all_qual = np.concatenate([lc.quality for lc in sorted_lcs]) if has_qual else None

    has_cad = all(lc.cadence_number is not None for lc in sorted_lcs)
    all_cad = np.concatenate([lc.cadence_number for lc in sorted_lcs]) if has_cad else None

    has_c1 = all(lc.centroid_column is not None for lc in sorted_lcs)
    all_c1 = np.concatenate([lc.centroid_column for lc in sorted_lcs]) if has_c1 else None

    has_c2 = all(lc.centroid_row is not None for lc in sorted_lcs)
    all_c2 = np.concatenate([lc.centroid_row for lc in sorted_lcs]) if has_c2 else None

    # Ensure strictly sorted by time
    order = np.argsort(all_time)
    all_time = all_time[order]
    all_flux = all_flux[order]
    if all_err is not None:
        all_err = all_err[order]
    if all_qual is not None:
        all_qual = all_qual[order]
    if all_cad is not None:
        all_cad = all_cad[order]
    if all_c1 is not None:
        all_c1 = all_c1[order]
    if all_c2 is not None:
        all_c2 = all_c2[order]

    sectors = [lc.provenance.sector for lc in sorted_lcs if lc.provenance.sector is not None]
    primary_prov = sorted_lcs[0].provenance
    combined_prov = Provenance(
        source_uri=primary_prov.source_uri,
        source_path=primary_prov.source_path,
        checksum_sha256=primary_prov.checksum_sha256,
        product_type="stitched_tess_light_curve",
        target_id=primary_prov.target_id,
        sector=sectors[0] if len(sectors) == 1 else None,
        time_system=primary_prov.time_system,
        time_reference=primary_prov.time_reference,
        metadata={
            **primary_prov.metadata,
            "stitched_sectors": sectors,
            "sector_count": len(sorted_lcs),
        },
    )

    return LightCurve(
        time_days=all_time,
        flux=all_flux,
        flux_error=all_err,
        quality=all_qual,
        cadence_number=all_cad,
        centroid_column=all_c1,
        centroid_row=all_c2,
        flux_kind=sorted_lcs[0].flux_kind,
        flux_unit=sorted_lcs[0].flux_unit,
        provenance=combined_prov,
        metadata={
            **sorted_lcs[0].metadata,
            "stitched_sectors": sectors,
        },
    )
