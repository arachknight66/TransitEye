"""Validated readers for SPOC and TESS-SPOC light-curve FITS products."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

from transiteye.domain import LightCurve, Provenance


class FitsValidationError(ValueError):
    """Raised when a local FITS product cannot support TransitEye analysis."""


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


def read_tess_light_curve(path: Path, flux_preference: str = "PDCSAP_FLUX") -> LightCurve:
    """Read one delivered TESS light curve without discarding any cadence.

    The caller owns subsequent quality masking and preprocessing. The reader keeps NaNs, quality
    flags, and optional centroid series so later stages can report exactly what they excluded.
    """

    source = path.expanduser().resolve()
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
                    "time_unit": _string(extension.get("TIMEUNIT")) or "d",
                },
            )
    except OSError as error:
        raise FitsValidationError(f"Could not open FITS file {source}: {error}") from error
