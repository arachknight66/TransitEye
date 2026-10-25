"""Ingestion and representation of TESS Target Pixel Files (TPF)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from numpy.typing import NDArray

from transiteye.domain.models import Provenance
from transiteye.io.fits import FitsValidationError, _checksum, _string, _time_reference


@dataclass(frozen=True, slots=True)
class TargetPixelData:
    """Delivered TESS pixel cube and aperture mask."""

    time_days: NDArray[np.float64]
    flux_cube: NDArray[np.float64]  # Shape: (N_time, N_rows, N_cols)
    flux_err_cube: NDArray[np.float64] | None
    aperture_mask: NDArray[np.bool_]  # Shape: (N_rows, N_cols)
    target_pixel_coord: tuple[float, float]  # (column, row) in pixel coords
    provenance: Provenance
    metadata: dict[str, Any]


def read_target_pixel_fits(path: Path) -> TargetPixelData:
    """Read a validated SPOC or TESS-SPOC target pixel file (_tp.fits)."""
    source = path.expanduser().resolve()
    if not source.is_file():
        raise FitsValidationError(f"TPF file does not exist: {source}")

    try:
        with fits.open(source, memmap=False) as hdul:
            if len(hdul) < 2 or not isinstance(hdul[1], fits.BinTableHDU):
                raise FitsValidationError("Expected a pixel binary table in FITS extension 1.")

            primary = hdul[0].header
            extension = hdul[1].header
            table = hdul[1].data

            if table is None or "TIME" not in table.names or "FLUX" not in table.names:
                raise FitsValidationError("TPF table must contain TIME and FLUX columns.")

            time_days = np.asarray(table["TIME"], dtype=np.float64)
            flux_cube = np.asarray(table["FLUX"], dtype=np.float64)

            flux_err_cube = None
            if "FLUX_ERR" in table.names:
                flux_err_cube = np.asarray(table["FLUX_ERR"], dtype=np.float64)

            # Extension 2 contains the photometric aperture bitmask image
            if len(hdul) >= 3 and isinstance(hdul[2], fits.ImageHDU):
                aperture_raw = np.asarray(hdul[2].data, dtype=np.int32)
                # Bit 2 (value 2) indicates pixel included in standard photometric aperture
                aperture_mask = (aperture_raw & 2) != 0
            else:
                # Default to center pixels if extension 2 is absent
                _, rows, cols = flux_cube.shape
                aperture_mask = np.ones((rows, cols), dtype=bool)

            target_id = _string(primary.get("TICID", extension.get("TICID")))
            sector_value = primary.get("SECTOR", extension.get("SECTOR"))

            # Approximate target pixel location in stamp
            # TESS reference pixel columns: 1CRV4P or header keywords
            _, n_r, n_c = flux_cube.shape
            target_pixel_coord = (float(n_c) / 2.0, float(n_r) / 2.0)

            provenance = Provenance(
                source_uri=f"file://{source}",
                source_path=str(source),
                checksum_sha256=_checksum(source),
                product_type="tess_target_pixel_file",
                target_id=target_id,
                sector=int(sector_value) if sector_value is not None else None,
                time_system=_string(extension.get("TIMESYS", primary.get("TIMESYS"))),
                time_reference=_time_reference(primary, extension),
            )

            return TargetPixelData(
                time_days=time_days,
                flux_cube=flux_cube,
                flux_err_cube=flux_err_cube,
                aperture_mask=aperture_mask,
                target_pixel_coord=target_pixel_coord,
                provenance=provenance,
                metadata={
                    "filename": source.name,
                    "target_id": target_id,
                    "crowdsap": _string(extension.get("CROWDSAP", primary.get("CROWDSAP"))),
                },
            )
    except OSError as err:
        raise FitsValidationError(f"Could not open TPF file {source}: {err}") from err
