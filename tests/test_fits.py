from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from transiteye.io.fits import FitsValidationError, read_tess_light_curve


def _write_tess_fixture(path: Path) -> None:
    columns = [
        fits.Column(name="TIME", format="D", array=np.array([1.0, 2.0, np.nan])),
        fits.Column(
            name="PDCSAP_FLUX", format="D", unit="electron / s", array=np.array([100.0, 99.0, 98.0])
        ),
        fits.Column(name="PDCSAP_FLUX_ERR", format="D", array=np.array([1.0, 1.0, 1.0])),
        fits.Column(name="SAP_FLUX", format="D", array=np.array([101.0, 100.0, 99.0])),
        fits.Column(name="QUALITY", format="K", array=np.array([0, 1, 0])),
        fits.Column(name="CADENCENO", format="K", array=np.array([10, 11, 12])),
        fits.Column(name="MOM_CENTR1", format="D", array=np.array([1.1, 1.2, 1.3])),
        fits.Column(name="MOM_CENTR2", format="D", array=np.array([2.1, 2.2, 2.3])),
    ]
    table = fits.BinTableHDU.from_columns(columns)
    table.header["TIMESYS"] = "TDB"
    table.header["BJDREFI"] = 2457000
    primary = fits.PrimaryHDU()
    primary.header["TICID"] = 123456
    primary.header["SECTOR"] = 42
    primary.header["MISSION"] = "TESS"
    primary.header["CREATOR"] = "fixture"
    fits.HDUList([primary, table]).writeto(path)


def test_reader_preserves_delivered_cadences_and_metadata(tmp_path: Path) -> None:
    path = tmp_path / "fixture_lc.fits"
    _write_tess_fixture(path)
    light_curve = read_tess_light_curve(path)
    assert light_curve.flux_kind == "PDCSAP_FLUX"
    assert len(light_curve.time_days) == 3
    assert np.isnan(light_curve.time_days[-1])
    assert light_curve.quality is not None and light_curve.quality.tolist() == [0, 1, 0]
    assert light_curve.provenance.target_id == "123456"
    assert light_curve.provenance.sector == 42
    assert light_curve.provenance.time_system == "TDB"
    assert light_curve.provenance.checksum_sha256 is not None


def test_reader_falls_back_to_sap_when_pdcsap_is_absent(tmp_path: Path) -> None:
    path = tmp_path / "sap_only.fits"
    table = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="TIME", format="D", array=np.array([1.0])),
            fits.Column(name="SAP_FLUX", format="D", array=np.array([1.0])),
        ]
    )
    fits.HDUList([fits.PrimaryHDU(), table]).writeto(path)
    assert read_tess_light_curve(path).flux_kind == "SAP_FLUX"


def test_reader_rejects_missing_time_column(tmp_path: Path) -> None:
    path = tmp_path / "bad.fits"
    table = fits.BinTableHDU.from_columns(
        [fits.Column(name="PDCSAP_FLUX", format="D", array=np.array([1.0]))]
    )
    fits.HDUList([fits.PrimaryHDU(), table]).writeto(path)
    with pytest.raises(FitsValidationError, match="TIME"):
        read_tess_light_curve(path)
