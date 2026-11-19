from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from transiteye.io.fits import (
    FitsValidationError,
    read_tess_light_curve,
    stitch_light_curves,
    validate_day_time_axis,
)


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
    table.header["TIMEDEL"] = 0.01
    table.header["NUM_FRM"] = 60
    table.header["FRAMETIM"] = 2.0
    table.header["INT_TIME"] = 1.98
    table.header["READTIME"] = 0.02
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
    assert light_curve.provenance.metadata["NUM_FRM"] == "60"
    assert light_curve.provenance.metadata["TIMEDEL"] == "0.01"
    assert light_curve.provenance.metadata["selected_flux"] == "PDCSAP_FLUX"
    assert read_tess_light_curve(path, flux_preference="SAP_FLUX").provenance.metadata["selected_flux"] == "SAP_FLUX"
    assert light_curve.metadata["time_resolution"] == "0.01"
    assert light_curve.metadata["num_frames"] == "60"
    assert light_curve.metadata["frame_time_seconds"] == "2.0"
    assert light_curve.metadata["integration_time_seconds"] == "1.98"
    assert "exposure_days" not in light_curve.metadata


def test_time_units_are_preserved_without_assuming_days(tmp_path: Path) -> None:
    path = tmp_path / "units_lc.fits"
    _write_tess_fixture(path)
    with fits.open(path, mode="update") as hdul:
        hdul[1].header["TUNIT1"] = "s"
    curve = read_tess_light_curve(path)
    assert curve.metadata["time_unit"] is None
    assert curve.metadata["time_column_unit"] == "s"
    with pytest.raises(FitsValidationError, match="TIME unit"):
        validate_day_time_axis(curve)


def test_bjd_column_unit_must_match_reference(tmp_path: Path) -> None:
    path = tmp_path / "bjd_lc.fits"
    _write_tess_fixture(path)
    with fits.open(path, mode="update") as hdul:
        hdul[1].header["TUNIT1"] = "BJD - 2457000"
    validate_day_time_axis(read_tess_light_curve(path))
    with fits.open(path, mode="update") as hdul:
        hdul[1].header["TUNIT1"] = "BJD - 2454833"
    with pytest.raises(FitsValidationError, match="conflicting TIME unit"):
        validate_day_time_axis(read_tess_light_curve(path))


def test_tess_reader_rejects_conflicting_telescope(tmp_path: Path) -> None:
    path = tmp_path / "wrong_telescope_lc.fits"
    _write_tess_fixture(path)
    with fits.open(path, mode="update") as hdul:
        hdul[0].header["TELESCOP"] = "Kepler"
    with pytest.raises(FitsValidationError, match="TELESCOP=Kepler"):
        read_tess_light_curve(path)


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


def test_stitch_light_curves_validation() -> None:
    # Empty sequence
    with pytest.raises(ValueError, match="Cannot stitch an empty sequence"):
        stitch_light_curves([])


def test_stitch_light_curves_single(tmp_path: Path) -> None:
    path = tmp_path / "single_lc.fits"
    _write_tess_fixture(path)
    lc = read_tess_light_curve(path)
    stitched = stitch_light_curves([lc])
    assert stitched is lc


def test_stitch_light_curves_multiple(tmp_path: Path) -> None:
    path1 = tmp_path / "s1.fits"
    path2 = tmp_path / "s2.fits"
    _write_tess_fixture(path1)

    # Write a second fixture with sector 43 and time offset
    columns = [
        fits.Column(name="TIME", format="D", array=np.array([4.0, 5.0])),
        fits.Column(name="PDCSAP_FLUX", format="D", unit="electron / s", array=np.array([97.0, 96.0])),
        fits.Column(name="PDCSAP_FLUX_ERR", format="D", array=np.array([1.0, 1.0])),
        fits.Column(name="SAP_FLUX", format="D", array=np.array([98.0, 97.0])),
        fits.Column(name="QUALITY", format="K", array=np.array([0, 0])),
        fits.Column(name="CADENCENO", format="K", array=np.array([13, 14])),
        fits.Column(name="MOM_CENTR1", format="D", array=np.array([1.4, 1.5])),
        fits.Column(name="MOM_CENTR2", format="D", array=np.array([2.4, 2.5])),
    ]
    table = fits.BinTableHDU.from_columns(columns)
    table.header["TIMESYS"] = "TDB"
    table.header["BJDREFI"] = 2457000
    primary = fits.PrimaryHDU()
    primary.header["TICID"] = 123456
    primary.header["SECTOR"] = 43
    primary.header["MISSION"] = "TESS"
    primary.header["CREATOR"] = "fixture"
    fits.HDUList([primary, table]).writeto(path2)

    lc1 = read_tess_light_curve(path1)
    lc2 = read_tess_light_curve(path2)

    # Pass in reverse order to verify chronological sorting
    stitched = stitch_light_curves([lc2, lc1])
    assert stitched.provenance.product_type == "stitched_tess_light_curve"
    assert stitched.provenance.metadata["stitched_sectors"] == [42, 43]
    assert len(stitched.time_days) == 5
    # Check times are monotonic (excluding any NaN)
    valid_t = stitched.time_days[~np.isnan(stitched.time_days)]
    assert np.all(np.diff(valid_t) > 0)


def test_stitch_light_curves_mismatched_target(tmp_path: Path) -> None:
    path1 = tmp_path / "t1.fits"
    path2 = tmp_path / "t2.fits"
    _write_tess_fixture(path1)

    columns = [
        fits.Column(name="TIME", format="D", array=np.array([4.0])),
        fits.Column(name="PDCSAP_FLUX", format="D", unit="electron / s", array=np.array([97.0])),
    ]
    table = fits.BinTableHDU.from_columns(columns)
    table.header["TIMESYS"] = "TDB"
    primary = fits.PrimaryHDU()
    primary.header["TICID"] = 999999  # different target ID!
    fits.HDUList([primary, table]).writeto(path2)

    lc1 = read_tess_light_curve(path1)
    lc2 = read_tess_light_curve(path2)

    with pytest.raises(ValueError, match="Cannot stitch light curves from different targets"):
        stitch_light_curves([lc1, lc2])

