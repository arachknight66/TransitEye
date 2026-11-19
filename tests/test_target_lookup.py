from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

from transiteye.io.fits import read_kepler_light_curve
from transiteye.io.lookup import TargetReference, discover_target


def test_target_reference_requires_mission_for_bare_number() -> None:
    assert TargetReference.parse("TIC 123").canonical_id == "123"
    assert TargetReference.parse("kic 42").mission == "Kepler"
    assert TargetReference.parse("42", catalog="KIC").canonical_id == "42"
    with pytest.raises(ValueError, match="TIC or KIC"):
        TargetReference.parse("42")
    with pytest.raises(ValueError, match="TIC or KIC"):
        TargetReference.parse("EPIC 42")


def test_kepler_discovery_filters_exact_target_and_official_light_curves() -> None:
    class FakeArchive:
        def query_criteria(self, **kwargs: object) -> Table:
            assert kwargs == {"obs_collection": "Kepler", "target_name": "kplr000000042"}
            return Table({"obsid": [9]})

        def get_product_list(self, observations: Table) -> Table:
            assert len(observations) == 1
            return Table({
                "productFilename": ["kplr000000042-1_llc.fits", "kplr000000042-1_lpd-targ.fits", "kplr000000043-1_llc.fits"],
                "dataURI": ["mast:one", "mast:pixel", "mast:other"],
                "target_name": ["42", "42", "43"],
                "provenance_name": ["Kepler", "Kepler", "Kepler"],
            })

    products = discover_target(TargetReference.parse("KIC 42"), archive=FakeArchive())
    assert [(p.data_uri, p.mission, p.target_id) for p in products] == [
        ("mast:one", "Kepler", "42")
    ]


def test_tess_discovery_uses_exact_target_and_official_observations() -> None:
    class FakeArchive:
        def query_criteria(self, **kwargs: object) -> Table:
            assert kwargs == {"obs_collection": ["TESS", "HLSP"], "target_name": "42"}
            return Table({
                "obsid": [9, 10], "target_name": ["42", "42"],
                "provenance_name": ["SPOC", "COMMUNITY"],
                "sequence_number": [3, 3],
            })

        def get_product_list(self, observations: Table) -> Table:
            assert len(observations) == 1
            return Table({
                "parent_obsid": [9, 9],
                "productFilename": ["tess-s0003-42_lc.fits", "tess-s0003-42_tp.fits"],
                "dataURI": ["mast:light", "mast:pixel"],
            })

    products = discover_target(TargetReference.parse("TIC 42"), archive=FakeArchive())
    assert [(p.data_uri, p.observation, p.provider) for p in products] == [
        ("mast:light", 3, "SPOC")
    ]


def test_tess_discovery_includes_exact_target_tess_spoc_hlsp() -> None:
    class FakeArchive:
        def query_criteria(self, **kwargs: object) -> Table:
            assert kwargs == {"obs_collection": ["TESS", "HLSP"], "target_name": "42"}
            return Table({
                "obsid": [9, 10, 11, 12],
                "target_name": ["42", "42", "43", "42"],
                "provenance_name": ["SPOC", "TESS-SPOC", "TESS-SPOC", "COMMUNITY"],
                "sequence_number": [3, 3, 3, 3],
            })

        def get_product_list(self, observations: Table) -> Table:
            assert set(observations["obsid"]) == {9, 10}
            return Table({
                "parent_obsid": [9, 10, 11, 10],
                "productFilename": [
                    "tess-s0003-42_lc.fits", "hlsp_tess-spoc_42-s0003-lc.fits",
                    "hlsp_tess-spoc_43-s0003-lc.fits", "hlsp_tess-spoc_42-s0003-tp.fits",
                ],
                "dataURI": ["mast:spoc", "mast:hlsp", "mast:other", "mast:pixel"],
            })

    products = discover_target(TargetReference.parse("TIC 42"), archive=FakeArchive())
    assert [(p.data_uri, p.provider) for p in products] == [
        ("mast:hlsp", "TESS-SPOC"), ("mast:spoc", "SPOC"),
    ]


def test_kepler_reader_preserves_native_times_flags_and_reference(tmp_path: Path) -> None:
    source = tmp_path / "kplr000000042-1_llc.fits"
    primary = fits.PrimaryHDU()
    primary.header["KEPLERID"] = 42
    primary.header["QUARTER"] = 3
    primary.header["MISSION"] = "Kepler"
    table = fits.BinTableHDU.from_columns([
        fits.Column(name="TIME", format="D", array=np.array([1.0, 2.0, np.nan])),
        fits.Column(name="PDCSAP_FLUX", format="D", unit="e-/s", array=np.array([100., 99., 98.])),
        fits.Column(name="PDCSAP_FLUX_ERR", format="D", array=np.array([1., 1., 1.])),
        fits.Column(name="SAP_QUALITY", format="K", array=np.array([0, 1, 0])),
        fits.Column(name="CADENCENO", format="K", array=np.array([1, 2, 3])),
    ])
    table.header["BJDREFI"] = 2454833
    table.header["TIMESYS"] = "TDB"
    table.header["TIMEUNIT"] = "d"
    fits.HDUList([primary, table]).writeto(source)
    curve = read_kepler_light_curve(source)
    assert curve.provenance.target_id == "42"
    assert curve.provenance.time_reference == "2454833.0"
    assert curve.provenance.metadata["quarter"] == "3"
    assert curve.quality is not None and curve.quality.tolist() == [0, 1, 0]
    assert np.isnan(curve.time_days[-1])
    assert curve.provenance.checksum_sha256


def test_downloaded_product_rejects_wrong_target(tmp_path: Path) -> None:
    from transiteye.io.lookup import ArchiveProduct, verify_downloaded_product

    source = tmp_path / "kplr000000042-1_llc.fits"
    primary = fits.PrimaryHDU()
    primary.header["KEPLERID"] = 43
    table = fits.BinTableHDU.from_columns([
        fits.Column(name="TIME", format="D", array=np.array([1.])),
        fits.Column(name="SAP_FLUX", format="D", array=np.array([100.])),
    ])
    table.header["BJDREFI"] = 2454833
    table.header["TIMESYS"] = "TDB"
    table.header["TIMEUNIT"] = "d"
    fits.HDUList([primary, table]).writeto(source)
    product = ArchiveProduct("mast:one", source.name, "Kepler", "42", None, "Kepler")
    with pytest.raises(ValueError, match="target identity"):
        verify_downloaded_product(product, source)


def test_downloaded_product_rejects_wrong_mission(tmp_path: Path) -> None:
    from transiteye.io.lookup import ArchiveProduct, verify_downloaded_product

    source = tmp_path / "tess-s0001-42_lc.fits"
    primary = fits.PrimaryHDU()
    primary.header["TICID"] = 42
    primary.header["SECTOR"] = 1
    primary.header["MISSION"] = "Kepler"
    table = fits.BinTableHDU.from_columns([
        fits.Column(name="TIME", format="D", array=np.array([1.])),
        fits.Column(name="SAP_FLUX", format="D", array=np.array([100.])),
    ])
    table.header["BJDREFI"] = 2454833
    table.header["TIMESYS"] = "TDB"
    table.header["TIMEUNIT"] = "d"
    fits.HDUList([primary, table]).writeto(source)
    product = ArchiveProduct("mast:one", source.name, "TESS", "42", 1, "SPOC")
    with pytest.raises(ValueError, match="MISSION"):
        verify_downloaded_product(product, source)


def test_lookup_command_prints_inventory(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from transiteye import __main__
    from transiteye.io.lookup import ArchiveProduct

    monkeypatch.setattr(
        __main__, "discover_target",
        lambda target: (ArchiveProduct("mast:one", "kplr000000042-1_llc.fits", "Kepler", "42", None, "Kepler"),),
        raising=False,
    )
    assert __main__.main(["lookup-target", "KIC 42"]) == 0
    assert '"data_uri": "mast:one"' in capsys.readouterr().out


def test_acquisition_rejects_mismatched_fits_and_keeps_manifest(tmp_path: Path) -> None:
    from transiteye.config import AcquisitionConfig
    from transiteye.io.lookup import ArchiveProduct, acquire_target_products

    source = tmp_path / "wrong.fits"
    primary = fits.PrimaryHDU()
    primary.header["KEPLERID"] = 43
    table = fits.BinTableHDU.from_columns([
        fits.Column(name="TIME", format="D", array=np.array([1.])),
        fits.Column(name="SAP_FLUX", format="D", array=np.array([100.])),
    ])
    fits.HDUList([primary, table]).writeto(source)

    class FakeClient:
        def download(self, data_uri: str, destination: Path) -> None:
            assert data_uri == "mast:one"
            destination.write_bytes(source.read_bytes())

    product = ArchiveProduct("mast:one", "kplr000000042-1_llc.fits", "Kepler", "42", None, "Kepler")
    with pytest.raises(ValueError, match="target identity"):
        acquire_target_products(
            TargetReference.parse("KIC 42"), (product,), FakeClient(), tmp_path / "cache",
            AcquisitionConfig(max_attempts=1),
        )
    assert len(list((tmp_path / "cache").glob("*/manifest.json"))) == 1


def test_fetch_command_freezes_and_reports_verified_product(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from transiteye import __main__
    from transiteye.io.lookup import ArchiveProduct

    source = tmp_path / "source.fits"
    primary = fits.PrimaryHDU()
    primary.header["KEPLERID"] = 42
    primary.header["MISSION"] = "Kepler"
    table = fits.BinTableHDU.from_columns([
        fits.Column(name="TIME", format="D", array=np.array([1.])),
        fits.Column(name="SAP_FLUX", format="D", array=np.array([100.])),
    ])
    table.header["BJDREFI"] = 2454833
    table.header["TIMESYS"] = "TDB"
    table.header["TIMEUNIT"] = "d"
    fits.HDUList([primary, table]).writeto(source)

    product = ArchiveProduct("mast:one", "kplr000000042-1_llc.fits", "Kepler", "42", None, "Kepler")
    monkeypatch.setattr(__main__, "discover_target", lambda target: (product,))
    monkeypatch.setattr(
        __main__.AstroqueryMastClient, "download",
        lambda self, data_uri, destination: destination.write_bytes(source.read_bytes()),
    )
    config = tmp_path / "config.toml"
    config.write_text(f'[workspace]\nroot = "{tmp_path / "workspace"}"\n')
    assert __main__.main(["fetch-target", "KIC 42", "--config", str(config)]) == 0
    assert len(list((tmp_path / "workspace" / "cache").glob("*/manifest.json"))) == 1
    assert '"verified_count": 1' in capsys.readouterr().out


def test_acquisition_rejects_product_outside_requested_target(tmp_path: Path) -> None:
    from transiteye.config import AcquisitionConfig
    from transiteye.io.lookup import ArchiveProduct, acquire_target_products

    class NoDownload:
        def download(self, data_uri: str, destination: Path) -> None:
            raise AssertionError("invalid products must fail before network access")

    target = TargetReference.parse("KIC 42")
    with pytest.raises(ValueError, match="product identity"):
        acquire_target_products(
            target,
            (ArchiveProduct("mast:bad", "../escape.fits", "Kepler", "43", None, "Kepler"),),
            NoDownload(), tmp_path, AcquisitionConfig(max_attempts=1),
        )


def test_acquisition_rejects_duplicate_filename_before_download(tmp_path: Path) -> None:
    from transiteye.config import AcquisitionConfig
    from transiteye.io.lookup import ArchiveProduct, acquire_target_products

    class NoDownload:
        def download(self, data_uri: str, destination: Path) -> None:
            raise AssertionError("duplicate filenames must fail before download")

    products = (
        ArchiveProduct("mast:first", "same_lc.fits", "TESS", "42", 3, "SPOC"),
        ArchiveProduct("mast:second", "same_lc.fits", "TESS", "42", 3, "TESS-SPOC"),
    )
    with pytest.raises(ValueError, match="duplicate filenames"):
        acquire_target_products(
            TargetReference.parse("TIC 42"), products, NoDownload(), tmp_path,
            AcquisitionConfig(max_attempts=1),
        )
    assert not list(tmp_path.iterdir())


def test_lookup_manifest_v2_retains_provider_and_old_manifest_remains_readable(tmp_path: Path) -> None:
    from transiteye.io.manifest import ManifestEntry, ProductManifest, read_manifest, write_manifest

    old = ProductManifest.create(
        "mast", (ManifestEntry("old", "mast:old", "42", 1, "tess_light_curve"),)
    )
    old_path = tmp_path / "old.json"
    write_manifest(old, old_path)
    assert read_manifest(old_path).entries[0].provider is None

    entry = ManifestEntry(
        "new", "mast:new", "42", 1, "tess_light_curve", provider="SPOC"
    )
    new = ProductManifest.create("mast", (entry,), schema_version="2")
    new_path = tmp_path / "new.json"
    write_manifest(new, new_path)
    restored = read_manifest(new_path)
    assert restored.schema_version == "2"
    assert restored.entries[0].provider == "SPOC"


@pytest.fixture
def tess_download(tmp_path: Path) -> bytes:
    primary = fits.PrimaryHDU()
    primary.header["TICID"] = 42
    primary.header["SECTOR"] = 3
    primary.header["MISSION"] = "TESS"
    table = fits.BinTableHDU.from_columns([
        fits.Column(name="TIME", format="D", array=np.array([1.0, 2.0, 3.0])),
        fits.Column(name="SAP_FLUX", format="D", array=np.array([100.0, 99.0, 98.0])),
        fits.Column(name="SAP_FLUX_ERR", format="D", array=np.ones(3)),
        fits.Column(name="QUALITY", format="K", array=np.zeros(3, dtype=int)),
    ])
    table.header["BJDREFI"] = 2457000
    table.header["TIMESYS"] = "TDB"
    table.header["TIMEUNIT"] = "d"
    source = tmp_path / "source.fits"
    fits.HDUList([primary, table]).writeto(source)
    return source.read_bytes()


@pytest.mark.parametrize("time_unit", [None, "s", "BJD - 2454833"])
def test_verified_download_requires_explicit_consistent_day_units(
    tmp_path: Path, tess_download: bytes, time_unit: str | None
) -> None:
    from transiteye.io.lookup import ArchiveProduct, verify_downloaded_product

    source = tmp_path / "changed.fits"
    source.write_bytes(tess_download)
    with fits.open(source, mode="update") as hdus:
        if time_unit is None:
            del hdus[1].header["TIMEUNIT"]
        else:
            hdus[1].header["TIMEUNIT"] = time_unit
    product = ArchiveProduct("mast:light", source.name, "TESS", "42", 3, "SPOC")
    with pytest.raises(ValueError, match="TIME unit"):
        verify_downloaded_product(product, source)


class LocalArchive:
    def __init__(self, payloads: list[bytes]) -> None:
        self.payloads = iter(payloads)
        self.download_calls = 0

    def download(self, data_uri: str, destination: Path) -> None:
        assert data_uri == "mast:light"
        self.download_calls += 1
        destination.write_bytes(next(self.payloads))


def test_lookup_freezes_receipt_and_manifest_through_cache_reuse_and_retry(
    tmp_path: Path, tess_download: bytes
) -> None:
    from transiteye.config import AcquisitionConfig
    from transiteye.io.lookup import ArchiveProduct, acquire_target_products

    product = ArchiveProduct("mast:light", "tess-s0003-42_lc.fits", "TESS", "42", 3, "SPOC")
    client = LocalArchive([tess_download, b"broken transfer", tess_download])
    config = AcquisitionConfig(max_attempts=2, retry_delay_seconds=0)
    target = TargetReference.parse("TIC 42")

    manifest, completed = acquire_target_products(target, (product,), client, tmp_path, config)
    cached = completed["mast:light"]
    receipt_path = cached.with_suffix(".fits.receipt.json")
    receipt_bytes = receipt_path.read_bytes()
    receipt = json.loads(receipt_bytes)
    assert receipt["sha256"] == hashlib.sha256(tess_download).hexdigest()
    assert receipt["manifest_id"] == manifest.manifest_id
    assert receipt["data_uri"] == product.data_uri
    assert receipt["filename"] == product.filename
    assert receipt["mission"] == "TESS"
    assert receipt["target_id"] == "42"
    assert receipt["sector"] == 3
    assert receipt["provider"] == "SPOC"
    assert receipt["verified_at"]
    manifest_path = cached.parent / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()

    reused, _ = acquire_target_products(target, (product,), client, tmp_path, config)
    assert client.download_calls == 1
    assert reused == manifest

    # Change a data value and retain a readable FITS with the same archive identity.
    with fits.open(cached, mode="update") as hdus:
        hdus[1].data["SAP_FLUX"][0] = 101.0
    repaired, _ = acquire_target_products(target, (product,), client, tmp_path, config)
    assert client.download_calls == 3
    assert cached.read_bytes() == tess_download
    assert repaired == manifest
    assert receipt_path.read_bytes() == receipt_bytes
    assert manifest_path.read_bytes() == manifest_bytes
    assert not list(cached.parent.glob("*.part"))


def test_inspect_and_fetch_refuse_changed_payload_pinned_by_receipt(
    tmp_path: Path, tess_download: bytes, capsys: pytest.CaptureFixture[str]
) -> None:
    from transiteye import __main__
    from transiteye.config import AcquisitionConfig
    from transiteye.io.lookup import ArchiveProduct, acquire_target_products

    product = ArchiveProduct("mast:light", "tess-s0003-42_lc.fits", "TESS", "42", 3, "SPOC")
    config = AcquisitionConfig(max_attempts=1)
    target = TargetReference.parse("TIC 42")
    _, completed = acquire_target_products(
        target, (product,), LocalArchive([tess_download]), tmp_path, config
    )
    cached = completed["mast:light"]
    receipt_path = cached.with_suffix(".fits.receipt.json")
    receipt_bytes = receipt_path.read_bytes()
    manifest_path = cached.parent / "manifest.json"
    assert __main__.main(["inspect-target-set", str(manifest_path)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["observations"][0]["input_sha256"] == hashlib.sha256(tess_download).hexdigest()

    with fits.open(cached, mode="update") as hdus:
        hdus[1].data["SAP_FLUX"][0] = 101.0
    with pytest.raises(ValueError, match="Checksum mismatch"):
        __main__.main(["inspect-target-set", str(manifest_path)])
    with pytest.raises(ValueError, match="Checksum mismatch"):
        acquire_target_products(
            target, (product,), LocalArchive([cached.read_bytes()]), tmp_path, config
        )
    assert receipt_path.read_bytes() == receipt_bytes


def test_lookup_refetches_legacy_cache_without_receipt_and_rejects_malformed_receipt(
    tmp_path: Path, tess_download: bytes
) -> None:
    from transiteye import __main__
    from transiteye.config import AcquisitionConfig
    from transiteye.io.lookup import ArchiveProduct, acquire_target_products

    product = ArchiveProduct("mast:light", "tess-s0003-42_lc.fits", "TESS", "42", 3, "SPOC")
    config = AcquisitionConfig(max_attempts=1)
    target = TargetReference.parse("TIC 42")
    client = LocalArchive([tess_download, tess_download])
    _, completed = acquire_target_products(target, (product,), client, tmp_path, config)
    cached = completed["mast:light"]
    receipt_path = cached.with_suffix(".fits.receipt.json")
    receipt_path.unlink()
    with pytest.raises(ValueError, match="Missing acquisition receipt"):
        __main__.main(["inspect-target-set", str(cached.parent / "manifest.json")])
    acquire_target_products(target, (product,), client, tmp_path, config)
    assert client.download_calls == 2

    receipt = json.loads(receipt_path.read_bytes())
    receipt["data_uri"] = "mast:wrong"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="receipt identity"):
        acquire_target_products(target, (product,), client, tmp_path, config)
    assert client.download_calls == 2


def test_acquire_manifest_cli_replays_v2_with_receipts(
    tmp_path: Path, tess_download: bytes, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from transiteye import __main__
    from transiteye.io.manifest import ManifestEntry, ProductManifest, write_manifest

    workspace = tmp_path / "workspace"
    config = tmp_path / "config.toml"
    config.write_text(f'[workspace]\nroot = "{workspace}"\n')
    entry = ManifestEntry(
        "mast:light", "mast:light", "42", 3, "tess_light_curve",
        local_filename="tess-s0003-42_lc.fits", provider="SPOC",
    )
    manifest = ProductManifest.create(
        "mast", (entry,), {"catalog": "TIC", "target_id": "42"}, schema_version="2",
    )
    path = tmp_path / "inventory.json"
    write_manifest(manifest, path)
    downloads: list[str] = []

    def download(_: object, data_uri: str, destination: Path) -> None:
        downloads.append(data_uri)
        destination.write_bytes(tess_download)

    monkeypatch.setattr(__main__.AstroqueryMastClient, "download", download)
    assert __main__.main(["acquire-manifest", str(path), "--config", str(config)]) == 0
    capsys.readouterr()
    cached = workspace / "cache" / manifest.manifest_id / entry.resolved_filename()
    receipt = cached.with_name(f"{cached.name}.receipt.json")
    assert json.loads(receipt.read_text())["sha256"] == hashlib.sha256(tess_download).hexdigest()
    assert __main__.main(["acquire-manifest", str(path), "--config", str(config)]) == 0
    capsys.readouterr()
    assert downloads == ["mast:light"]

    document = json.loads(receipt.read_text())
    document["target_id"] = "43"
    receipt.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="receipt identity"):
        __main__.main(["acquire-manifest", str(path), "--config", str(config)])
    assert downloads == ["mast:light"]
