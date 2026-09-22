from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from astropy.table import Table

from transiteye.config import AcquisitionConfig
from transiteye.io.manifest import ManifestEntry, ProductManifest, read_manifest, write_manifest
from transiteye.io.mast import (
    AstroqueryMastClient,
    Catalogs,
    Observations,
    acquire_manifest,
    build_manifest,
)


class FakeArchiveClient:
    def __init__(self) -> None:
        self.download_calls = 0

    def discover(self, target_id: str, sector: int | None = None) -> tuple[ManifestEntry, ...]:
        return (
            ManifestEntry(
                product_id="product-1",
                data_uri="mast:product-1",
                target_id=target_id,
                sector=sector,
                product_type="tess_light_curve",
                local_filename="product-1.fits",
            ),
        )

    def download(self, data_uri: str, destination: Path) -> None:
        self.download_calls += 1
        destination.write_bytes(b"fixture-data")


def test_manifest_round_trip_is_stable(tmp_path: Path) -> None:
    entry = ManifestEntry("product-1", "mast:product-1", "123", 1, "tess_light_curve")
    manifest = ProductManifest.create("mast", (entry,), {"target_id": "123"})
    path = tmp_path / "manifest.json"
    write_manifest(manifest, path)
    assert read_manifest(path) == manifest


def test_build_manifest_freezes_client_discovery() -> None:
    manifest = build_manifest(FakeArchiveClient(), "123", sector=5)
    assert manifest.entries[0].target_id == "123"
    assert manifest.entries[0].sector == 5


def test_astroquery_client_discovers_tess_light_curves_by_tic_coordinate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def query_catalog(**kwargs: object) -> Table:
        captured["catalog"] = kwargs
        return Table({"ra": [1.5], "dec": [-2.5]})

    def query_observations(**kwargs: object) -> Table:
        captured["observations"] = kwargs
        return Table({"obsid": [1]})

    def products(_: Table) -> Table:
        return Table(
            {
                "productFilename": ["tess-s0042_lc.fits", "tess-s0042_tp.fits"],
                "dataURI": ["mast:light", "mast:pixel"],
            }
        )

    monkeypatch.setattr(Catalogs, "query_criteria", query_catalog)
    monkeypatch.setattr(Observations, "query_criteria", query_observations)
    monkeypatch.setattr(Observations, "get_product_list", products)
    entries = AstroqueryMastClient().discover("123", sector=42)
    assert len(entries) == 1
    assert entries[0].data_uri == "mast:light"
    assert captured["catalog"] == {"catalog": "TIC", "ID": 123}
    assert captured["observations"] == {
        "obs_collection": "TESS",
        "coordinates": "1.5 -2.5",
        "radius": "0.0001 deg",
    }


def test_acquisition_retries_and_reuses_valid_cache(tmp_path: Path) -> None:
    payload = b"fixture-data"
    checksum = hashlib.sha256(payload).hexdigest()
    entry = ManifestEntry(
        "product-1",
        "mast:product-1",
        "123",
        1,
        "tess_light_curve",
        expected_sha256=checksum,
        local_filename="product-1.fits",
    )
    manifest = ProductManifest.create("mast", (entry,))
    client = FakeArchiveClient()
    config = AcquisitionConfig(max_attempts=2, retry_delay_seconds=0)
    completed = acquire_manifest(client, manifest, tmp_path, config)
    assert completed["product-1"].read_bytes() == payload
    assert client.download_calls == 1
    acquire_manifest(client, manifest, tmp_path, config)
    assert client.download_calls == 1


def test_acquisition_rejects_checksum_mismatch(tmp_path: Path) -> None:
    entry = ManifestEntry(
        "product-1",
        "mast:product-1",
        "123",
        1,
        "tess_light_curve",
        expected_sha256="0" * 64,
    )
    manifest = ProductManifest.create("mast", (entry,))
    with pytest.raises(ValueError, match="Checksum mismatch"):
        acquire_manifest(FakeArchiveClient(), manifest, tmp_path, AcquisitionConfig(max_attempts=1))
