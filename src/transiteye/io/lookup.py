"""Mission-qualified target lookup against official MAST light-curve products."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from astroquery.mast import Observations

from transiteye.config import AcquisitionConfig
from transiteye.io.fits import (
    read_kepler_light_curve,
    read_tess_light_curve,
    validate_day_time_axis,
)
from transiteye.io.manifest import ManifestEntry, ProductManifest, freeze_manifest
from transiteye.io.mast import acquire_manifest


@dataclass(frozen=True, slots=True)
class TargetReference:
    catalog: str
    canonical_id: str
    mission: str

    @classmethod
    def parse(cls, value: str, *, catalog: str | None = None) -> TargetReference:
        match = re.fullmatch(r"\s*(?:(TIC|KIC)\s*[-:]?\s*)?(\d+)\s*", value, re.I)
        if match is None:
            raise ValueError("Enter a TIC or KIC followed by a positive numeric ID.")
        qualified = match.group(1)
        requested = catalog.upper() if catalog is not None else None
        if requested not in (None, "TIC", "KIC") or (qualified and requested and qualified.upper() != requested):
            raise ValueError("Choose a matching TIC or KIC catalog.")
        namespace = qualified.upper() if qualified else requested
        if namespace is None:
            raise ValueError("Choose TIC or KIC for a bare numeric ID.")
        identifier = int(match.group(2))
        if identifier <= 0:
            raise ValueError("TIC or KIC ID must be positive.")
        return cls(namespace, str(identifier), "TESS" if namespace == "TIC" else "Kepler")


@dataclass(frozen=True, slots=True)
class ArchiveProduct:
    data_uri: str
    filename: str
    mission: str
    target_id: str
    observation: int | None
    provider: str


def discover_target(
    target: TargetReference,
    *,
    archive: Any = Observations,
) -> tuple[ArchiveProduct, ...]:
    """List only official, target-identical light-curve products.

    MAST target_name uses bare TIC numbers and zero-padded kplr names. Product
    identity is checked against these exact observations and later against FITS.
    """
    if target.catalog == "TIC":
        observations = archive.query_criteria(
            obs_collection=["TESS", "HLSP"], target_name=target.canonical_id
        )
        if len(observations) == 0:
            return ()
        selected = observations[[
            str(row["target_name"]) == target.canonical_id
            and str(row["provenance_name"]) in ("SPOC", "TESS-SPOC")
            for row in observations
        ]]
        if len(selected) == 0:
            return ()
        by_obsid = {
            str(row["obsid"]): (str(row["provenance_name"]), int(row["sequence_number"]))
            for row in selected
        }
        discovered_tess: dict[str, ArchiveProduct] = {}
        for product in archive.get_product_list(selected):
            parent = by_obsid.get(str(product["parent_obsid"]))
            filename = str(product["productFilename"])
            if parent is None or not filename.endswith(("_lc.fits", "-lc.fits")):
                continue
            if Path(filename).name != filename:
                continue
            uri = str(product["dataURI"])
            discovered_tess[uri] = ArchiveProduct(
                uri, filename, "TESS", target.canonical_id, parent[1], parent[0]
            )
        return tuple(sorted(discovered_tess.values(), key=lambda item: (item.observation or 0, item.filename)))
    if target.catalog != "KIC":
        raise ValueError(f"Unsupported target catalog: {target.catalog}")
    observations = archive.query_criteria(
        obs_collection="Kepler", target_name=f"kplr{int(target.canonical_id):09d}"
    )
    if len(observations) == 0:
        return ()
    products = archive.get_product_list(observations)
    discovered: dict[str, ArchiveProduct] = {}
    for row in products:
        filename = str(row["productFilename"])
        if Path(filename).name != filename:
            continue
        match = re.fullmatch(r"kplr(\d+)-[^/]+_([ls])lc\.fits", filename, re.I)
        if match is None or str(int(match.group(1))) != target.canonical_id:
            continue
        provider = str(row["provenance_name"]) if "provenance_name" in products.colnames else "Kepler"
        if provider.casefold() != "kepler":
            continue
        uri = str(row["dataURI"])
        discovered[uri] = ArchiveProduct(
            uri, filename, "Kepler", target.canonical_id, None, "Kepler"
        )
    return tuple(sorted(discovered.values(), key=lambda item: (item.filename, item.data_uri)))


def verify_downloaded_product(product: ArchiveProduct, path: Path) -> str:
    """Check archive-file identity and return the file's recorded SHA-256."""
    if product.mission == "Kepler":
        curve = read_kepler_light_curve(path)
        if product.observation is not None and curve.provenance.metadata.get("quarter") != str(product.observation):
            raise ValueError(f"Archive product observation mismatch: {path}")
    elif product.mission == "TESS":
        curve = read_tess_light_curve(path)
        if product.observation is not None and curve.provenance.sector != product.observation:
            raise ValueError(f"Archive product observation mismatch: {path}")
    else:
        raise ValueError(f"Unsupported mission: {product.mission}")
    if curve.provenance.metadata.get("mission") != product.mission:
        raise ValueError(f"Archive product mission mismatch: {path}")
    if curve.provenance.target_id != product.target_id:
        raise ValueError(f"Archive product target identity mismatch: {path}")
    validate_day_time_axis(curve)
    return curve.provenance.checksum_sha256 or ""


def acquire_target_products(
    target: TargetReference,
    products: tuple[ArchiveProduct, ...],
    client: Any,
    cache_root: Path,
    config: AcquisitionConfig,
) -> tuple[ProductManifest, dict[str, Path]]:
    """Freeze inventory before downloading and verify each FITS before accepting it."""
    if any(
        product.target_id != target.canonical_id
        or product.mission != target.mission
        or Path(product.filename).name != product.filename
        or not product.filename.lower().endswith(".fits")
        for product in products
    ):
        raise ValueError("Archive product identity or filename conflicts with the requested target.")
    if len({product.data_uri for product in products}) != len(products):
        raise ValueError("Archive product identity includes duplicate data URIs.")
    if len({product.filename for product in products}) != len(products):
        raise ValueError("Archive product identity includes duplicate filenames.")
    entries = tuple(
        ManifestEntry(
            product_id=product.data_uri.replace("/", "_"),
            data_uri=product.data_uri,
            target_id=target.canonical_id,
            sector=product.observation if target.mission == "TESS" else None,
            product_type="tess_light_curve" if target.mission == "TESS" else "kepler_light_curve",
            local_filename=product.filename,
            provider=product.provider,
        )
        for product in products
    )
    if len({entry.product_id for entry in entries}) != len(entries):
        raise ValueError("Archive product identity includes colliding product IDs.")
    manifest = ProductManifest.create(
        "mast", entries, {"catalog": target.catalog, "target_id": target.canonical_id},
        schema_version="2",
    )
    manifest_root = cache_root / manifest.manifest_id
    manifest = freeze_manifest(manifest, manifest_root / "manifest.json")
    indexed = {entry.product_id: product for entry, product in zip(entries, products, strict=True)}

    def _validate(entry: ManifestEntry, path: Path) -> None:
        verify_downloaded_product(indexed[entry.product_id], path)

    completed = acquire_manifest(
        client, manifest, cache_root, config, validate=_validate, with_receipts=True
    )
    return manifest, completed


def acquire_frozen_target_manifest(
    manifest: ProductManifest,
    client: Any,
    cache_root: Path,
    config: AcquisitionConfig,
) -> dict[str, Path]:
    """Replay a v2 target inventory with the same checks as a fresh lookup."""
    if manifest.schema_version != "2" or manifest.source != "mast":
        raise ValueError("Expected a v2 MAST target lookup manifest.")
    catalog = manifest.query.get("catalog")
    target_id = manifest.query.get("target_id")
    if catalog not in ("TIC", "KIC") or target_id is None:
        raise ValueError("Expected a qualified TIC/KIC target lookup manifest.")
    target = TargetReference.parse(target_id, catalog=catalog)
    products: dict[str, ArchiveProduct] = {}
    filenames: set[str] = set()
    uris: set[str] = set()
    for entry in manifest.entries:
        filename = entry.resolved_filename()
        expected_type = "tess_light_curve" if target.mission == "TESS" else "kepler_light_curve"
        if (
            entry.product_type != expected_type
            or entry.target_id != target.canonical_id
            or Path(filename).name != filename
            or not filename.lower().endswith(".fits")
            or entry.provider is None
        ):
            raise ValueError("Frozen target product identity or filename is invalid.")
        if filename in filenames or entry.data_uri in uris:
            raise ValueError("Frozen target inventory has duplicate filenames or data URIs.")
        filenames.add(filename)
        uris.add(entry.data_uri)
        products[entry.product_id] = ArchiveProduct(
            entry.data_uri, filename, target.mission, entry.target_id,
            entry.sector, entry.provider,
        )
    return acquire_manifest(
        client, manifest, cache_root, config,
        validate=lambda entry, path: verify_downloaded_product(products[entry.product_id], path),
        with_receipts=True,
    )
