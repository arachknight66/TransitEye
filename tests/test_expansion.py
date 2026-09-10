"""Offline tests for frozen real-cohort expansion behavior."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from transiteye.acquisition.expansion import (
    ExpansionManifestError,
    build_expansion_manifest,
    load_expansion_manifest,
    write_expansion_manifest,
)
from transiteye.config import BlsSettings, ExpansionSettings, PreprocessingSettings
from transiteye.expansion import _detect_selected, _preprocess_selected


def _settings() -> ExpansionSettings:
    return ExpansionSettings(
        product_snapshot_id="mast-products-example",
        required_author="TESS-SPOC",
        required_product_suffix="_lc.fits",
        preferred_exposure_seconds=200.0,
        sectors_per_tic=2,
        sector_order="earliest",
    )


def _products() -> pd.DataFrame:
    rows = []
    for object_id in ("tic-1", "tic-2"):
        for sector in (1, 2, 3):
            rows.append(
                {
                    "object_id": object_id,
                    "mast_obs_id": f"obs-{object_id}-{sector}",
                    "sector": sector,
                    "author": "TESS-SPOC",
                    "exposure_seconds": 200.0,
                    "product_filename": f"{object_id}-{sector}_lc.fits",
                    "data_uri": f"mast:{object_id}-{sector}",
                    "product_size_bytes": sector,
                    "data_release": 1.0,
                    "catalog_period": 99.0,
                    "source_disposition": "CP",
                }
            )
    rows.append(
        {
            "object_id": "tic-1",
            "mast_obs_id": "wrong-cadence",
            "sector": 4,
            "author": "TESS-SPOC",
            "exposure_seconds": 600.0,
            "product_filename": "wrong_lc.fits",
            "data_uri": "mast:wrong",
            "product_size_bytes": 1,
            "data_release": 1.0,
            "catalog_period": 1.0,
            "source_disposition": "FP",
        }
    )
    return pd.DataFrame(rows)


def test_expansion_manifest_is_deterministic_label_and_ephemeris_independent(
    tmp_path: Path,
) -> None:
    first = build_expansion_manifest(
        _products(), settings=_settings(), product_snapshot_sha256="a" * 64
    )
    altered = _products().assign(catalog_period=-123.0, source_disposition="FA")
    second = build_expansion_manifest(
        altered, settings=_settings(), product_snapshot_sha256="a" * 64
    )
    assert first.manifest_id == second.manifest_id
    assert first.selected_products[["object_id", "sector"]].to_dict("records") == [
        {"object_id": "tic-1", "sector": 1},
        {"object_id": "tic-1", "sector": 2},
        {"object_id": "tic-2", "sector": 1},
        {"object_id": "tic-2", "sector": 2},
    ]
    assert set(first.selected_products["exposure_seconds"]) == {200.0}
    directory = write_expansion_manifest(first, root=tmp_path)
    _, loaded = load_expansion_manifest(directory)
    assert loaded.manifest_id == first.manifest_id
    assert write_expansion_manifest(first, root=tmp_path) == directory
    (directory / "selected_products.parquet").write_bytes(b"corrupt")
    with pytest.raises(ExpansionManifestError, match="checksum"):
        load_expansion_manifest(directory)


def test_preprocessing_failure_is_recorded_without_silent_row_loss(tmp_path: Path) -> None:
    receipts = pd.DataFrame(
        [
            {
                "object_id": "tic-1",
                "sector": 1,
                "data_uri": "mast:one",
                "relative_raw_path": "tess/tic-1/sector-0001/one.fits",
                "sha256": "1" * 64,
                "status": "downloaded",
            }
        ]
    )
    products = _products().iloc[[0]].copy()
    products["data_uri"] = "mast:one"
    output = _preprocess_selected(
        receipts,
        products,
        root=tmp_path,
        manifest_root=tmp_path / "manifest",
        preprocessing_hash="a" * 20,
        settings=PreprocessingSettings(
            flux_stream="PDCSAP_FLUX",
            gap_days=0.5,
            trend_window_cadences=101,
            positive_spike_mad=8.0,
        ),
        manifest_id="expansion-test",
    )
    assert len(output) == 1
    assert output.loc[0, "status"] == "failed"
    assert output.loc[0, "source_raw_checksum"] == "1" * 64


def test_bls_failure_is_recorded_without_candidate_creation(tmp_path: Path) -> None:
    checksum = "2" * 64
    processed_dir = tmp_path / "data/processed/mvp" / checksum[:20]
    processed_dir.mkdir(parents=True)
    pd.DataFrame({"not_time": [1.0]}).to_parquet(processed_dir / "cadences.parquet")
    receipts = pd.DataFrame(
        [
            {
                "object_id": "tic-1",
                "data_uri": "mast:one",
                "source_raw_checksum": checksum,
                "processed_checksum": "3" * 64,
                "preprocessing_config_hash": "a" * 20,
                "status": "processed",
            }
        ]
    )
    settings = BlsSettings(
        min_period_days=0.5,
        max_period_days=20.0,
        min_transits=2,
        durations_days=(0.04, 0.08),
        frequency_factor=5.0,
        top_k=5,
        local_peak_fraction=0.01,
        harmonic_tolerance=0.02,
        period_match_tolerance=0.02,
        phase_match_tolerance=0.1,
    )
    detections, candidates, matches, _ = _detect_selected(
        receipts,
        _products().iloc[[0]].assign(data_uri="mast:one"),
        pd.DataFrame(columns=["object_id", "toi_id", "period_days", "transit_epoch_bjd"]),
        root=tmp_path,
        manifest_root=tmp_path / "manifest",
        preprocessing_hash="a" * 20,
        settings=settings,
    )
    assert detections.loc[0, "status"] == "failed"
    assert not candidates and not matches
