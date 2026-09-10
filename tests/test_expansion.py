"""Offline tests for frozen real-cohort expansion behavior."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import transiteye.expansion as expansion_module
from transiteye.acquisition.downloader import DownloadReceipt
from transiteye.acquisition.expansion import (
    ExpansionManifestError,
    build_expansion_manifest,
    load_expansion_manifest,
    write_expansion_manifest,
)
from transiteye.config import BlsSettings, ExpansionSettings, PreprocessingSettings
from transiteye.expansion import (
    _detect_selected,
    _preprocess_selected,
    run_frozen_cohort_expansion,
)


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


def test_downloader_records_success_and_verifies_local_cache_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    products = _products().iloc[[0]].copy()
    calls: list[str] = []

    def fake_download(
        product: pd.Series, *, fetcher: object, raw_root: Path, known_checksum: str | None = None
    ) -> DownloadReceipt:
        calls.append("reuse" if known_checksum else "download")
        return DownloadReceipt(
            object_id=str(product["object_id"]),
            sector=int(product["sector"]),
            product_filename=str(product["product_filename"]),
            data_uri=str(product["data_uri"]),
            relative_raw_path="tess/tic-1/sector-0001/one.fits",
            expected_size_bytes=1,
            actual_size_bytes=1,
            sha256="a" * 64,
            downloaded_at_utc="2026-01-01T00:00:00Z",
            status="reused" if known_checksum else "downloaded",
        )

    monkeypatch.setattr(expansion_module, "AstroqueryProductFetcher", lambda: object())
    monkeypatch.setattr(expansion_module, "download_product", fake_download)
    receipts, reuse_count = expansion_module._download_selected(
        products,
        root=tmp_path,
        manifest_root=tmp_path / "manifest",
        manifest_id="expansion-test",
    )
    assert receipts.loc[0, "status"] == "downloaded"
    assert reuse_count == 1
    assert calls == ["download", "reuse"]


def test_preprocessing_success_preserves_configured_raw_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checksum = "b" * 64
    raw_path = tmp_path / "data/raw/tess/tic-1/sector-0001/one.fits"
    raw_path.parent.mkdir(parents=True)
    raw_path.write_bytes(b"raw")
    receipts = pd.DataFrame(
        [
            {
                "object_id": "tic-1",
                "sector": 1,
                "data_uri": "mast:one",
                "relative_raw_path": "tess/tic-1/sector-0001/one.fits",
                "sha256": checksum,
                "status": "downloaded",
            }
        ]
    )
    products = _products().iloc[[0]].copy().assign(data_uri="mast:one")
    monkeypatch.setattr(
        expansion_module, "read_tess_lightcurve", lambda path, observation_id: object()
    )
    monkeypatch.setattr(
        expansion_module,
        "preprocess",
        lambda source, **kwargs: SimpleNamespace(
            data=pd.DataFrame({"time": [1.0], "detrended_flux": [1.0]}),
            metadata={"source_raw_sha256": checksum},
        ),
    )
    monkeypatch.setattr(expansion_module, "plot_preprocessing_qa", lambda processed, path: None)
    output = _preprocess_selected(
        receipts,
        products,
        root=tmp_path,
        manifest_root=tmp_path / "manifest",
        preprocessing_hash="c" * 20,
        settings=PreprocessingSettings(
            flux_stream="PDCSAP_FLUX",
            gap_days=0.5,
            trend_window_cadences=101,
            positive_spike_mad=8.0,
        ),
        manifest_id="expansion-test",
    )
    assert output.loc[0, "status"] == "processed"
    assert output.loc[0, "source_raw_checksum"] == checksum
    metadata = json.loads(
        (tmp_path / f"data/processed/mvp/{checksum[:20]}/metadata.json").read_text()
    )
    assert metadata["preprocessing_config_hash"] == "c" * 20


def test_detection_freezes_before_post_detection_matching(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checksum = "d" * 64
    processed_path = tmp_path / f"data/processed/mvp/{checksum[:20]}/cadences.parquet"
    processed_path.parent.mkdir(parents=True)
    pd.DataFrame({"time": [1.0], "detrended_flux": [1.0]}).to_parquet(processed_path)
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
    preprocessing_hash = "e" * 20
    expected_hash = expansion_module.content_hash(
        {"bls": settings.model_dump(mode="json"), "preprocessing_hash": preprocessing_hash}
    )
    order: list[str] = []
    monkeypatch.setattr(
        expansion_module,
        "run_bls",
        lambda *args, **kwargs: SimpleNamespace(
            config_hash=expected_hash, periodogram=pd.DataFrame({"period": [2.0], "power": [1.0]})
        ),
    )
    monkeypatch.setattr(
        expansion_module,
        "extract_peaks",
        lambda result, settings: pd.DataFrame(
            [{"candidate_id": "candidate-1", "rank": 1, "period": 2.0}]
        ),
    )

    def fake_match(
        candidates: pd.DataFrame, events: pd.DataFrame, settings: BlsSettings
    ) -> pd.DataFrame:
        order.append("match")
        assert candidates.iloc[0]["candidate_id"] == "candidate-1"
        return pd.DataFrame([{"candidate_id": "candidate-1", "toi_id": "100.01", "matched": True}])

    monkeypatch.setattr(expansion_module, "match_candidates", fake_match)
    receipts = pd.DataFrame(
        [
            {
                "object_id": "tic-1",
                "data_uri": "mast:one",
                "source_raw_checksum": checksum,
                "processed_checksum": "f" * 64,
                "preprocessing_config_hash": preprocessing_hash,
                "status": "processed",
            }
        ]
    )
    products = _products().iloc[[0]].copy().assign(data_uri="mast:one")
    events = pd.DataFrame(
        [
            {
                "object_id": "tic-1",
                "toi_id": "100.01",
                "period_days": 2.0,
                "transit_epoch_bjd": 1.0,
                "source_disposition": "CP",
            }
        ]
    )
    detections, candidates, matches, result_hash = _detect_selected(
        receipts,
        products,
        events,
        root=tmp_path,
        manifest_root=tmp_path / "manifest",
        preprocessing_hash=preprocessing_hash,
        settings=settings,
    )
    assert detections.loc[0, "status"] == "success"
    assert result_hash == expected_hash
    assert len(candidates) == len(matches) == 1
    assert order == ["match"]


def test_frozen_real_expansion_replays_from_local_artifacts() -> None:
    """A completed expansion must reuse local immutable artifacts without network access."""
    root = Path(__file__).resolve().parents[1]
    required = root / "data/manifests/expansion-be59541b9388b685e1f9/metadata.json"
    if not required.is_file():
        pytest.skip("Frozen real expansion artifacts are not present in this checkout.")
    result = run_frozen_cohort_expansion(root)
    assert result.manifest_id == "expansion-be59541b9388b685e1f9"
    assert result.dataset_version == "dataset-4b84e8acaa6f6b334c2f"
    assert result.selected_products == 15
    assert result.preprocessed_products == 15
    assert result.searched_products == 15
    assert result.qa_summary["coverage"]["bls_searched_tics"] == 8
    assert result.qa_summary["splits"]["zero_tic_overlap"] is True
