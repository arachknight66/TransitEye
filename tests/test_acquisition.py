"""Offline tests for B012--B015 acquisition contracts."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from transiteye.acquisition.downloader import (
    DownloadIntegrityError,
    download_product,
    download_products,
)
from transiteye.acquisition.manifest import (
    AcquisitionManifestError,
    apply_product_policy,
    build_acquisition_plan,
    load_acquisition_plan,
    select_download_pilot,
    write_acquisition_plan,
)
from transiteye.acquisition.mast import (
    MastSearchError,
    mast_query_parameters,
    normalize_mast_observations,
    normalize_mast_product_search,
    search_tess_products,
)
from transiteye.config import AcquisitionSettings, load_config


@pytest.fixture
def settings() -> AcquisitionSettings:
    return AcquisitionSettings(
        source="mast",
        mission="TESS",
        product_kind="lightcurve",
        preferred_author="SPOC",
        preferred_exposure_seconds=None,
        sector_policy="all_available",
        duplicate_resolution="latest_release",
        allow_fallback=False,
        download_pilot_target_limit=3,
        download_pilot_sectors_per_target=1,
    )


@pytest.fixture
def observations() -> list[dict[str, object]]:
    return [
        {
            "target_name": "100",
            "obsid": 101,
            "obs_id": "tess-s0001-tic-100",
            "sequence_number": 1,
            "provenance_name": "SPOC",
            "t_exptime": 120.0,
            "dataReleaseDate": "2020-01-01",
        },
        {
            "target_name": "100",
            "obsid": 102,
            "obs_id": "tess-s0002-tic-100",
            "sequence_number": 2,
            "provenance_name": "QLP",
            "t_exptime": 600.0,
            "dataReleaseDate": "2020-02-01",
        },
    ]


@pytest.fixture
def products() -> list[dict[str, object]]:
    return [
        {
            "obsID": 101,
            "productFilename": "spoc-s0001-100-lc.fits",
            "dataURI": "mast:spoc/one",
            "size": 4,
            "productType": "SCIENCE",
            "productSubGroupDescription": "Light curves",
        },
        {
            "obsID": 101,
            "productFilename": "spoc-s0001-100-tpf.fits",
            "dataURI": "mast:spoc/tpf",
            "size": 5,
            "productType": "SCIENCE",
            "productSubGroupDescription": "Target pixel files",
        },
        {
            "obsID": 102,
            "productFilename": "qlp-s0002-100-lc.fits",
            "dataURI": "mast:qlp/one",
            "size": 6,
            "productType": "SCIENCE",
            "productSubGroupDescription": "Light curves",
        },
    ]


def _normalized(observations: object, products: object) -> pd.DataFrame:
    return normalize_mast_product_search(
        object_id="TIC 100",
        observations=observations,
        products=products,
        searched_at_utc=datetime(2026, 1, 1, tzinfo=UTC),
        query=mast_query_parameters("tic-100"),
    )


def test_query_generation_has_stable_explicit_fields() -> None:
    assert mast_query_parameters("TIC 00100") == {
        "provenance_name": "TESS-SPOC",
        "target_name": ["100"],
    }


def test_batched_numeric_tics_map_back_to_internal_object_ids() -> None:
    query = mast_query_parameters(["tic-123456", "TIC 7"])
    assert query == {"provenance_name": "TESS-SPOC", "target_name": ["123456", "7"]}
    observations = [{"target_name": "7", "obsid": 1}, {"target_name": "123456", "obsid": 2}]
    normalized = normalize_mast_observations(
        observations,
        object_ids=["tic-123456", "tic-7"],
        searched_at_utc=datetime(2026, 1, 1, tzinfo=UTC),
        query=query,
    )
    assert normalized["object_id"].tolist() == ["tic-7", "tic-123456"]
    assert '"provenance_name":"TESS-SPOC"' in normalized.loc[0, "query_parameters_json"]


def test_normalization_retains_all_products_and_joins_metadata(
    observations: object, products: object
) -> None:
    result = _normalized(observations, products)
    assert len(result) == 3
    assert result.loc[0, "author"] == "SPOC"
    assert result.loc[0, "sector"] == 1
    assert result.loc[0, "object_id"] == "tic-100"
    assert result.loc[0, "source_product_json"]


def test_normalization_accepts_dataframe_and_handles_missing_optional_values() -> None:
    observations = pd.DataFrame([{"obsid": 1, "target_name": "100", "sequence_number": "bad"}])
    products = pd.DataFrame(
        [
            {
                "obsID": 1,
                "productFilename": "a.fits",
                "dataURI": "mast:a",
                "size": "not-a-size",
            }
        ]
    )
    result = _normalized(observations, products)
    assert pd.isna(result.loc[0, "sector"])
    assert pd.isna(result.loc[0, "product_size_bytes"])


def test_malformed_mast_table_and_naive_timestamp_are_rejected() -> None:
    with pytest.raises(MastSearchError, match="supported tabular"):
        _normalized({"not": "a table"}, [])
    with pytest.raises(MastSearchError, match="timezone-aware"):
        normalize_mast_product_search(
            object_id="tic-100",
            observations=[],
            products=[],
            searched_at_utc=datetime(2026, 1, 1),
            query=mast_query_parameters("tic-100"),
        )


def test_incomplete_product_response_is_rejected(observations: object) -> None:
    with pytest.raises(MastSearchError, match="productFilename"):
        _normalized(observations, [{"obsID": 101, "dataURI": "mast:one"}])


def test_search_uses_mocked_client_without_downloads(
    observations: object, products: object
) -> None:
    class FakeMastClient:
        def query_observations(self, **criteria: object) -> object:
            assert criteria == {"provenance_name": "TESS-SPOC", "target_name": ["100"]}
            return observations

        def get_product_list(self, input_observations: object) -> object:
            assert input_observations == observations
            return products

    result = search_tess_products(["tic-100"], client=FakeMastClient())
    assert len(result.products) == 3 and len(result.observations) == 2


def test_empty_mast_observations_are_preserved_as_a_valid_empty_result() -> None:
    class EmptyMastClient:
        def query_observations(self, **criteria: object) -> object:
            return []

        def get_product_list(self, input_observations: object) -> object:
            raise AssertionError("Products must not be requested for empty observations")

    result = search_tess_products(["tic-100"], client=EmptyMastClient())
    assert result.products.empty
    assert "data_uri" in result.products.columns


def test_policy_is_deterministic_and_enforces_preferred_author(
    settings: AcquisitionSettings, observations: object, products: object
) -> None:
    annotated = apply_product_policy(_normalized(observations, products), settings)
    assert annotated.loc[annotated["is_selected_core"], "data_uri"].tolist() == ["mast:spoc/one"]
    assert "preferred_author_unavailable" not in annotated["selection_reason"].tolist()
    assert "nonpreferred_author" in annotated["selection_reason"].tolist()
    assert "not_lightcurve_fits" in annotated["selection_reason"].tolist()


def test_preferred_author_absence_is_reported(
    settings: AcquisitionSettings, observations: object
) -> None:
    only_qlp = [
        {
            "obsID": 102,
            "productFilename": "q.fits",
            "dataURI": "mast:q",
            "size": 1,
            "productSubGroupDescription": "Light curves",
        }
    ]
    annotated = apply_product_policy(_normalized(observations, only_qlp), settings)
    assert not annotated["is_selected_core"].any()
    assert annotated.loc[0, "selection_reason"] == "preferred_author_unavailable"


def test_policy_can_exclude_nonpreferred_exposure(
    settings: AcquisitionSettings, observations: object, products: object
) -> None:
    strict = settings.model_copy(update={"preferred_exposure_seconds": 20.0})
    annotated = apply_product_policy(_normalized(observations, products), strict)
    assert not annotated["is_selected_core"].any()
    assert "nonpreferred_exposure" in annotated["selection_reason"].tolist()


def test_duplicate_resolution_and_plan_round_trip(
    tmp_path: Path, settings: AcquisitionSettings, observations: object, products: object
) -> None:
    duplicate = dict(products[0])
    duplicate["dataURI"] = "mast:spoc/newer"
    duplicate["productFilename"] = "spoc-s0001-new-lc.fits"
    normalized = _normalized(observations, [*products, duplicate])
    plan = build_acquisition_plan(
        normalized, settings=settings, requested_object_ids=["tic-100", "tic-999"]
    )
    assert len(plan.selected_products) == 1
    assert plan.qa_summary["estimated_selected_download_bytes"] == 4
    assert plan.qa_summary["tics_with_no_mast_results"] == ["tic-999"]
    directory = write_acquisition_plan(plan, root=tmp_path)
    loaded = load_acquisition_plan(directory)
    pd.testing.assert_frame_equal(loaded.selected_products, plan.selected_products)
    assert write_acquisition_plan(plan, root=tmp_path) == directory
    (directory / "selected_products.parquet").write_bytes(b"corrupt")
    with pytest.raises(AcquisitionManifestError, match="checksum"):
        load_acquisition_plan(directory)


def test_policy_rejects_incomplete_search_rows(settings: AcquisitionSettings) -> None:
    with pytest.raises(AcquisitionManifestError, match="required"):
        apply_product_policy(pd.DataFrame([{"object_id": "tic-1"}]), settings)


def test_download_subset_is_small_and_deterministic(
    settings: AcquisitionSettings, observations: object, products: object
) -> None:
    plan = build_acquisition_plan(
        _normalized(observations, products), settings=settings, requested_object_ids=["tic-100"]
    )
    subset = select_download_pilot(plan, settings)
    assert len(subset) == 1
    assert subset.loc[0, "download_selection_reason"].startswith("small_development_pilot")


class FakeFetcher:
    def __init__(self, payload: bytes, *, fail: bool = False) -> None:
        self.payload = payload
        self.fail = fail
        self.calls = 0

    def fetch(self, data_uri: str, destination: Path) -> None:
        self.calls += 1
        destination.write_bytes(self.payload)
        if self.fail:
            raise RuntimeError("interrupted")


def _approved_row() -> pd.Series:
    return pd.Series(
        {
            "object_id": "tic-100",
            "sector": 1,
            "product_filename": "pilot.fits",
            "data_uri": "mast:pilot",
            "product_size_bytes": 4,
            "download_selection_reason": "test",
        }
    )


def test_download_finalizes_atomically_and_reuses_valid_file(tmp_path: Path) -> None:
    row = _approved_row()
    fetcher = FakeFetcher(b"fits")
    first = download_product(row, fetcher=fetcher, raw_root=tmp_path)
    final = tmp_path / first.relative_raw_path
    assert first.status == "downloaded" and final.read_bytes() == b"fits"
    assert not list(final.parent.glob("*.part"))
    second = download_product(row, fetcher=fetcher, raw_root=tmp_path, known_checksum=first.sha256)
    assert second.status == "reused" and fetcher.calls == 1


def test_partial_and_corrupt_cache_are_not_trusted(tmp_path: Path) -> None:
    row = _approved_row()
    final = tmp_path / "tess/tic-100/sector-0001/pilot.fits"
    final.parent.mkdir(parents=True)
    partial = final.with_name(".pilot.fits.part")
    partial.write_bytes(b"bad")
    receipt = download_product(row, fetcher=FakeFetcher(b"fits"), raw_root=tmp_path)
    assert receipt.status == "downloaded" and final.read_bytes() == b"fits"
    final.write_bytes(b"bad!!")
    with pytest.raises(DownloadIntegrityError, match="Size mismatch"):
        download_product(row, fetcher=FakeFetcher(b"fits"), raw_root=tmp_path)


def test_checksum_mismatch_and_explicit_approval_are_enforced(tmp_path: Path) -> None:
    row = _approved_row()
    with pytest.raises(DownloadIntegrityError, match="Checksum mismatch"):
        download_product(
            row, fetcher=FakeFetcher(b"fits"), raw_root=tmp_path, known_checksum="0" * 64
        )
    without_approval = pd.DataFrame([_approved_row().drop("download_selection_reason")])
    with pytest.raises(DownloadIntegrityError, match="approved"):
        download_products(without_approval, fetcher=FakeFetcher(b"fits"), raw_root=tmp_path)


def test_downloader_rejects_missing_fields_unsafe_name_and_missing_partial(tmp_path: Path) -> None:
    row = _approved_row()
    row["product_filename"] = "nested/file.fits"
    with pytest.raises(DownloadIntegrityError, match="path components"):
        download_product(row, fetcher=FakeFetcher(b"fits"), raw_root=tmp_path)
    with pytest.raises(DownloadIntegrityError, match="missing fields"):
        download_product(
            pd.Series({"object_id": "tic-1"}), fetcher=FakeFetcher(b"fits"), raw_root=tmp_path
        )

    class NoWriteFetcher:
        def fetch(self, data_uri: str, destination: Path) -> None:
            return None

    with pytest.raises(DownloadIntegrityError, match="did not create"):
        download_product(_approved_row(), fetcher=NoWriteFetcher(), raw_root=tmp_path)


def test_manifest_immutability_rejects_conflicting_content(
    tmp_path: Path, settings: AcquisitionSettings, observations: object, products: object
) -> None:
    plan = build_acquisition_plan(
        _normalized(observations, products), settings=settings, requested_object_ids=["tic-100"]
    )
    directory = write_acquisition_plan(plan, root=tmp_path)
    (directory / "metadata.json").write_text('{"plan_id":"different"}', encoding="utf-8")
    with pytest.raises(AcquisitionManifestError):
        write_acquisition_plan(plan, root=tmp_path)


def test_config_loads_acquisition_policy() -> None:
    config = load_config("configs/acquisition/pilot_spoc.yaml")
    assert config.acquisition is not None
    assert config.acquisition.preferred_author == "TESS-SPOC"
