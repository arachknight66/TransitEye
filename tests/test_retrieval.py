from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from transiteye.acquisition.retrieval import freeze_products, retrieve_pending


def _observations() -> pd.DataFrame:
    return pd.DataFrame(
        [{"mast_obs_id": "1", "object_id": "tic-1"}, {"mast_obs_id": "2", "object_id": "tic-2"}]
    )


def test_retrieval_checkpoint_resume_and_freeze(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def query_criteria(*, obsid: str) -> list[dict[str, str]]:
        calls.append(obsid)
        return [{"obsid": obsid}]

    monkeypatch.setattr("astroquery.mast.Observations.query_criteria", query_criteria)
    monkeypatch.setattr(
        "transiteye.acquisition.retrieval.AstroqueryMastClient.get_product_list",
        lambda _self, raw: (
            []
            if raw[0]["obsid"] == "2"
            else [{"obsID": "1", "productFilename": "x_lc.fits", "dataURI": "mast:x"}]
        ),
    )
    monkeypatch.setattr(
        "transiteye.acquisition.retrieval.normalize_mast_products",
        lambda products, observations: pd.DataFrame(
            [{"mast_obs_id": observations.iloc[0].mast_obs_id, "data_uri": "mast:x"}]
            if products
            else []
        ),
    )
    status = retrieve_pending(_observations(), tmp_path)
    assert status.status.tolist() == ["success", "zero_products"]
    assert retrieve_pending(_observations(), tmp_path).status.tolist() == status.status.tolist()
    assert calls == ["1", "2"]
    frozen = freeze_products(tmp_path)
    metadata = json.loads((frozen / "metadata.json").read_text())
    assert metadata["snapshot_id"].startswith("mast-products-")
    assert metadata["coverage_counts"] == {"success": 1, "zero_products": 1}


def test_freeze_rejects_pending(tmp_path: Path) -> None:
    pd.DataFrame([{"mast_obs_id": "1", "status": "pending"}]).to_parquet(
        tmp_path / "coverage.parquet"
    )
    with pytest.raises(ValueError, match="pending"):
        freeze_products(tmp_path)
