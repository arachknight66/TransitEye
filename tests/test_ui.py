from __future__ import annotations

import socket
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from transiteye.ui.data import (
    ArtifactUnavailableError,
    candidate_scorecard,
    candidates_with_scorecards,
    figure_path,
    filter_candidates,
    lazy_candidate_artifacts,
    load_json,
    project_state,
    run_demo_smoke,
    run_verification,
)
from transiteye.ui.formatting import humanize, number

ROOT = Path(".").resolve()


def test_ui_loads_authoritative_state_and_frozen_identities() -> None:
    index, status, release = project_state()
    assert index["official_model"] == "model-4a58f313bc50b9c3dc41"
    assert status["official_threshold"] == 0.325
    assert release["release_version_id"] == "release-66d8552756eeb14f83aa"


def test_ui_candidate_tables_and_filtering_are_read_only() -> None:
    scientific = candidates_with_scorecards("scientific")
    demo = candidates_with_scorecards("demo")
    assert len(scientific) == 75
    assert len(demo) == 375
    assert set(scientific["dataset_role"]) == {"scientific"}
    assert set(scientific["gold_candidate_label"]) == {"unlabeled"}
    filtered = filter_candidates(scientific, tics=[scientific.iloc[0]["object_id"]], ranks=(1, 5))
    assert not filtered.empty
    assert filtered["object_id"].nunique() == 1
    assert (filtered["bls_candidate_rank"] <= 5).all()


def test_ui_scorecard_and_figures_resolve_without_mutation() -> None:
    scientific = candidates_with_scorecards("scientific")
    scorecard = candidate_scorecard("scientific", str(scientific.iloc[0]["candidate_id"]))
    assert scorecard["model_version"] == "model-4a58f313bc50b9c3dc41"
    assert scorecard["frozen_threshold"] == 0.325
    assert figure_path("figure_1_system_architecture").is_file()
    cadences, periodogram = lazy_candidate_artifacts(scorecard)
    assert cadences is None or isinstance(cadences, pd.DataFrame)
    assert periodogram is None or isinstance(periodogram, pd.DataFrame)


def test_ui_reports_missing_required_artifacts_safely(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    project_state.clear()
    load_json.clear()
    with pytest.raises(ArtifactUnavailableError, match="Required artifact"):
        project_state()
    project_state.clear()
    load_json.clear()


def test_ui_safe_actions_reuse_existing_offline_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked_socket(*args: object, **kwargs: object) -> None:
        raise AssertionError("Network access attempted")

    monkeypatch.setattr(socket, "socket", blocked_socket)
    assert run_verification()["status"] == "verified"
    assert run_demo_smoke()["status"] == "passed"


@pytest.mark.parametrize(
    "page_path",
    [
        "app.py",
        "pages/1_Overview.py",
        "pages/2_Demo_Mode.py",
        "pages/3_Scientific_Candidates.py",
        "pages/4_Candidate_Explorer.py",
        "pages/5_Model_Evaluation.py",
        "pages/6_Architecture.py",
        "pages/7_Reproducibility.py",
    ],
)
def test_streamlit_pages_render(page_path: str) -> None:
    app = AppTest.from_file(ROOT / page_path)
    app.run(timeout=30)
    assert not app.exception
    assert app.title[0].value == "TransitEye"


def test_ui_claims_keep_scientific_candidates_unlabeled() -> None:
    ui_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [
            ROOT / "app.py",
            *(ROOT / "pages").glob("*.py"),
            ROOT / "src/transiteye/ui/components.py",
        ]
    ).lower()
    assert "unlabeled" in ui_text
    assert "demo-trained" in ui_text
    assert "not confirmed exoplanets" in ui_text
    assert "not evidence of planet status" in ui_text


def test_ui_formatting_is_readable_for_feature_values() -> None:
    assert humanize("ls_dominant_power") == "LS Dominant Power"
    assert number(None) == "—"
    assert number("not-a-number") == "not-a-number"
