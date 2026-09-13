"""Cached, read-only access to authoritative TransitEye artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from transiteye.reproducibility.workflow import demo_smoke_mode, verify_mode


class ArtifactUnavailableError(RuntimeError):
    """Raised when a required frozen artifact is not available."""


def repository_root() -> Path:
    """Return the repository root when the app is launched from it."""
    return Path.cwd()


def _required(path: Path) -> Path:
    if not path.is_file():
        raise ArtifactUnavailableError(f"Required artifact is missing: {path.as_posix()}")
    return path


@st.cache_data(show_spinner=False)
def load_json(relative_path: str) -> dict[str, Any]:
    path = _required(repository_root() / relative_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ArtifactUnavailableError(f"Expected JSON object: {relative_path}")
    return value


@st.cache_data(show_spinner=False)
def project_state() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load the three authoritative UI entry points."""
    return (
        load_json("results/index.json"),
        load_json("results/project_status.json"),
        load_json("results/release_manifest.json"),
    )


def _artifact_path(category: str, version: str, filename: str) -> str:
    return f"data/{category}/{version}/{filename}"


@st.cache_data(show_spinner=False)
def load_candidates(dataset: str) -> pd.DataFrame:
    """Load a frozen candidate table selected through the result index."""
    index, _, _ = project_state()
    key = "demo_dataset" if dataset == "demo" else "scientific_dataset"
    version = str(index[key])
    table = pd.read_parquet(
        _required(repository_root() / _artifact_path("datasets", version, "candidates.parquet"))
    )
    table = table.copy()
    table["dataset_role"] = dataset if "dataset_role" not in table else table["dataset_role"]
    return table


@st.cache_data(show_spinner=False)
def load_scorecards() -> pd.DataFrame:
    index, _, _ = project_state()
    path = _artifact_path(
        "evaluation", str(index["final_evaluation"]), "candidate_scorecards.parquet"
    )
    return pd.read_parquet(_required(repository_root() / path))


def candidates_with_scorecards(dataset: str) -> pd.DataFrame:
    """Join frozen candidate identity fields to frozen scorecards for display only."""
    candidates = load_candidates(dataset)
    scorecards = load_scorecards()
    artifact_role = "scientific_unlabeled" if dataset == "scientific" else dataset
    scorecards = scorecards.loc[scorecards["dataset_role"] == artifact_role].copy()
    scorecards["dataset_role"] = dataset
    duplicate_columns = set(scorecards.columns).intersection(candidates.columns) - {"candidate_id"}
    right = scorecards.drop(columns=sorted(duplicate_columns))
    return candidates.merge(right, on="candidate_id", how="left", validate="one_to_one")


@st.cache_data(show_spinner=False)
def load_local_explanations() -> pd.DataFrame:
    index, _, _ = project_state()
    path = _artifact_path(
        "evaluation", str(index["final_evaluation"]), "local_explanations.parquet"
    )
    return pd.read_parquet(_required(repository_root() / path))


@st.cache_data(show_spinner=False)
def load_evaluation_table(filename: str) -> pd.DataFrame:
    index, _, _ = project_state()
    path = _artifact_path("evaluation", str(index["robustness_evaluation"]), filename)
    return pd.read_parquet(_required(repository_root() / path))


@st.cache_data(show_spinner=False)
def load_result_table(filename: str) -> pd.DataFrame:
    return pd.read_csv(_required(repository_root() / "results" / "tables" / filename))


def figure_path(stem: str) -> Path:
    """Resolve an indexed PNG figure by name without inventing a location."""
    index, _, _ = project_state()
    expected = f"{stem}.png"
    for item in index.get("figures", []):
        if isinstance(item, dict) and str(item.get("path", "")).endswith(expected):
            return _required(repository_root() / str(item["path"]))
    raise ArtifactUnavailableError(f"Figure is not registered by results/index.json: {expected}")


def candidate_scorecard(dataset: str, candidate_id: str) -> pd.Series:
    scorecards = load_scorecards()
    artifact_role = "scientific_unlabeled" if dataset == "scientific" else dataset
    subset = scorecards.loc[
        (scorecards["dataset_role"] == artifact_role) & (scorecards["candidate_id"] == candidate_id)
    ]
    if subset.empty:
        raise ArtifactUnavailableError(f"No frozen scorecard exists for candidate: {candidate_id}")
    return subset.iloc[0]


def filter_candidates(
    frame: pd.DataFrame,
    *,
    tics: list[str] | None = None,
    score_range: tuple[float, float] | None = None,
    ranks: tuple[int, int] | None = None,
    period_range: tuple[float, float] | None = None,
    triage: list[str] | None = None,
    support: str | None = None,
) -> pd.DataFrame:
    """Apply read-only explorer filters while preserving candidate rows."""
    filtered = frame.copy()
    if tics:
        filtered = filtered.loc[filtered["object_id"].isin(tics)]
    if score_range and "score" in filtered:
        filtered = filtered.loc[filtered["score"].between(*score_range)]
    rank_column = "bls_candidate_rank" if "bls_candidate_rank" in filtered else "candidate_rank"
    if ranks and rank_column in filtered:
        filtered = filtered.loc[filtered[rank_column].between(*ranks)]
    period_column = (
        "bls_candidate_period" if "bls_candidate_period" in filtered else "candidate_period"
    )
    if period_range and period_column in filtered:
        filtered = filtered.loc[filtered[period_column].between(*period_range)]
    if triage and "triage_status" in filtered:
        filtered = filtered.loc[filtered["triage_status"].isin(triage)]
    if support == "in support" and "outside_robust_support_count" in filtered:
        filtered = filtered.loc[filtered["outside_robust_support_count"] == 0]
    if support == "out of support" and "outside_robust_support_count" in filtered:
        filtered = filtered.loc[filtered["outside_robust_support_count"] > 0]
    return (
        filtered.sort_values("score", ascending=False, kind="stable")
        if "score" in filtered
        else filtered
    )


def run_verification() -> dict[str, Any]:
    """Invoke the existing frozen-artifact verifier; no UI-specific verifier exists."""
    return verify_mode(repository_root())


def run_demo_smoke() -> dict[str, Any]:
    """Invoke the existing compact offline smoke path."""
    return demo_smoke_mode(repository_root())


def lazy_candidate_artifacts(
    scorecard: pd.Series,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Load processed cadences and a BLS periodogram only for a selected candidate."""
    raw_checksum = str(scorecard.get("source_raw_checksum", ""))
    bls_hash = str(scorecard.get("bls_config_hash", ""))
    if len(raw_checksum) < 20 or not bls_hash:
        return None, None
    token = raw_checksum[:20]
    root = repository_root()
    processed = root / "data" / "processed" / "mvp" / token / "cadences.parquet"
    periodogram = root / "data" / "candidates" / bls_hash / f"{token}-periodogram.parquet"
    cadence_frame = pd.read_parquet(processed) if processed.is_file() else None
    periodogram_frame = pd.read_parquet(periodogram) if periodogram.is_file() else None
    return cadence_frame, periodogram_frame
