"""The exploratory audit must never include test-group labels in its counts."""

from pathlib import Path

import pytest

from transiteye.evaluation.label_audit import run, summarize_training


def test_audit_counts_only_training_groups_and_keeps_proxy_labels_separate() -> None:
    labels = [
        {"source_group_id": "tic-1", "label": "eclipse_like", "label_source": "exoplanet_archive",
         "review_status": "verified", "evidence_identifier": "TOI-1", "notes": "deep false positive"},
        {"source_group_id": "tic-2", "label": "stellar_variability", "label_source": "variable_star_catalog",
         "review_status": "verified", "evidence_identifier": "TIC-2-ATL", "notes": ""},
        {"source_group_id": "tic-3", "label": "other_artifact", "label_source": "exoplanet_archive",
         "review_status": "reviewed", "evidence_identifier": "TOI-3", "notes": "low-amplitude"},
    ]
    assignments = [{"source_group_id": "tic-1", "split": "train"},
                   {"source_group_id": "tic-2", "split": "train"},
                   {"source_group_id": "tic-3", "split": "test"}]
    result = summarize_training(labels, assignments)
    assert result["training_labels"] == 2
    assert result["deep_false_positive_eclipse_proxies"] == 1
    assert result["atl_variability_proxies"] == 1
    assert result["other_artifact_low_amplitude_notes"] == 0
    with pytest.raises(ValueError, match="duplicate source group"):
        summarize_training(labels, assignments + [assignments[0]])


def test_audit_rejects_other_paths_before_reading_them(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="historical v1 manifest paths"):
        run(tmp_path / "not-opened.json", tmp_path / "not-opened-splits.json", tmp_path / "output.json")
    assert not (tmp_path / "output.json").exists()
