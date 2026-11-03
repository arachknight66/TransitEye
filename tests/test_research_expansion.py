"""Guard the frozen test boundary of the research data expansion."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture(params=["expand_research_cohort", "expand_research_cohort_large", "expand_variability_cohort"])
def study(request, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    spec = importlib.util.spec_from_file_location(
        request.param, Path(__file__).resolve().parents[1] / f"scripts/{request.param}.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _manifest(output: Path, monkeypatch: pytest.MonkeyPatch, study) -> None:
    monkeypatch.setattr(study, "code_fingerprint", lambda: "frozen-code")
    content = json.dumps({"backend_sha256": "frozen-code", "input_sha256": {}, "records": []})
    (output / "manifest.json").write_text(content)
    checksum = hashlib.sha256(content.encode()).hexdigest()
    (output / "manifest.sha256.json").write_text(json.dumps({"sha256": checksum}))


def test_test_data_cannot_be_opened_without_locked_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, study) -> None:
    _manifest(tmp_path, monkeypatch, study)
    with pytest.raises(ValueError, match="(?i)lock(?: the)? model"):
        study.run(tmp_path, "test", 1)
    assert not (tmp_path / "fits").exists()


def test_test_data_cannot_be_opened_with_changed_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, study) -> None:
    _manifest(tmp_path, monkeypatch, study)
    (tmp_path / "research-estimator.joblib").write_bytes(b"changed")
    (tmp_path / "model-lock.json").write_text(json.dumps({"model_sha256": "different", "manifest_sha256": "different"}))
    with pytest.raises(ValueError, match="(?i)(differs from lock|locked model or manifest changed)"):
        study.run(tmp_path, "test", 1)
    assert not (tmp_path / "fits").exists()
