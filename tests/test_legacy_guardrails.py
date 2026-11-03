"""Historical data builders and the opened evaluation must fail before accessing data."""

from pathlib import Path
from runpy import run_path

import pytest


@pytest.mark.parametrize("script", ["build_cohort_v3.py", "build_expanded_cohort.py"])
def test_historical_synthetic_builders_are_disabled(script: str) -> None:
    builder = run_path(str(Path(__file__).resolve().parents[1] / "scripts" / script))["main"]
    with pytest.raises(RuntimeError, match="Historical synthetic cohort builder is disabled"):
        builder()


def test_opened_historical_evaluation_is_disabled_before_reading_inputs(tmp_path: Path) -> None:
    evaluate = run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "evaluate_real_cohort.py"))["evaluate"]
    with pytest.raises(RuntimeError, match="test was already opened"):
        evaluate(tmp_path / "missing-config.toml", tmp_path / "output")
    assert not (tmp_path / "output").exists()
