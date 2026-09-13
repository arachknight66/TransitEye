from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(".")
README = ROOT / "README.md"
REPORT = ROOT / "reports/TransitEye_Final_Report.md"
FINAL_DOCS = [
    README,
    ROOT / "docs/architecture.md",
    ROOT / "docs/scientific_method.md",
    ROOT / "docs/demo_mode.md",
    ROOT / "docs/evaluation.md",
    ROOT / "docs/usage.md",
    ROOT / "docs/reproducibility.md",
    ROOT / "docs/limitations.md",
    ROOT / "docs/artifact_lineage.md",
    REPORT,
    ROOT / "presentation/DEMO_SCRIPT.md",
    ROOT / "presentation/DEMO_CHECKLIST.md",
    ROOT / "presentation/DEMO_RESULTS.md",
    ROOT / "CHANGELOG.md",
    ROOT / "RELEASE_NOTES.md",
]
STATUS_KEYS = (
    "software_pipeline_status",
    "demo_pipeline_status",
    "scientific_readiness",
    "official_model",
    "official_threshold",
    "final_evaluation",
    "repro_version",
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _json(path: Path) -> dict[str, object]:
    return json.loads(_read(path))


def _local_links(path: Path) -> list[Path]:
    links = re.findall(r"!?(?:\[[^]]*\])\(([^)]+)\)", _read(path))
    return [path.parent / link.split("#", 1)[0] for link in links if "://" not in link]


def test_readme_is_current_portable_and_has_valid_commands_and_figure() -> None:
    status = _json(ROOT / "results/project_status.json")
    text = _read(README)
    assert "Software pipeline: COMPLETE" in text
    assert "Demo pipeline: COMPLETE" in text
    assert "Scientific readiness: BLOCKED" in text
    for key in STATUS_KEYS[3:]:
        assert str(status[key]) in text
    for command in (
        "uv sync --all-groups --locked",
        "uv run python scripts/reproduce_project.py --verify",
        "uv run python scripts/reproduce_project.py --report",
        "uv run python scripts/reproduce_project.py --demo-smoke",
        "uv run python scripts/reproduce_project.py --full-replay",
    ):
        assert command in text
    assert (ROOT / "results/figures/figure_1_system_architecture.png") in _local_links(README)
    assert all(link.exists() for link in _local_links(README))
    assert "/home/" not in text


def test_final_report_has_required_sections_assets_metrics_and_status() -> None:
    text = _read(REPORT)
    headings = re.findall(r"^## (\d+)\. ", text, flags=re.MULTILINE)
    assert headings == [str(number) for number in range(1, 29)]
    for title in (
        "Abstract",
        "System Architecture",
        "Label Policy",
        "Demo Mode",
        "Feature Engineering",
        "Evaluation Methodology",
        "Scientific Readiness",
        "Reproducibility",
        "References",
    ):
        assert re.search(rf"^## \d+\. {re.escape(title)}$", text, flags=re.MULTILINE)
    links = _local_links(REPORT)
    assert sum(path.suffix == ".png" for path in links) >= 8
    assert sum(path.suffix == ".csv" for path in links) == 7
    assert all(path.exists() for path in links)

    robustness = pd.read_csv(ROOT / "results/tables/table_5_grouped_robustness.csv")
    loto_f1 = robustness.loc[
        (robustness["summary"] == "pooled_loto") & (robustness["metric"] == "f1"),
        "value",
    ].iloc[0]
    assert f"F1 {loto_f1:.3f}" in text
    status = _json(ROOT / "results/project_status.json")
    for key in STATUS_KEYS:
        assert str(status[key]) in text
    assert "two held-out TICs" in text
    assert "scientific_readiness = blocked" in text


def test_demo_package_is_offline_explicit_and_uses_authoritative_ids() -> None:
    script = ROOT / "presentation/DEMO_SCRIPT.md"
    checklist = ROOT / "presentation/DEMO_CHECKLIST.md"
    results = ROOT / "presentation/DEMO_RESULTS.md"
    assert all(path.exists() for path in (script, checklist, results))
    combined = "\n".join(_read(path) for path in (script, checklist, results))
    assert "uv run python scripts/reproduce_project.py --demo-smoke" in combined
    assert "offline" in combined.lower()
    assert "scientific" in combined.lower() and "demo" in combined.lower()
    index = _json(ROOT / "results/index.json")
    assert str(index["official_model"]) in combined
    assert str(index["repro_version"]) in combined
    assert "unlabeled" in combined.lower()


def test_final_documentation_is_portable_and_claim_bounded() -> None:
    prohibited_affirmative = (
        "75 detected planets",
        "75 exoplanets discovered",
        "real TESS accuracy =",
        "scientific accuracy =",
        "confirmed planets",
        "validated scientific predictions",
    )
    for path in FINAL_DOCS:
        assert path.exists(), path
        text = _read(path)
        assert "/home/" not in text, path
        lowered = text.lower()
        assert not any(pattern.lower() in lowered for pattern in prohibited_affirmative), path


def test_documentation_ids_match_project_status_and_index() -> None:
    status = _json(ROOT / "results/project_status.json")
    index = _json(ROOT / "results/index.json")
    assert status["official_model"] == index["official_model"]
    assert status["final_evaluation"] == index["final_evaluation"]
    assert status["repro_version"] == index["repro_version"]
    for path in (README, REPORT, ROOT / "docs/reproducibility.md"):
        text = _read(path)
        assert str(status["official_model"]) in text
        assert str(status["official_threshold"]) in text
        assert str(status["final_evaluation"]) in text
        assert str(status["repro_version"]) in text
