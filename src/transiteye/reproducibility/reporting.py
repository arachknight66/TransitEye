"""B059 deterministic final tables and figures from frozen artifacts."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, cast

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "transiteye-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from transiteye.acquisition.checksums import sha256_file
from transiteye.features.registry import FEATURE_REGISTRY
from transiteye.features.schemas import ColumnRole
from transiteye.serialization import canonical_json

from .manifest import (
    DEMO_DATASET,
    DEMO_FEATURES,
    DEMO_MANIFEST,
    EXPANSION_ID,
    FINAL_EVALUATION,
    MODEL,
    ROBUSTNESS,
    SCIENTIFIC_DATASET,
    SCIENTIFIC_FEATURES,
)

TABLE_NAMES = (
    "table_1_data_pipeline_summary.csv",
    "table_2_feature_summary.csv",
    "table_3_model_comparison.csv",
    "table_4_feature_ablation.csv",
    "table_5_grouped_robustness.csv",
    "table_6_injection_recovery.csv",
    "table_7_scientific_readiness.csv",
)
FIGURE_STEMS = (
    "figure_1_system_architecture",
    "figure_2_pipeline_flow",
    "figure_3_model_comparison",
    "figure_4_feature_ablation",
    "figure_5_grouped_robustness",
    "figure_6_injection_recovery",
    "figure_7_score_distributions",
    "figure_8_domain_shift",
    "figure_9_feature_importance",
    "figure_10_scientific_readiness",
)


def _json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _write_table(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.12g")


def build_final_tables(repository: Path, table_directory: Path) -> dict[str, list[str]]:
    """Derive seven report tables without recomputing model outputs."""
    table_directory.mkdir(parents=True, exist_ok=True)
    demo_candidates = pd.read_parquet(
        repository / "data/datasets" / DEMO_DATASET / "candidates.parquet"
    )
    science_candidates = pd.read_parquet(
        repository / "data/datasets" / SCIENTIFIC_DATASET / "candidates.parquet"
    )
    demo_labels = pd.read_parquet(
        repository / "data/features" / DEMO_DATASET / DEMO_FEATURES / "labels.parquet"
    )
    demo_validation = _json(
        repository / "data/features" / DEMO_DATASET / DEMO_FEATURES / "feature_validation.json"
    )
    science_validation = _json(
        repository
        / "data/features"
        / SCIENTIFIC_DATASET
        / SCIENTIFIC_FEATURES
        / "feature_validation.json"
    )
    expansion = repository / "data/manifests/expansion-be59541b9388b685e1f9"
    downloads = pd.read_parquet(expansion / "download_receipts.parquet")
    preprocessing = pd.read_parquet(expansion / "preprocessing_receipts.parquet")
    table1 = pd.DataFrame(
        [
            (
                "scientific_tic_count",
                science_candidates["object_id"].nunique(),
                "unlabeled scientific path",
            ),
            (
                "demo_source_tic_count",
                demo_candidates["object_id"].nunique(),
                "controlled demo path",
            ),
            (
                "raw_products_used",
                (downloads["status"] == "downloaded").sum(),
                "frozen expansion receipts",
            ),
            (
                "processed_light_curves",
                (preprocessing["status"] == "processed").sum(),
                "scientific source curves",
            ),
            ("demo_candidate_count", len(demo_candidates), "controlled demo candidates"),
            (
                "demo_positive_candidates",
                (demo_labels["gold_candidate_label"] == "positive").sum(),
                "synthetic truth",
            ),
            (
                "demo_negative_candidates",
                (demo_labels["gold_candidate_label"] == "negative").sum(),
                "synthetic truth",
            ),
            (
                "demo_unlabeled_candidates",
                (demo_labels["gold_candidate_label"] == "unlabeled").sum(),
                "not used as gold",
            ),
            ("scientific_unlabeled_candidates", len(science_candidates), "no gold labels"),
        ],
        columns=["metric", "value", "context"],
    )
    _write_table(table1, table_directory / TABLE_NAMES[0])

    group_counts: dict[str, int] = {}
    for definition in FEATURE_REGISTRY:
        if definition.role is ColumnRole.FEATURE:
            group_counts[definition.group] = group_counts.get(definition.group, 0) + 1
    table2 = pd.DataFrame(
        [
            (
                "time_domain_feature_count",
                group_counts["time_domain"],
                group_counts["time_domain"],
                "feature_registry",
            ),
            ("bls_feature_count", group_counts["bls"], group_counts["bls"], "feature_registry"),
            (
                "lomb_scargle_feature_count",
                group_counts["lomb_scargle"],
                group_counts["lomb_scargle"],
                "feature_registry",
            ),
            ("fft_feature_count", group_counts["fft"], group_counts["fft"], "feature_registry"),
            (
                "total_model_features",
                sum(group_counts.values()),
                sum(group_counts.values()),
                "frozen model schema",
            ),
            (
                "all_null_feature_count",
                demo_validation["all_null_feature_count"],
                science_validation["all_null_feature_count"],
                "feature QA",
            ),
            (
                "infinite_value_count",
                demo_validation["infinite_value_count"],
                science_validation["infinite_value_count"],
                "feature QA",
            ),
            (
                "missing_value_count",
                sum(demo_validation["missing_count"].values()),
                sum(science_validation["missing_count"].values()),
                "feature QA across all cells",
            ),
        ],
        columns=["metric", "demo_value", "scientific_value", "source"],
    )
    _write_table(table2, table_directory / TABLE_NAMES[1])

    metrics = _json(repository / "data/models" / MODEL / "metrics.json")
    model_labels = {
        "logistic_regression": "Logistic Regression",
        "random_forest": "Random Forest",
        "svm_rbf": "SVM",
        "elm": "ELM",
    }
    model_rows = []
    for family, label in model_labels.items():
        values = metrics["validation"][f"time_bls_frequency:{family}"]
        model_rows.append(
            {
                "model": label,
                **{
                    name: values[name]
                    for name in (
                        "pr_auc",
                        "roc_auc",
                        "balanced_accuracy",
                        "precision",
                        "recall",
                        "f1",
                    )
                },
                "evaluation_role": "development-demo validation",
            }
        )
    _write_table(pd.DataFrame(model_rows), table_directory / TABLE_NAMES[2])

    ablation_rows = []
    for key, label in (
        ("time_bls", "Time + BLS"),
        ("time_bls_frequency", "Time + BLS + Lomb-Scargle + FFT"),
    ):
        values = metrics["ablation_winners"][key]
        for split_name in ("validation", "test"):
            ablation_rows.append(
                {
                    "feature_set": label,
                    "split": split_name,
                    "model": values["model_family"],
                    "threshold": values["threshold"],
                    **{
                        name: values[split_name][name]
                        for name in (
                            "pr_auc",
                            "roc_auc",
                            "balanced_accuracy",
                            "precision",
                            "recall",
                            "f1",
                        )
                    },
                    "context": "test contains 2 held-out TICs; development-demo only"
                    if split_name == "test"
                    else "development-demo validation",
                }
            )
    _write_table(pd.DataFrame(ablation_rows), table_directory / TABLE_NAMES[3])

    group = _json(repository / "data/evaluation" / ROBUSTNESS / "group_metrics.json")
    uncertainty = _json(repository / "data/evaluation" / ROBUSTNESS / "uncertainty.json")
    per_tic = group["per_tic"]
    pooled = group["pooled"]
    table5 = pd.DataFrame(
        [
            {
                "summary": "pooled_loto",
                "metric": metric,
                "value": pooled[metric],
                "lower": uncertainty["metrics"].get(metric, {}).get("lower"),
                "upper": uncertainty["metrics"].get(metric, {}).get("upper"),
                "independent_tics": len(group["development_tics"]),
            }
            for metric in ("pr_auc", "roc_auc", "balanced_accuracy", "precision", "recall", "f1")
        ]
        + [
            {
                "summary": "per_tic_range",
                "metric": metric,
                "value": None,
                "lower": min(row[metric] for row in per_tic),
                "upper": max(row[metric] for row in per_tic),
                "independent_tics": len(group["development_tics"]),
            }
            for metric in ("pr_auc", "f1")
        ]
    )
    _write_table(table5, table_directory / TABLE_NAMES[4])

    benchmark = _json(repository / "data/evaluation" / FINAL_EVALUATION / "benchmark_summary.json")
    injection_rows = []
    for event_class, values in benchmark["end_to_end_injection_performance"].items():
        injection_rows.append(
            {
                "synthetic_event_class": event_class,
                **values,
                "scope": "controlled synthetic injection; not real TESS accuracy",
            }
        )
    _write_table(
        pd.DataFrame(injection_rows).sort_values("synthetic_event_class"),
        table_directory / TABLE_NAMES[5],
    )

    readiness = _json(
        repository / "data/evaluation" / FINAL_EVALUATION / "scientific_readiness.json"
    )
    table7 = pd.DataFrame(readiness["criteria"])
    table7["status"] = np.where(table7["passed"], "PASS", "FAIL")
    _write_table(table7[["criterion", "status", "evidence"]], table_directory / TABLE_NAMES[6])
    return {
        TABLE_NAMES[0]: [SCIENTIFIC_DATASET, DEMO_DATASET, "expansion-be59541b9388b685e1f9"],
        TABLE_NAMES[1]: [DEMO_FEATURES, SCIENTIFIC_FEATURES, MODEL],
        TABLE_NAMES[2]: [MODEL],
        TABLE_NAMES[3]: [MODEL],
        TABLE_NAMES[4]: [ROBUSTNESS],
        TABLE_NAMES[5]: [ROBUSTNESS, FINAL_EVALUATION],
        TABLE_NAMES[6]: [FINAL_EVALUATION],
    }


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 160,
            "font.size": 9,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save(fig: Figure, directory: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(directory / f"{stem}.png", metadata={"Software": "TransitEye"})
    fig.savefig(
        directory / f"{stem}.pdf",
        metadata={"Creator": "TransitEye", "CreationDate": None, "ModDate": None},
    )
    plt.close(fig)


def _box(ax: Axes, x: float, y: float, text: str, color: str, width: float = 0.18) -> None:
    patch = FancyBboxPatch(
        (x - width / 2, y - 0.06),
        width,
        0.12,
        boxstyle="round,pad=0.012",
        facecolor=color,
        edgecolor="#374151",
        linewidth=1,
    )
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=8)


def _arrow(
    ax: Axes, start: tuple[float, float], end: tuple[float, float], color: str = "#4b5563"
) -> None:
    ax.annotate(
        "", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": color, "lw": 1.4}
    )


def build_final_figures(repository: Path, figure_directory: Path) -> dict[str, list[str]]:
    """Generate ten deterministic, claim-bounded figures."""
    figure_directory.mkdir(parents=True, exist_ok=True)
    _style()
    sources: dict[str, list[str]] = {}

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_title("TransitEye system architecture and evidence boundary")
    for box_x, text in (
        (0.1, "Real TESS\nproducts"),
        (0.3, "Preprocess +\nblind BLS"),
        (0.5, "Scientific\ndataset"),
    ):
        _box(ax, box_x, 0.72, text, "#dbeafe")
    for box_x, text in (
        (0.1, "Real TESS\nsubstrates"),
        (0.3, "Controlled\ninjection"),
        (0.5, "Demo\ndataset"),
    ):
        _box(ax, box_x, 0.28, text, "#dcfce7")
    for y in (0.72, 0.28):
        _arrow(ax, (0.19, y), (0.21, y))
        _arrow(ax, (0.39, y), (0.41, y))
        _arrow(ax, (0.59, y), (0.67, 0.5))
    for box_x, text in (
        (0.75, "Shared features\n+ frozen model"),
        (0.92, "Scores +\nevaluation"),
    ):
        _box(ax, box_x, 0.5, text, "#f3f4f6", width=0.15)
    _arrow(ax, (0.825, 0.5), (0.845, 0.5))
    ax.text(
        0.5,
        0.88,
        "Unlabeled scientific path: readiness blocked",
        ha="center",
        color="#b91c1c",
        weight="bold",
    )
    ax.text(
        0.5,
        0.1,
        "Controlled demo path: validation complete",
        ha="center",
        color="#166534",
        weight="bold",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    _save(fig, figure_directory, FIGURE_STEMS[0])
    sources[FIGURE_STEMS[0]] = [
        SCIENTIFIC_DATASET,
        DEMO_MANIFEST,
        DEMO_DATASET,
        MODEL,
        FINAL_EVALUATION,
    ]

    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.set_title("TransitEye processing flow with truth-data separation")
    nodes = [
        "Raw TESS",
        "Preprocessing",
        "Blind BLS",
        "Candidates",
        "Features",
        "Frozen model",
        "Scores / evaluation",
    ]
    xs = np.linspace(0.07, 0.93, len(nodes))
    for box_x, text in zip(xs, nodes, strict=True):
        _box(ax, float(box_x), 0.62, text, "#e0f2fe", width=0.12)
    for left, right in zip(xs[:-1], xs[1:], strict=True):
        _arrow(ax, (float(left + 0.06), 0.62), (float(right - 0.06), 0.62))
    _box(ax, 0.72, 0.2, "Catalog / synthetic truth\nEvaluation only", "#fef3c7", width=0.24)
    _arrow(ax, (0.78, 0.28), (0.91, 0.53), color="#b45309")
    ax.text(
        0.72,
        0.06,
        "Truth never enters feature generation or model inputs",
        ha="center",
        color="#92400e",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    _save(fig, figure_directory, FIGURE_STEMS[1])
    sources[FIGURE_STEMS[1]] = [
        EXPANSION_ID,
        SCIENTIFIC_DATASET,
        DEMO_DATASET,
        DEMO_FEATURES,
        MODEL,
    ]

    metrics = _json(repository / "data/models" / MODEL / "metrics.json")
    model_order = ["logistic_regression", "random_forest", "svm_rbf", "elm"]
    labels = ["Logistic\nRegression", "Random\nForest", "SVM", "ELM"]
    values = [metrics["validation"][f"time_bls_frequency:{name}"]["pr_auc"] for name in model_order]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(labels, values, color="#3b82f6")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Validation PR-AUC")
    ax.set_title("Frozen development-demo model comparison")
    _save(fig, figure_directory, FIGURE_STEMS[2])
    sources[FIGURE_STEMS[2]] = [MODEL]

    ablations = metrics["ablation_winners"]
    labels = ["Time + BLS", "Time + BLS +\nfrequency"]
    plot_x = np.arange(2)
    width = 0.22
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for offset, metric, color in ((-width, "pr_auc", "#2563eb"), (0, "f1", "#059669")):
        ax.bar(
            plot_x + offset,
            [ablations[k]["validation"][metric] for k in ("time_bls", "time_bls_frequency")],
            width,
            label=f"Validation {metric.upper()}",
            color=color,
        )
    ax.bar(
        plot_x + width,
        [ablations[k]["test"]["f1"] for k in ("time_bls", "time_bls_frequency")],
        width,
        label="Locked-test F1",
        color="#f59e0b",
    )
    ax.set_xticks(plot_x, labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Metric value")
    ax.set_title("Frozen feature ablation (test: 2 TICs, demo only)")
    ax.legend(fontsize=8)
    _save(fig, figure_directory, FIGURE_STEMS[3])
    sources[FIGURE_STEMS[3]] = [MODEL]

    group = _json(repository / "data/evaluation" / ROBUSTNESS / "group_metrics.json")
    per_tic = group["per_tic"]
    plot_x = np.arange(len(per_tic))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(
        plot_x - width / 2,
        [row["pr_auc"] for row in per_tic],
        width,
        label="PR-AUC",
        color="#2563eb",
    )
    ax.bar(
        plot_x + width / 2,
        [row["f1"] for row in per_tic],
        width,
        label="F1",
        color="#059669",
    )
    ax.set_xticks(
        plot_x,
        [row["object_id"].replace("tic-", "TIC ") for row in per_tic],
        rotation=35,
        ha="right",
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("LOTO metric")
    ax.set_title("Grouped development robustness across six independent TICs")
    ax.legend()
    _save(fig, figure_directory, FIGURE_STEMS[4])
    sources[FIGURE_STEMS[4]] = [ROBUSTNESS]

    benchmark = _json(repository / "data/evaluation" / FINAL_EVALUATION / "benchmark_summary.json")
    injection = benchmark["end_to_end_injection_performance"]
    names = sorted(injection)
    plot_x = np.arange(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for offset, key, label, color in (
        (-width, "bls_recovery_rate", "BLS recovery", "#2563eb"),
        (0, "conditional_classifier_correctness", "Classifier | recovered", "#f59e0b"),
        (width, "end_to_end_success_rate", "End-to-end", "#059669"),
    ):
        ax.bar(
            plot_x + offset,
            [injection[name][key] for name in names],
            width,
            label=label,
            color=color,
        )
    ax.set_xticks(
        plot_x,
        [name.replace("_", " ").title() for name in names],
        rotation=20,
        ha="right",
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Controlled demo rate")
    ax.set_title("Synthetic-injection recovery stages")
    ax.legend(fontsize=8)
    _save(fig, figure_directory, FIGURE_STEMS[5])
    sources[FIGURE_STEMS[5]] = [ROBUSTNESS, FINAL_EVALUATION]

    final_dir = repository / "data/evaluation" / FINAL_EVALUATION
    scorecards = pd.read_parquet(final_dir / "candidate_scorecards.parquet")
    demo_eval = pd.read_parquet(final_dir / "demo_scorecard_evaluation.parquet")
    demo_scores = scorecards.loc[
        scorecards["dataset_role"] == "demo", ["candidate_id", "score"]
    ].merge(demo_eval[["candidate_id", "gold_candidate_label"]], on="candidate_id")
    science_scores = scorecards.loc[scorecards["dataset_role"] == "scientific_unlabeled", "score"]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bins = np.linspace(0, 1, 21)
    for histogram_label, histogram_values, color in (
        (
            "Demo positive",
            demo_scores.loc[demo_scores["gold_candidate_label"] == "positive", "score"],
            "#059669",
        ),
        (
            "Demo negative",
            demo_scores.loc[demo_scores["gold_candidate_label"] == "negative", "score"],
            "#dc2626",
        ),
        (
            "Demo unlabeled",
            demo_scores.loc[demo_scores["gold_candidate_label"] == "unlabeled", "score"],
            "#6b7280",
        ),
        ("Scientific unlabeled", science_scores, "#2563eb"),
    ):
        ax.hist(
            histogram_values.to_numpy(dtype=float),
            bins=bins.tolist(),
            alpha=0.45,
            density=True,
            label=histogram_label,
            color=color,
        )
    ax.axvline(0.325, color="black", linestyle="--", label="Frozen threshold 0.325")
    ax.set_xlabel("Frozen model score")
    ax.set_ylabel("Density")
    ax.set_title("Demo and unlabeled scientific score distributions")
    ax.legend(fontsize=8)
    _save(fig, figure_directory, FIGURE_STEMS[6])
    sources[FIGURE_STEMS[6]] = [FINAL_EVALUATION, MODEL]

    risk = (
        pd.read_parquet(final_dir / "shift_importance_risk.parquet")
        .head(10)
        .sort_values("risk_score")
    )
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(
        risk["feature"],
        risk["risk_score"],
        color=np.where(risk["risk_flag"] == "high", "#dc2626", "#f59e0b"),
    )
    ax.set_xlabel("Importance × shift risk score")
    ax.set_title("Strongest demo-to-scientific transfer risks")
    _save(fig, figure_directory, FIGURE_STEMS[7])
    sources[FIGURE_STEMS[7]] = [ROBUSTNESS, FINAL_EVALUATION]

    importance = (
        pd.read_parquet(final_dir / "global_importance.parquet")
        .head(12)
        .sort_values("permutation_importance_mean")
    )
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(
        importance["feature"],
        importance["permutation_importance_mean"],
        xerr=importance["permutation_importance_std"],
        color="#2563eb",
        alpha=0.85,
    )
    ax.set_xlabel("Validation permutation importance (PR-AUC decrease)")
    ax.set_title("Frozen Random Forest feature importance")
    _save(fig, figure_directory, FIGURE_STEMS[8])
    sources[FIGURE_STEMS[8]] = [MODEL, FINAL_EVALUATION]

    readiness = _json(final_dir / "scientific_readiness.json")
    criteria = readiness["criteria"]
    plot_y = np.arange(len(criteria))
    passed = np.array([bool(row["passed"]) for row in criteria])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(
        plot_y,
        np.ones(len(plot_y)),
        color=np.where(passed, "#16a34a", "#dc2626"),
    )
    ax.set_yticks(plot_y, [row["criterion"].replace("_", " ") for row in criteria])
    ax.set_xticks([])
    ax.set_xlim(0, 1)
    ax.set_title("Scientific readiness: blocked")
    for index, value in enumerate(passed):
        ax.text(
            0.5,
            index,
            "PASS" if value else "FAIL",
            ha="center",
            va="center",
            color="white",
            weight="bold",
        )
    _save(fig, figure_directory, FIGURE_STEMS[9])
    sources[FIGURE_STEMS[9]] = [FINAL_EVALUATION]
    return sources


def generate_final_results(root: str | Path, repro_directory: Path) -> Path:
    repository = Path(root)
    results = repository / "results"
    tables = results / "tables"
    figures = results / "figures"
    table_sources = build_final_tables(repository, tables)
    figure_sources = build_final_figures(repository, figures)
    repro_version = _json(repro_directory / "project_manifest.json")["repro_version"]
    (tables / "table_sources.json").write_text(
        canonical_json(table_sources) + "\n", encoding="utf-8"
    )
    (figures / "figure_sources.json").write_text(
        canonical_json(figure_sources) + "\n", encoding="utf-8"
    )
    generated_tables = [
        {
            "path": f"results/tables/{name}",
            "sha256": sha256_file(tables / name),
            "source_artifact_ids": table_sources[name],
        }
        for name in TABLE_NAMES
    ]
    generated_figures = [
        {
            "path": f"results/figures/{stem}.{suffix}",
            "sha256": sha256_file(figures / f"{stem}.{suffix}"),
            "source_artifact_ids": figure_sources[stem],
        }
        for stem in FIGURE_STEMS
        for suffix in ("png", "pdf")
    ]
    status = _json(repository / "data/evaluation" / FINAL_EVALUATION / "scientific_readiness.json")
    project_status = {
        "software_pipeline_status": "complete",
        "demo_pipeline_status": "complete",
        "scientific_readiness": "blocked",
        "scientific_blockers": status["failed_criteria"],
        "official_model": MODEL,
        "official_threshold": 0.325,
        "final_evaluation": FINAL_EVALUATION,
        "repro_version": repro_version,
    }
    index = {
        "repro_version": repro_version,
        "scientific_dataset": SCIENTIFIC_DATASET,
        "demo_dataset": DEMO_DATASET,
        "scientific_features": SCIENTIFIC_FEATURES,
        "demo_features": DEMO_FEATURES,
        "official_model": MODEL,
        "robustness_evaluation": ROBUSTNESS,
        "final_evaluation": FINAL_EVALUATION,
        "reproduction_manifest": f"data/reproducibility/{repro_version}/project_manifest.json",
        "tables": generated_tables,
        "figures": generated_figures,
        "table_source_metadata": "results/tables/table_sources.json",
        "figure_source_metadata": "results/figures/figure_sources.json",
        "project_status": "results/project_status.json",
    }
    results.mkdir(parents=True, exist_ok=True)
    (results / "project_status.json").write_text(
        canonical_json(project_status) + "\n", encoding="utf-8"
    )
    (results / "index.json").write_text(canonical_json(index) + "\n", encoding="utf-8")
    return results
