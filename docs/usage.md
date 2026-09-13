# Usage guide

## 1. Install

Use Python 3.11 and synchronize the exact locked environment:

```bash
uv sync --all-groups --locked
```

Run commands from the repository root. Verification requires the accepted
frozen `data/` tree supplied with the project distribution. Because scientific
artifacts are intentionally generated/local rather than ordinary source files,
a source-only Git checkout must restore that artifact bundle at its recorded
relative paths. Runtime caches are not required.

## 2. Verify frozen artifacts

```bash
uv run python scripts/reproduce_project.py --verify
```

This offline, read-only command checks all registered immutable files, SHA-256
values, and available identity references. Missing or changed artifacts cause a
useful failure.

## 3. Inspect canonical results

Open `results/index.json` first. It points to the scientific and demo datasets,
both feature matrices, the official model, robustness and final evaluations,
tables, figures, and reproducibility manifest. `results/project_status.json`
contains the authoritative readiness state and official IDs.

## 4. Regenerate tables and figures

```bash
uv run python scripts/reproduce_project.py --report
```

The command verifies inputs before and after rebuilding `results/tables/` and
`results/figures/`. It works offline and does not modify scientific, feature,
model, or evaluation artifacts.

## 5. Run the compact demo

```bash
uv run python scripts/reproduce_project.py --demo-smoke
```

The smoke path sends a tiny deterministic box transit through existing
preprocessing and BLS APIs, then runs frozen inference on two demo feature rows.
It writes `results/demo_smoke.json` and requires no network access.

## 6. Optional full replay

```bash
uv run python scripts/reproduce_project.py --full-replay
```

This expensive local replay invokes existing demo, feature, modeling,
robustness, and final-evaluation scripts. It does not redownload MAST products.
It is unnecessary for ordinary inspection or presentation.

## 7. Output locations

Use the pointers in `results/index.json`. In summary, immutable artifacts live
under `data/`; report assets under `results/tables/` and `results/figures/`; the
long report under `reports/`; and presentation materials under `presentation/`.

## 8. Candidate scorecards

Resolve `final_evaluation` from `results/index.json`, then inspect
`candidate_scorecards.parquet` in that versioned evaluation directory. Each row
preserves candidate/TIC identity, score, threshold, BLS and selected diagnostic
features, support counts, provenance, and a neutral triage status. Scientific
rows have dataset role `scientific_unlabeled` and contain no gold label.

## 9. Scientific readiness

Read `results/project_status.json` for the current gate and
`results/tables/table_7_scientific_readiness.csv` for criterion-level evidence.
The current scientific state is blocked while the demo pipeline is complete.
