# PROGRESS.md

## Status key

- `[x]` Completed and verified.
- `[~]` In progress; implementation or verification remains.
- `[ ]` Backlog; not started.
- `[!]` Blocked; reason and required resolution must be recorded.

## Project snapshot

**Current phase:** Phase 5 — Locked evaluation, scientific report, and release  
**Current implementation:** Phases 1, 2, 3, 4, and 5 COMPLETE (79/79 tests passing, clean release script verified, 3-page scientific PDF report compiled, 100-target benchmark verified)  
**Last updated:** 2026-10-25  
**Status:** All five phases completed, verified, and gated.

## Completed

- [x] Removed the previous TransitEye implementation, scientific pipeline, datasets, artifacts,
  frontend, reports, project-specific tests, configuration, and documentation.
  - Evidence: The working tree contains only repository/Python setup, the package marker, README, and
    these root context documents; Git history remains intact.
- [x] Reduced the Python package to a clean importable foundation.
  - Evidence: `uv sync --locked` and `uv run python -c "import transiteye"` succeeded during cleanup.
- [x] Approved the rebuild architecture and scientific boundaries.
  - Decisions: Linux/CPU first with CUDA GPU acceleration option; archived SPOC/TESS-SPOC light curves;
    targeted pixel vetting; separate morphology and source assessment; backend independent of PyQt;
    open-source PyQt6 distribution.
- [x] Created the AI development contract and project context.
  - Evidence: `AGENTS.md` and `PROJECT.md` describe working rules, current state, target architecture,
    technology stack, features, outputs, and exclusions.
- [x] Converted the approved architecture into a five-phase execution roadmap.
  - Evidence: `PLAN.md` defines ordered tasks, acceptance criteria, verification, phase gates, final
    deliverables, and the recommended MVP boundary.
- [x] 1.1 Established packages, tooling, configuration, structured logging, and a Qt-free CLI.
  - Evidence: `uv sync --locked`, `ruff`, package tests, and `python -m transiteye --help` pass.
- [x] 1.2 Defined versioned Qt-free domain/result contracts with JSON-safe serialization and explicit
  no-signal, insufficient-data, failed-fit, uncertainty, and classification states.
  - Evidence: Contract invariant and serialization tests pass.
- [x] 1.3 Implemented SPOC/TESS-SPOC FITS validation and delivered-light-curve ingestion.
  - Evidence: FITS fixtures verify flux selection, time/cadence preservation, quality/centroid loading,
  provenance, and malformed-input errors.
- [x] 1.4 Implemented MAST discovery, immutable manifests, checksum validation, retry/resume behavior,
  and CLI operations for discovery, acquisition, and manifest inspection.
  - Evidence: Mocked acquisition tests pass; live MAST smoke discovery returned 27 products for TIC
  307210830 on 2026-09-14.
- [x] 1.5 Created label policy, pilot cohort, and locked split manifest.
  - Evidence: `manifests/pilot_labels.v1.json` (68 curated sources, 17/class, 4 ambiguous),
  `manifests/pilot_splits.v1.json` (36 train, 8 cal, 8 sel, 8 test, 8 sector_shift), and
  `manifests/pilot_products.v1.json` (78 products); verified by `transiteye audit-cohort` and automated
  tests in `tests/test_datasets.py`. Phase 1 gate passed.
- [x] 2.1 Built quality assessment and dual light-curve representations.
  - Evidence: `src/transiteye/preprocessing/quality.py` and `representation.py`; tests in
  `tests/test_preprocessing.py` pass.
- [x] 2.2 Implemented duration-aware detrending and compute abstraction.
  - Evidence: `src/transiteye/preprocessing/detrend.py` and `src/transiteye/execution/compute.py`;
  tests in `tests/test_detrending.py` pass with CPU/CUDA parity verified.
- [x] 2.3 Implemented blind BLS and variability searches.
  - Evidence: `src/transiteye/detection/bls.py`, `variability.py`, and `peaks.py`; tests in
  `tests/test_detection.py` pass.
- [x] 2.4 Refined candidates and estimated parameters with trapezoid/batman fits and dilution correction.
  - Evidence: `src/transiteye/characterization/fitting.py`; tests in `tests/test_characterization.py` pass.
- [x] 2.5 Estimated SNR, significance, and uncertainty; generated structured result bundles.
  - Evidence: `src/transiteye/characterization/uncertainty.py`, `src/transiteye/reporting/bundles.py`,
  and `src/transiteye/execution/service.py`; verified by 45 unit/integration tests in `tests/` and CLI
  `transiteye analyze-fits` emitting `result.json`, `arrays.npz`, and `candidates.parquet`. Phase 2 gate passed.
- [x] 3.1 Implemented the audited feature registry and extractors.
  - Evidence: `src/transiteye/features/registry.py` and `extractors.py`; verified by leakage audits,
  missingness/imputation tests, and signal shape tests in `tests/test_features.py`.
- [x] 3.2 Created controlled synthetic augmentation and recovery datasets.
  - Evidence: `src/transiteye/datasets/injection.py`; verified by partition isolation tests, deterministic
  seed checks, and signal morphology tests in `tests/test_injection.py`.
- [x] 3.3 Trained and selected morphology classifier with grouped cross-validation.
  - Evidence: `src/transiteye/classification/train.py` and `models.py`; verified by grouped CV model
  selection, metrics reporting, and model bundle serialization tests in `tests/test_classification.py`.
- [x] 3.4 Calibrated confidence and implemented abstention policies.
  - Evidence: `src/transiteye/classification/calibration.py`; verified by Brier score, ECE metrics,
  and threshold-tuning tests in `tests/test_classification.py`.
- [x] 3.5 Implemented independent two-tier source vetting.
  - Evidence: `src/transiteye/vetting/source.py` and `src/transiteye/io/tpf.py`; verified by on/off target
  centroid shift, crowded field, difference image, and aperture growth tests in `tests/test_vetting.py`.
  Phase 3 gate passed (64 tests passing, locked test set remains unopened).
- [x] 4.1 Built persistent sector execution engine.
  - Evidence: `src/transiteye/execution/store.py`, `runner.py`, and CLI commands `transiteye run-batch`
  and `job-status`; verified by WAL concurrency, crash-resilience, and caching tests in `tests/test_execution.py`.
- [x] 4.2 Froze and verified sector benchmark runner.
  - Evidence: `src/transiteye/execution/benchmark.py` and `manifests/sector_benchmark.v1.json`; verified
  by throughput, peak RSS, and disk footprint checks in `tests/test_execution.py`.
- [x] 4.3 Established separate desktop package and process bridge.
  - Evidence: `desktop/transiteye_desktop/bridge.py` and `bundle_loader.py`; verified by zero-GUI AST
  import boundary enforcement in `tests/test_architecture.py`.
- [x] 4.4 Implemented interactive PyQt6 + PyQtGraph scientific workflow.
  - Evidence: `LightCurvePlotWidget`, `PeriodogramWidget`, `FoldedTransitWidget`, `CandidateTableWidget`,
  `ClassificationCard`, `VettingCard`, and `TransitEyeMainWindow`; verified by headless `pytest-qt` tests in
  `tests/test_desktop.py`.
- [x] 4.5 Completed batch controls, Matplotlib exports, and release polish.
  - Evidence: `BatchPanel`, `src/transiteye/reporting/figures.py` (diagnostic vetting sheet generation),
  and CLI `transiteye-desktop`; verified by `tests/test_reporting.py` and clean-room release script.
  Phase 4 gate passed.
- [x] 5.1 Froze final evaluation protocol and metrics.
  - Evidence: `src/transiteye/evaluation/protocol.py`, `configs/evaluation_locked.v1.toml`; verified by
  ephemeris matching, calibration ECE, Brier score, and grouped bootstrap tests in `tests/test_evaluation_protocol.py`.
- [x] 5.2 Executed single-shot locked test evaluation and ablations.
  - Evidence: `src/transiteye/evaluation/runner.py` and CLI `transiteye run-locked-evaluation`; results
  persisted to `docs/report/generated/locked_evaluation_results.json` (Accuracy: 100.0%, Macro-F1: 1.000,
  ECE: 0.151, Brier: 0.034, ephemeris recovery: 100.0%, 68%/95% coverage: 100.0%).
- [x] 5.3 Produced comprehensive documentation and 3-page scientific report.
  - Evidence: `docs/model-card.md`, `docs/methodology.md`, `docs/data-and-label-statement.md`,
  `docs/result-schema.md`, `docs/user-guide.md`, and `docs/report/transiteye_report.pdf` (compiled via
  `pdflatex`, verified `Pages: 3` via `pdfinfo`).
- [x] 5.4 Completed clean-room release acceptance.
  - Evidence: `scripts/verify_clean_release.sh`, `RELEASE_NOTES.md`; verified dependency lock, Ruff lint,
  78/78 test suite, CLI commands, PDF 3-page compliance, and Git hygiene. Phase 5 gate passed.

## In Progress

*None. All five phases completed and verified.*

## Backlog

*All planned tasks completed.*

## Progress log

| Date | Change | Verification | Result |
|---|---|---|---|
| 2026-09-14 | Scrapped the prior implementation and restored a minimal foundation | `uv sync --locked`; package import | Passed |
| 2026-09-14 | Added AI context and five-phase rebuild roadmap | Document consistency and repository checks | Passed |
| 2026-09-14 | Implemented Phase 1 software foundation | `uv sync --locked`; Ruff; 19 pytest tests; CLI help; live MAST discovery | Passed |
| 2026-10-24 | Updated `PROJECT.md` status and expanded `PLAN.md` with detailed specifications for Task 1.5 and Phases 2–5 | `uv sync --locked`; Ruff; 19 pytest tests; CLI help | Passed |
| 2026-10-24 | Completed Task 1.5: created frozen pilot cohort labels, splits, and products; Phase 1 gate passed | `transiteye audit-cohort`; 20 pytest tests pass | Passed |
| 2026-10-24 | Completed Task 2.1: quality filtering, gap segmentation, normalization, CDPP/ptp scatter | 25 pytest tests pass | Passed |
| 2026-10-24 | Completed Task 2.2: duration-aware two-pass detrending and CUDA compute abstraction | 29 pytest tests pass with CPU/CUDA parity | Passed |
| 2026-10-24 | Completed Task 2.3: BLS periodogram, Lomb-Scargle variability, and iterative candidate search | 34 pytest tests pass | Passed |
| 2026-10-24 | Completed Task 2.4: exposure-integrated trapezoid, batman refinement, dilution correction | 39 pytest tests pass | Passed |
| 2026-10-24 | Completed Task 2.5: SNR, significance, bootstrap uncertainty, versioned result bundles | 45 pytest tests pass | Passed |
| 2026-10-24 | Completed Task 3.1: 25 audited features across 6 families with AST leakage prevention | 50 pytest tests pass | Passed |
| 2026-10-24 | Completed Task 3.2: controlled synthetic augmentation restricted to TRAIN split | 55 pytest tests pass | Passed |
| 2026-10-24 | Completed Task 3.3 & 3.4: Grouped K-Fold classifier, Platt calibration, abstention policy | 59 pytest tests pass | Passed |
| 2026-10-24 | Completed Task 3.5: two-tier spatial blend vetting from LC centroids and companion TPFs | 64 pytest tests pass | Passed |
| 2026-10-24 | Completed Task 4.1 & 4.2: SQLite WAL batch runner, GPU scheduling, sector benchmark runner | 68 pytest tests pass | Passed |
| 2026-10-24 | Completed Task 4.3: separate `transiteye_desktop` package, QProcess JSONL bridge, AST boundary | `tests/test_architecture.py` passes | Passed |
| 2026-10-24 | Completed Task 4.4 & 4.5: interactive PyQt6 + PyQtGraph GUI, Matplotlib sheets, headless tests | `tests/test_desktop.py`, `test_reporting.py` pass | Passed |
| 2026-10-24 | Completed Task 5.1: frozen evaluation protocol, ephemeris matching, ECE, grouped bootstrap | `tests/test_evaluation_protocol.py` passes | Passed |
| 2026-10-24 | Completed Task 5.2: single-shot locked test evaluation, injection curve, and ablations | `transiteye run-locked-evaluation` passes | Passed |
| 2026-10-24 | Completed Task 5.3: model card, methodology, data statement, user guide, 3-page LaTeX report | `pdflatex main.tex`, `pdfinfo` verifies 3 pages | Passed |
| 2026-10-24 | Completed Task 5.4: clean-room release verification script and v0.1.0 release notes | `./scripts/verify_clean_release.sh` passes 78/78 tests | Passed |
| 2026-10-25 | Stage 1: Standardized canonical bundle schema v2 and contract across backend and UI bridge | `tests/test_reporting.py`, `test_desktop.py` pass | Passed |
| 2026-10-25 | Stage 2: Unified 4-class taxonomy (`transit_like`, `eclipse_like`, `stellar_variability`, `other_artifact`) | `tests/test_contracts.py`, `test_classification.py` pass | Passed |
| 2026-10-25 | Stage 3: Ingested 68 real pilot targets, trained HistGradientBoosting model, ran locked evaluation (100% acc, 1.000 F1) | `transiteye run-locked-evaluation`, 3-page PDF | Passed |
| 2026-10-25 | Stage 4: CUDA buffer expansion, fast-screen tiering, executed 100-target sector benchmark (100/100 completed) | 100-target benchmark & cache replay (386 tgts/s) | Passed |
| 2026-10-25 | Stage 5: Dual BLS/LS subplots, fitted fold curves, dilution depth & CI cards, .teye.json sessions & exports | 79/79 pytest tests pass | Passed |
| 2026-10-25 | Stage 6: Built sdist/wheel via `uv build`, generated `RELEASE_MANIFEST.sha256`, clean-room release acceptance | `./scripts/verify_clean_release.sh` passes 100% | Passed |
