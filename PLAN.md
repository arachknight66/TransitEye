# PLAN.md

## Roadmap contract

This is the authoritative five-phase implementation sequence. Work within a phase follows task order
unless a task explicitly states otherwise. A phase is complete only after its verification gate
passes and `PROGRESS.md` records the evidence.

## Build preferences

- **Mode:** Autonomous implementation in reviewable vertical slices.
- **Verification:** Automated checks at every slice and a mandatory phase gate.
- **Git cadence:** One coherent commit per completed vertical slice when commits are requested.
- **Platform:** Linux and CPU first; no GPU requirement.
- **Data scope:** Archived SPOC/TESS-SPOC light curves with targeted target-pixel vetting.
- **Scientific policy:** Favor defensible baselines; add complexity only after measured failure.

## Phase 1 — Contracts, ingestion, and curated data foundation

**Goal:** Establish trustworthy inputs, typed outputs, dataset provenance, and leakage-safe partitions.

- [ ] **1.1 Establish packages, tooling, and configuration**
  - Build: Add constrained runtime/dev groups, backend package boundaries, structured logging, typed
    configuration, and CLI entry point.
  - Acceptance: A clean clone syncs, imports, lints, tests, and exposes CLI help.
  - Verify: `uv sync --locked && uv run ruff check . && uv run pytest && uv run python -m transiteye --help`

- [ ] **1.2 Define domain and result contracts**
  - Build: Immutable typed records for light curves, provenance, masks, configurations, candidates,
    classification/source assessments, uncertainties, progress, and complete analysis results.
  - Acceptance: Units, missing-value reasons, versions, no-signal, insufficient-data, and failure
    states serialize and round-trip without Qt imports.
  - Verify: Contract/schema unit and compatibility tests.

- [ ] **1.3 Implement FITS ingestion and validation**
  - Build: SPOC/TESS-SPOC adapters for time, PDCSAP/SAP, flux errors, cadence, quality, centroids,
    crowding metadata, and FITS provenance.
  - Acceptance: Reference fixtures preserve time standard, units, cadence, flags, arrays, and source
    metadata; malformed and unsupported inputs fail clearly.
  - Verify: Fixture-based FITS tests and manual CLI inspection of representative products.

- [ ] **1.4 Implement MAST discovery and immutable manifests**
  - Build: Product discovery, bounded downloads, checksums, retry/resume, caching, and offline replay.
  - Acceptance: A frozen pilot manifest reproduces the same inventory; interrupted downloads resume;
    unavailable products are accounted for.
  - Verify: Mocked network tests plus online pilot acquisition recorded by manifest checksum.

- [ ] **1.5 Create label policy, pilot cohort, and locked split manifest**
  - Build: Curate confirmed/known transits, vetted eclipsing binaries, cataloged variables, and
    reviewed other/artifact cases with evidence and ambiguity fields. Create source-grouped 60/10/10/20
    train/calibration/selection/test partitions and a sector-shift set.
  - Acceptance: Every label has provenance; conflicts and uncertainty are explicit; no astrophysical
    source or derivative crosses partitions; the test set remains sealed.
  - Verify: Automated label audit, class/support report, and zero-overlap split checks.

**Phase gate:** Frozen pilot inputs and labels are auditable, schemas are versioned, and leakage tests
pass. No model training may begin before this gate.

## Phase 2 — Preprocessing, detection, and characterization baseline

**Goal:** Produce scientifically interpretable candidates and parameter estimates without learned
classification.

- [ ] **2.1 Build quality assessment and dual light-curve representations**
  - Build: Invalid-sample handling, documented TESS quality masks, gap/sector segmentation,
    normalization, robust scatter/noise diagnostics, and minimally processed output.
  - Acceptance: No interpolation crosses gaps; negative excursions remain; every removed point has a
    recorded reason.
  - Verify: Synthetic edge cases and known FITS fixtures.

- [ ] **2.2 Implement duration-aware detrending**
  - Build: Wotan biweight detrending with configurable windows and event-masked second pass; retain
    trend and masks.
  - Acceptance: Injection tests quantify recovery and parameter bias across durations, variability,
    cadence, and gaps; defaults are frozen from development data only.
  - Verify: Signal-preservation test matrix and diagnostic plots.

- [ ] **2.3 Implement blind BLS and variability searches**
  - Build: Uncertainty-weighted BLS, Lomb-Scargle, refined peak neighborhoods, harmonic alternatives,
    odd/even and secondary checks, distinct-peak selection, and residual searches.
  - Acceptance: Known and injected signals recover within defined tolerances; sparse/short light
    curves return explicit states; maximum candidate count is enforced.
  - Verify: Detection unit tests, injection recovery, and blind pilot run.

- [ ] **2.4 Refine candidates and estimate parameters**
  - Build: Exposure-integrated trapezoid fits, local baselines, optional `batman` refinement for
    supported transit-like candidates, and alternate-period comparison.
  - Acceptance: Results report epoch/time standard, period, depth, duration, event count, phase
    coverage, fit diagnostics, aliases, and measured versus dilution-corrected depth distinctly.
  - Verify: Simulation bias/error tests and regression fixtures.

- [ ] **2.5 Estimate SNR, significance, and uncertainty**
  - Build: Depth SNR with correlated-noise correction, residual block-bootstrap intervals, and a
    noise-surrogate empirical search significance path for shortlisted candidates.
  - Acceptance: The three score types are separately named; 68%/95% intervals and finite-surrogate
    resolution are exposed; failed and multimodal fits remain visible.
  - Verify: Coverage experiments, null tests, and deterministic seed reproduction.

**Phase gate:** The CLI analyzes local FITS end to end, produces versioned result bundles, and meets
frozen development thresholds for recovery, false alerts, parameter error, and interval coverage.

## Phase 3 — Features, multiclass classification, and source vetting

**Goal:** Classify candidate morphology with calibrated confidence and assess likely contamination.

- [ ] **3.1 Implement the audited feature registry**
  - Build: Morphology, consistency, eclipse, variability, noise/quality, and source-evidence feature
    families with definitions, units, missing policy, and lineage.
  - Acceptance: Features contain no prohibited label/identity/split data and remain stable across
    serialization and batch execution.
  - Verify: Leakage audit, invariance tests, schema snapshot, and feature-distribution review.

- [ ] **3.2 Create controlled synthetic augmentation and recovery datasets**
  - Build: Inject exposure-integrated transits/eclipses into training-only real-noise substrates with
    recorded truth and deterministic seeds.
  - Acceptance: No test substrate or source appears in augmentation; synthetic examples remain
    identifiable and never enter real-data headline metrics.
  - Verify: Lineage checks and stratified recovery summaries.

- [ ] **3.3 Train and select the morphology classifier**
  - Build: Compare multinomial logistic regression, class-weighted random forest, and histogram
    gradient boosting using grouped CV. Select by macro-F1, then transit recall and simplicity.
  - Acceptance: One reproducible model bundle records preprocessing, features, training manifest,
    code/config versions, and evaluation evidence.
  - Verify: Grouped CV report, confusion matrices, class metrics, and reproducible model hash.

- [ ] **3.4 Calibrate confidence and add abstention**
  - Build: Sigmoid calibration on the held-out calibration partition and thresholds selected on the
    selection partition for ambiguous/unsupported candidates.
  - Acceptance: Reliability, log loss/Brier-style calibration metrics, and accuracy-versus-coverage
    support the chosen policy; confidence is labeled as classifier confidence.
  - Verify: Calibration curves, abstention tests, and partition-use audit.

- [ ] **3.5 Implement independent source vetting**
  - Build: Centroid diagnostics, difference-image location, aperture sensitivity, target-pixel review,
    crowding metadata, and nearby-source evidence.
  - Acceptance: Return target-consistent, blend-suspected, or unresolved independently of morphology;
    missing pixel evidence never becomes a definitive blend decision.
  - Verify: Known on/off-target cases, synthetic centroid shifts, and evidence-display review.

**Phase gate:** Grouped real-data validation supports the deployed classes, calibration, abstention,
and source-assessment claims. The locked test set remains unopened.

## Phase 4 — Sector execution and PyQt desktop product

**Goal:** Scale the validated backend and expose it through a polished, responsive desktop workflow.

- [ ] **4.1 Build persistent sector execution**
  - Build: SQLite job state, bounded multiprocessing, single-coordinator writes, checksum-keyed cache,
    cancellation, resume/retry, and fast-screen versus detailed-analysis tiers.
  - Acceptance: Each target is independently restartable; results are deterministic; concurrency and
    memory remain bounded; every skipped/failed target is counted.
  - Verify: Interruption/restart integration tests and increasing-load benchmarks.

- [ ] **4.2 Freeze and run a sector benchmark**
  - Build: Execute one immutable sector manifest and record throughput, runtime, peak memory, disk,
    cache behavior, unavailable inputs, and failures.
  - Acceptance: The full manifest reaches a terminal state with reproducible output inventory and no
    silent losses.
  - Verify: Manifest reconciliation and clean rerun/cache-hit checks.

- [ ] **4.3 Establish the separate desktop package and process bridge**
  - Build: PyQt6 Widgets shell, presenter/controller boundary, typed job requests and progress events,
    supervised backend `QProcess`, session persistence, and error recovery.
  - Acceptance: No Qt imports exist in `transiteye`; the UI remains responsive during analysis and can
    cancel/reconnect; CLI and desktop resolve identical result bundles.
  - Verify: Architecture import test, pytest-qt process tests, and CLI/UI equivalence test.

- [ ] **4.4 Implement the interactive scientific workflow**
  - Build: FITS opening, linked delivered/denoised plots, mask/trend overlays, BLS/variability views,
    peak and candidate selection, phase-folding, fit/residual views, probabilities, source evidence,
    parameter/SNR/significance/uncertainty panels, and no-signal/error states.
  - Acceptance: Candidate selection synchronizes all views; displayed units/definitions are clear;
    display downsampling does not affect calculations.
  - Verify: UI behavior tests and manual review on representative signal classes and failures.

- [ ] **4.5 Complete batch controls, exports, and visual polish**
  - Build: Sector queue, progress/cancel/resume/retry controls, saved sessions, CSV/Parquet/JSON/array
    exports, Matplotlib PNG/PDF/SVG figures, accessible styling, keyboard navigation, and packaging.
  - Acceptance: A new Linux user can install, analyze one FITS file, run a batch, reopen a session, and
    export a complete result without the UI freezing.
  - Verify: Headless UI suite, packaged-app smoke test, accessibility review, and export round trips.

**Phase gate:** The sector benchmark passes, the packaged Linux desktop completes the full workflow,
and desktop results match the backend CLI.

## Phase 5 — Locked evaluation, scientific report, and release

**Goal:** Establish final evidence, document limitations, and produce a reproducible release.

- [ ] **5.1 Freeze the final evaluation protocol**
  - Build: Lock matching rules, metrics, thresholds, bootstrap grouping, robustness strata, ablations,
    report layout, and artifact manifest before reading locked test labels.
  - Acceptance: Protocol covers detection, classification, parameters, calibration, source assessment,
    end-to-end misses, shift, and runtime; no success metric can silently drop failures.
  - Verify: Independent protocol checklist and configuration checksum.

- [ ] **5.2 Run locked and robustness evaluations once**
  - Build: Evaluate grouped real test data, sector shift, injections, feature/detrending ablations, and
    runtime with target-group confidence intervals.
  - Acceptance: Outputs include support counts, failures, harmonics, abstentions, limitations, and
    distinct real/synthetic conclusions. No tuning follows test inspection.
  - Verify: Reproduction from frozen manifests and checksum reconciliation.

- [ ] **5.3 Produce publication and user deliverables**
  - Build: Model card, methodology, data/label statement, result schema, user guide, reproducible
    figures/tables, and a scientific report of at most three pages including references.
  - Acceptance: Every headline statement traces to a frozen artifact; figures remain readable at final
    size; the report covers problem/data/method, evaluation/results, limitations/conclusion.
  - Verify: Link/provenance checks, rendered page-count and visual inspection, documentation smoke test.

- [ ] **5.4 Perform clean-room release acceptance**
  - Build: Produce the backend package, GPLv3-compatible desktop distribution, notices, checksums,
    release notes, and exact reproduction commands.
  - Acceptance: From a clean environment, dependencies lock, tests pass, package imports, CLI works,
    desktop launches, reference analysis matches, exports open, and no large/generated data is tracked.
  - Verify: Automated acceptance script plus manual Linux installation walkthrough.

**Phase gate:** All deliverables reproduce from documented inputs, scientific claims match the locked
evidence, the three-page limit passes, and the release workflow succeeds from a clean environment.

## Recommended MVP cut

The first usable MVP ends at **Phase 3** and consists of:

```text
SPOC/TESS-SPOC FITS
  → quality filtering and duration-aware robust detrending
  → BLS plus Lomb-Scargle
  → trapezoidal characterization and uncertainties
  → compact audited features
  → calibrated random forest with abstention
  → independent contamination assessment
  → versioned CLI result bundle
```

Build order is fixed: **contracts/data → preprocessing/detection → characterization/uncertainty →
features/classification/vetting → sector execution → desktop → locked evaluation/report/release**.

