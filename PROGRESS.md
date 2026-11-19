# PROGRESS.md

## Status key

- `[x]` Completed and verified.
- `[~]` In progress; implementation or verification remains.
- `[ ]` Backlog; not started.
- `[!]` Blocked; reason and required resolution must be recorded.

## Project snapshot

**Current phase:** R4 — Independent validation and release
**Current implementation:** Historical phases are implemented, but their release and scientific gates
require revalidation. R1 guardrails have passed. The real MAST cohort has a historical locked
proxy-label evaluation; its test split is already open, and the later EB holdout remains sealed.
**Last updated:** 2026-11-18
**Status:** R1–R3 passed; R4 scientific and release gates remain open.

## Completed

- [x] Downloaded and verified the full Kaggle Kepler Labelled Time Series Data v3 archive and
  both CSVs on 2026-11-18. The archive and CSVs occupy 334 MiB on disk in the ignored workspace
  at `.transiteye-workspace/data/kaggle_kepler_labelled/`.
  [Acquisition manifest](manifests/kaggle_kepler_labelled.v1.acquisition.json) records SHA-256,
  byte sizes, and CSV dimensions. Research only. No label-policy or training changes.

- [x] User-requested cleaning and training of the isolated labeled candidate-sequence experiment:
  combined only unambiguous transit-like/eclipsing-like sidecars, deduplicated by product, removed
  623 TICs already assigned to prior nontrain splits, excluded seven products without valid event
  timing, and froze the cohort before opening FITS. The frozen cohort contains 23,084 products from
  6,886 TIC groups, with 6,255 development and 631 fresh-holdout groups. Blind searches matched
  3,940 development candidates; 2,315 were recorded as unmatched. Fit a fresh same-candidate random
  forest and a PyTorch CNN from scratch on 3,162 train sequences; calibrated on 392 and selected on
  386. Selection macro-F1: CNN 0.9254, random forest 0.8158; these selection results are not
  independent performance estimates. The CNN selection confusion matrix (transit-like, eclipse-like
  order) is `[[25, 5], [3, 353]]`; selection support is 30 and 356 respectively. After locking the
  model, performed one inference-only evaluation on the 631-source fresh holdout. The blind matcher
  found 364 candidates (355 eclipse-like, 9 transit-like); 267 sources had no matched candidate.
  Holdout CNN macro-F1/balanced accuracy were 0.7429/0.7693; the same-candidate forest scored
  0.6135/0.8410. The transit-like support is only nine, so its class metrics are highly uncertain;
  overall accuracy is dominated by the eclipse-like majority. The holdout is now spent and must not
  be used for tuning. No desktop integration or four-class claim. Report, model weights, baseline, matched
  sequences, and candidate audit are stored outside Git in
  `.transiteye-workspace/models/labeled_candidate_sequence_v1/`. Cohort manifest SHA-256:
  `16fb30ca0d6d1ab93a001d53757750fb9a5832dc1b73d6292aa95414276e07d6`. Ruff, Python compilation,
  checkpoint reload/forward pass, holdout artifact-hash and coverage checks, and `git diff --check`
  passed; pytest was not run.

- [x] User-requested eclipsing-binary label cross-reference: exact-joined 32,057 downloaded SPOC
  products to three frozen EB catalog tables and required a catalog-listed product sector plus a
  primary-eclipse ephemeris window within the FITS product interval. Assigned 10,595 `eclipse_like`
  catalog-proxy labels across 4,305 TIC source groups; left 21,462 products unlabeled (20,774
  catalog-sector mismatches, 542 context-only matches without sector data, 146 primary epochs
  outside the product interval). All FITS TIC/sector headers matched, and the frozen product
  manifest remains unlabeled. The Parquet sidecar, all 32,532 catalog-event matches, and provenance
  report are under `.transiteye-workspace/data/mast_eclipsing_binary_catalogs_v1/`; generator:
  `scripts/label_mast_eclipsing_binary_catalogs.py`. Ruff, Python compilation, and sidecar integrity
  checks passed. The labels are not flux-shape or source-local validation and cannot serve as
  independent morphology truth.

- [x] Added TOI-derived labels to the eclipsing-binary cohort: exact-joined the frozen 8,175-row
  ExoFOP-TESS snapshot (retrieved 2026-11-11; SHA-256
  `d7b12ec7e2fd1c59850eff41a6aee19714294afd14d9a734c4f3f0fa4e70c786`) to all 32,057 products by
  TIC and required same-sector evidence. The approved dispositions yielded 248 `transit_like` TOI
  proxy products; 441 products with same-sector ambiguous/unsupported dispositions and 303 with
  off-sector TOIs remain unlabeled, as do 31,065 without a TOI match. There are 105 products labeled
  by both catalog overlays, all with conflicting `eclipse_like`/`transit_like` product classes;
  the combined sidecar leaves these unlabeled. It provides 10,490 unambiguous `eclipse_like` and
  143 `transit_like` product proxies, with 21,424 unlabeled. Separate TOI and combined Parquet
  overlays, all event matches, the catalog snapshot, and report are in
  `.transiteye-workspace/data/mast_eclipsing_binary_catalogs_v1/`; generator:
  `scripts/crossmatch_eb_products_toi.py`. Ruff, Python compilation, and output consistency checks
  passed. Same-sector proxies still require event-to-candidate matching before candidate-level
  training; no catalog absence was treated as a negative.

- [x] Completed the confirmed-exoplanet TESS cohort and event-timed transit proxy overlay: the
  frozen NASA Exoplanet Archive snapshot contains 6,464 planet rows (6,157 with TIC IDs, 4,561
  distinct host TICs); 4,242 hosts have 22,880 SPOC sector products and 319 have no matching product.
  Downloaded all 22,880 products with zero failures (15,163 previously verified, 7,717 newly
  downloaded). Exact host-TIC joins plus each confirmed transiting planet's period, BJD transit
  epoch, and duration yielded 16,989 `transit_like` catalog-proxy product labels across 3,290 TICs.
  The other 5,891 products remain unlabeled: 4,436 host-sector products have no confirmed planet
  with `discoverymethod=Transit`, 227 have incomplete transit ephemerides, and 1,228 have no predicted
  transit window in the product interval. All FITS TIC/sector headers matched; the frozen product
  manifest still has every morphology label unset. Parquet labels, per-planet event matches, and a
  provenance report are under `.transiteye-workspace/data/mast_confirmed_exoplanet_tess_v1/`;
  generator: `scripts/label_confirmed_planet_transits.py`. Ruff, Python compilation, and sidecar
  integrity checks passed. The archive snapshot lacks ephemeris uncertainties; time-window labels
  use stored parameter precision and duration and are not flux-shape or source-local ground truth.

- [x] User-requested restoration of the `real_catalog_cohort_v3` training dataset directly through
  the MAST Portal Download API: acquired and verified 986 latest-sector FITS products (774 train,
  104 calibration, 108 selection) in the external workspace. All 986 match frozen SHA-256 values
  and FITS TIC/sector/table/column checks; 123 v3 test-assigned sources were excluded before archive
  requests. The test partition is not a fresh independent holdout because some identities overlap
  the historically opened v1 test. Product manifest ID `be22af78c3a062f6`; split manifest ID
  `ed040a75524d3c72`. Acquisition details are recorded in
  `.transiteye-workspace/data/real_catalog_cohort_v3_dev_acquisition.v1.json`. Labels remain catalog
  proxies; no model was trained or evaluated in this acquisition task.

- [x] Fourth-class review-candidate acquisition: queried 100 NASA TOI false-alarm events and,
  after known prior/reserved identity exclusions and confirmed-planet checks, downloaded 59 TESS
  SPOC light curves. All 59 passed target/sector, FITS checksum and TransitEye ingestion checks;
  41 have a good-quality cadence near the approximate original TOI epoch. Three targets have a
  possible Gaia SOLAR_LIKE classification. `docs/fourth-class-candidates-v1.md` and external
  `data/false_alarm_candidates_v1/` record the sample and its limitations. These are unreviewed
  false-alarm candidates, not verified other/artifact labels. The fourth-class model was not
  trained on them; source- and event-local adjudication remains necessary.

- [x] Retrained the three-class research estimator on both rounds of examples. From 450 entries,
  74 were excluded for source conflicts or reserved partitions; 23 of the remaining 376 failed
  catalog matching. A 200-tree random forest trained on 307 usable sources; 46 held development
  validation features from all three classes. Source groups stayed disjoint and all 151 prior
  source partitions/checkpoints matched. Grouped training CV macro-F1 was 0.908 and development
  validation macro-F1 was 0.978; both are exploratory, selected data rather than independent test
  evidence. Artifact, hashes and full counts are in external `remediation/three-class-research-v3/`
  and `docs/three-class-research-model-v2.md`. The estimator is uncalibrated, lacks other/artifact
  examples, is not an application ModelBundle, and does not change desktop behavior. Holdout
  observations and labels were not opened.

- [x] Added 225 exploratory TESS light curves: 75 confirmed transiting-planet hosts, 75 Gaia
  eclipse-like candidates and 75 Gaia stellar-variability candidates. Excluded prior collection and
  known reserved TIC/Gaia identities before opening files; checked zero known identity overlap with
  the original examples and between the new sets. All 225 pass data-inclusive FITS checksums,
  TIC/sector verification and TransitEye ingestion. Catalog snapshots, selection and provenance
  are in `data/confirmed_transiting_hosts_v2/`, `data/eclipse_like_gaia_v2/`, and
  `data/stellar_variability_gaia_v2/` in the external workspace. SHA-256 values were calculated
  from MAST bytes, not compared to independently published archive SHA-256 values. Catalog labels
  remain provisional and the sets are not independent evaluation data; the sealed holdout was not
  opened. Alias exclusion is limited to known identity mappings.

- [x] Exploratory Gaia-variable download: selected and downloaded 75 stellar-variability and 75
  eclipse-like TESS light curves using Gaia DR3 catalog classifications and TIC cross-matches.
  Each set contains 75 distinct Gaia sources and TIC targets, with no TIC overlap between the sets;
  all 150 passed SHA-256, data-inclusive FITS checksums, target/sector identity and TransitEye
  ingestion. The stellar set is catalog-classified, and the eclipse set consists of ECL candidates
  also in Gaia's eclipsing-binary table; neither is independently adjudicated ground truth. Catalog
  snapshots, selection heuristics, provenance and source files are in external workspace
  `data/stellar_variability_gaia_v1/`, `data/eclipse_like_gaia_v1/`, and
  `data/gaia_variable_acquisition/`. These are exploratory and may overlap opened data; the sealed
  TESS EB holdout was not opened, and no training split or classifier labels were changed.

- [x] Exploratory confirmed-transiting-planet download: queried NASA Exoplanet Archive confirmed
  planets, selected 75 distinct TIC hosts with BJD-TDB ephemerides and repeated local dips at the
  predicted times, and downloaded matching TESS light-curve FITS from MAST. All 75 passed archive
  SHA-256, data-inclusive FITS checksums, target/sector checks and TransitEye ingestion. Catalog
  snapshot, heuristic limitations, provenance, scripts and list are in external workspace
  `data/confirmed_transiting_hosts_v1/`. This is a user inspection set, not a newly frozen independent
  evaluation or a classifier label set; these sources may overlap previously opened datasets.

- [x] R3 Durable batch and desktop workflows: supervised crash recovery and retry, visible failures,
  session restore and reporting, and shared desktop/batch settings. The user confirmed the Linux
  session, export, batch, pause/resume, failed-target retry and visible failure workflow. A real
  TOI-1437 FITS run gave identical settings, provenance, candidates, classifications, intervals,
  missing reasons, seeds, errors, code/model fingerprints and device in CLI and desktop bundles:
  `.transiteye-workspace/remediation/r3-cli-final` and
  `.transiteye-workspace/remediation/r3-desktop-smoke/smoke.teye.json`. The full headless suite
  passed 132 tests; locked sync, desktop sync, Ruff, CLI help and diff check passed. A previous
  intermittent Qt teardown crash did not recur but is not conclusively explained. The candidate
  is a weak two-event alias, not a confirmed recovery.
  A subsequent clean wheel reproduced the Qt/PyQtGraph teardown crash. The detached-view fix passed
  three clean-wheel desktop suites, native X11 checks, and all 165 clean-wheel tests. The user then
  confirmed that analysis, session restore, export, batch pause/resume, and failed-target retry all
  worked in a repeated manual walkthrough on the current build; the R3 gate is restored.
- [x] R1 Scientific integrity guardrails: disabled the legacy locked evaluator and label-conditioned
  transforms; pilot training uses unmodified real features and rejects augmentation; analysis no
  longer auto-trains/selects a classifier or reports unvalidated fixed-phase significance. Explicit
  trusted model loading checks estimator and calibrator integrity, and missing classifier outputs
  retain source assessment without inventing probabilities. Historical artifacts remain intact.
  Verification: `uv sync --locked`, desktop extra sync, Ruff, CLI help, and 90 headless tests passed;
  an explicitly selected real-catalog v1 bundle analyzed one development-partition FITS on CPU and
  produced a result bundle at `.transiteye-workspace/remediation/r1-real-development`. No locked
  evaluation was opened. This is a smoke test, not a model-performance or parameter-accuracy result.
- [x] R2 Shared scientific configuration and models: typed settings, explicit trusted classifier,
  v3 provenance and integrity, input/setting/code/model-aware cache, FITS flux/frame provenance,
  explicit exposure propagation, exact trapezoid integration, short-data and per-candidate failures.
  Training and inference feature extraction agree on a controlled signal with stellar temperature.
  Gate: Ruff and `git diff --check` passed; 108 headless tests passed; CLI and batch produced identical
  candidates, classifications, intervals, settings, seeds, device and fingerprints on a real
  development-partition FITS file with a frozen model and explicit exposure. Both bundle manifests
  verified. Physical-model quadrature convergence, interval coverage and independent model validation
  remain R4 scientific gates; no locked test or EB holdout was opened.
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

- [x] User-requested 50,000-target MAST expansion: acquired 50,000 distinct target light curves
  through the MAST Portal Download API from the frozen Sector 1–13 inventory. The inventory had
  126,994 new source IDs after excluding 5,403 known project identities. Manifest
  `d1f546b027534bb7` fixes the selection, with every label unset. On 2026-11-11 14:19 IST, all
  50,000 products passed FITS structure, TIC/sector, and light-curve column checks; all 50,000 have
  recorded local SHA-256 values, and the report lists zero failed products. The corpus occupies
  95,773,392,000 bytes outside Git. FITS checksum audit: PRIMARY CHECKSUM valid; LIGHTCURVE and
  APERTURE CHECKSUM invalid; DATASUM absent in all three HDUs. These archive-header statuses are
  recorded per product. The corpus remains unlabeled and is not a supervised training set or fresh
  evaluation cohort. A follow-up TOI label overlay is tracked below; the frozen product manifest
  remains unchanged.

## In Progress

- [~] Lookup acquisition integrity follow-up: persist verified per-product SHA-256 receipts
  outside the frozen discovery manifest, require receipt checks on cache reuse and inspection,
  and preserve manifest identity on repeated fetch. This is an additive workspace artifact with
  no change to legacy result bundles or v1 manifest reading. Mocked corruption/retry tests and
  a bounded live TIC/KIC acquisition smoke are required before closing the first lookup gate.
  The same gate now checks TESS-SPOC discovery from MAST's HLSP collection and rejects missing
  or conflicting FITS TIME units instead of assuming days.

- [~] TIC/KIC lookup extension: implement mission-qualified archive discovery, verified
  acquisition, and Kepler FITS ingestion as the first reviewable slice. Combined analysis,
  editable transforms, desktop integration, and model comparison follow only after this
  ingestion slice is verified. Qualified lookup and fetch CLI commands, a target-checking cache
  validator, and a Kepler FITS reader are implemented. Exact MAST searches returned 47 TESS
  light-curve products for TIC 307210830 and official Kepler products for KIC 11904151; these
  were discovery-only checks, not downloaded analyses. Focused lookup tests passed (9 tests),
  the full headless suite passed (198 tests), locked sync with the desktop extra passed, and
  Ruff, CLI help, and diff checks passed. Combined analysis, desktop lookup, and model training
  remain open. No new scientific performance claim is made.

- [~] Combined-observation input foundation: `ObservationSet` retains each verified FITS curve,
  input checksum and archive URI, maps native TIME to BJD-TDB only with explicit time metadata,
  and rejects mixed targets or missions, duplicate files, missing products and mission-mismatched
  FITS. It retains provider and cadence provenance and exposes overlap/analysis masks, overlap
  winners, and per-cadence reasons; the Kepler quality mask is mission-specific while the existing
  TESS mask remains unchanged. `inspect-target-set` reports per-observation cadence, overlap-winner,
  selected/excluded and reason counts plus BJD-TDB coverage. BLS has an explicit optional maximum
  trial-period budget. A standalone joint trapezoid fitter shares timing/shape and fits depth and
  baseline separately per observation. Focused lookup/FITS/observation/BLS/joint-fit tests passed
  (35 tests). These pieces are not wired into end-to-end combined analysis or the UI; flux merging,
  integrated detection, resumable combined jobs, and user-facing review remain open. No model
  training or performance claim is made. The phase gate has not passed.
  Cadence width is a sampling-window proxy, not an exposure measurement. With at least three distinct
  timestamps it uses the lower median of positive spacings; with two, TIMEDEL is accepted only within
  20% of measured spacing; singleton TIMEDEL is unknown. Unknown widths in intersecting product spans
  remain unresolved.
  The separate exposure resolver uses a user override or `INT_TIME*NUM_FRM` as a frame-sum boxcar
  approximation, reports missing when unavailable, and ignores `TIMEDEL` and aggregate
  `EXPOSURE`/`LIVETIME`; it is not integrated into combined analysis. Its focused tests passed
  (7 tests). The 35 passing focused tests above apply to the preceding observation/BLS/joint-fit
  slice.
  `prepare_observation_set` independently normalizes and detrends each observation, retains original
  cadence-row mapping and common BJD-TDB origin, and rejects unresolved overlap. Its focused tests
  passed (6 tests). This preparation does not run a search or joint fit; those components remain
  unwired to an end-to-end combined service.
  New lookup manifests now use v2 to retain the MAST provider; v1 remains readable without
  rewriting its existing files or IDs. The v1/v2 round trip and focused lookup tests passed;
  the full headless suite passed (208 tests), with Ruff, CLI help, and diff checks passing.

- [~] User-requested confirmed-exoplanet TESS data: NASA Exoplanet Archive `pscomppars` currently
  returns 6,464 confirmed planets across 4,857 host names; 6,157 planet rows map to 4,561 distinct
  TIC IDs. Across 107 MAST SPOC sector inventories, 4,242 hosts have 22,880 available products;
  319 TIC hosts have no matching product. Frozen manifest `a5d1e030a4c424d5`. At 2026-11-11
  15:54 IST, 850 products were on disk and the latest acquisition report checkpoint recorded 750
  successful downloads with no failures. Download every matched sector light curve. This is a
  confirmed-host cohort, not a light-curve morphology label set; products must stay source-grouped.

- [x] User-requested TOI label cross-reference: exact-joined the current ExoFOP-TESS TOI CSV (8,175
  rows; file SHA-256 and source URL recorded in the report) to the frozen 50,000-target corpus by
  TIC ID and required the TOI sector to include the product sector. Found 139 matching products and
  157 TOI events. Added 111 `transit_like` catalog proxy labels where TESS and TFOPWG dispositions
  both say `PC`; kept 28 products with ambiguous/conflicting dispositions and 49,861 products with
  no TOI match unlabeled. All raw same-sector event dispositions are retained. The 50,000-row label
  overlay is stored separately as Parquet, with 157 event-level matches in JSONL and a provenance
  report in the external workspace. The frozen product manifest and model inputs are unchanged;
  catalog labels are proxies, not independently adjudicated morphology.

- [x] User-requested further data expansion: queried 2,000 Gaia variability candidates and
  cross-matched a bounded 500-source subset; 29 qualified TESS light curves were downloaded and
  verified. Trained the three-class research model on 455 unique source groups, including 24 new
  variability examples. Five new held-out variability proxies were predicted as variable; this
  single-class check cannot establish overall performance or source localization. Prior test
  groups remain excluded from training; the model is uncalibrated and not a desktop ModelBundle.
  See `docs/variability-expansion-20261102.md` and external `verification.json`.

- [x] User-requested larger research cohort: froze and downloaded 100 additional catalog-proxy
  TESS files; all passed FITS and target checks. Trained an uncalibrated three-class model on 431
  source-separated vectors (307 prior, 41 previous expansion, 83 new); four new training matches
  failed. The 13-source test had one catalog mismatch and 12/12 proxy-label agreement among the
  usable cases, including only one variable source. SHA-256, source grouping and saved predictions
  were independently checked. `docs/three-class-larger-20261102.md` records limitations; the
  research artifact does not replace the four-class desktop model or satisfy the R4 gate.

- [x] User-requested data expansion: froze 60 additional catalog-proxy sources, downloaded and
  verified 60 TESS FITS files, and trained the three-class research estimator on 307 previous and
  41 new usable training sources (one new training catalog mismatch was retained and excluded).
  A source-disjoint 18-source development test, opened after model locking, had 18 usable and
  matched predictions; this small selected set does not validate real-sky performance. Verified
  checksums, source grouping, finite features and reproduced predictions; see
  `docs/three-class-expansion-20261102.md`. Unadjudicated fourth-class sources remain unlabeled,
  the application model is unchanged, and the separate sealed EB holdout remains closed.

- [x] Added an accessible desktop graph guide for presenting the existing analysis, describing
  plotted data, fit curves, period previews, and scientific limitations without altering results.
  Verified the guide opens and 30 headless desktop tests pass; no new manual walkthrough claimed.

- [x] Corrected the desktop demo to select the historical classifier when the sample is explicitly
  launched, checking all three files against the frozen protocol and retaining a visible experimental
  indication and normal abstention behavior. The actual `--demo` startup completed offscreen with
  the selected historical model and a transit-like result for TIC 198356533. Missing or modified
  model files stop the sample with an error. All 29 headless desktop tests, Ruff and diff checks
  passed. No new manual Linux check or independent model validation is claimed.

- [x] Desktop morphology access: exposed explicit opt-in to the existing historical four-class
  bundle for single-target and new batch runs, passing its frozen calibrator checksum to the same
  backend as the CLI. Explain missing models and the model's unvalidated current-pipeline status;
  preserve abstentions and do not substitute the incompatible three-class research estimator.
  Verified 28 headless desktop tests, Ruff, diff check, CLI help, and an offscreen real TOI-1437
  analysis: the historical model produced transit-like with no abstention and 0.4182 top score.
  Its output is experimental and historical performance does not validate the current pipeline.
  No new manual Linux walkthrough or independent classifier evaluation is claimed.

- [x] Desktop sample workflow: added a keyboard shortcut and command-line option using a real
  workspace sample and the normal backend; the status bar identifies it as sample data. The default
  FITS checksum is verified; alternate local inputs and errors are supported. No scientific behavior,
  result schema or classifier changed. Verification: 26 headless desktop tests, Ruff and diff check
  passed; a real offscreen backend run searched 1.00–19.99 days and produced one candidate for
  TIC 198356533. This is an automated desktop run, not a new manual Linux walkthrough.




- [x] User-requested three-class development training: froze 151 source-grouped entries after
  excluding 70 pre-existing non-training assignments and four conflicting collection entries;
  13 candidates failed catalog matching. Trained a standalone uncalibrated 200-tree random forest
  on 129 usable sources (30 transit, 56 eclipse, 43 variable) and recorded grouped development CV.
  Nine remaining validation examples are all eclipse-like and do not support multiclass evaluation.
  SHA, features, grouping and artifact reload verified; `docs/three-class-research-model-v1.md`
  records results and limitations. Existing split identities were checked but no sealed holdout
  observations or labels were opened. The fourth class is absent and the artifact is not a compatible
  application ModelBundle; no independent performance claim or desktop change is made.

- [x] Physical-model contact diagnostic: the frozen 21-case development grid compared Batman
  against independent annular intensity integration. All 18 cases without native warnings passed
  the 0.01 ppm criterion (maximum 0.001032 ppm); all three exact-contact cases emitted native
  convergence warnings and remain unusable. The nonzero exit, all rows and diagnostics, versions,
  hashes and source snapshot are preserved outside Git. Twenty-eight focused tests and the full
  168-test headless suite passed; Ruff and diff checks passed. See
  `docs/physical-contact-diagnostic-v1.md`. Production fitting and partitions are unchanged; adding
  the evaluation module changes the backend fingerprint for newly generated results.
- [x] Physical-fit sampling pilot: eight frozen synthetic cases compared production five-point
  and development-only 51-point Simpson fits on identical data. All 16 fits completed with no
  native warning and truth-reference differences below 0.1 ppm. The 30-minute fits differed
  substantially in some cases; one noise realization per setting cannot establish general
  performance or interval coverage. All rows, configuration and source hashes are in the external
  workspace; results and limitations are in `docs/physical-fit-sampling-pilot-v1.md`. Optimizer
  parity and worker failure tests passed; production fitting and partitions remain unchanged.
  The added evaluation module changes the backend fingerprint for subsequently generated results.
- [x] Repeated physical-fit sampling study: 48 frozen synthetic cases at two noise levels and
  three draws each produced 96 paired fit records. Eight production and two experimental fits
  returned failure, all at 30-minute exposures; no native warnings or failed truth checks occurred.
  Paired usable counts and error medians, all failures, seeds, hashes and limitations are retained
  in `docs/physical-fit-repeated-v1.md` and the external workspace. This study changes no production
  fitting and does not establish uncertainty coverage. Worker failure, malformed output and timeout
  handling are tested; 33 focused tests and Ruff passed. Added evaluation code changes later
  backend fingerprints.
- [x] Synthetic uncertainty and recovery study: 60 frozen white-noise injections plus 20 noise-only
  controls, requesting 100 bootstrap refits per candidate, completed and were reconciled against 80
  unique integrity-verified bundles. Recovery was 42/60; 11 candidate fits and two optional physical
  fits failed; null controls produced 0/20 candidates. Conditional 68% coverage for period, depth
  and duration was 27/42, 25/42 and 28/42; 95% coverage was 37/42, 34/42 and 38/42. Full counts,
  matching rule, errors, hashes and limitations: `docs/injection-uncertainty-development-v1.md`.
  Same-environment resume passed. Results are exploratory, not real-data calibration. The current
  bootstrap silently skips failed refits and does not record how many succeeded; 100 requested
  refits is not evidence that 100 succeeded. This is an unresolved scientific reporting defect.
- [~] R4 Independent validation and release: begin with a reproducible development-only physical
  model exposure-sampling convergence study, then investigate the recorded fit failure and measure
  parameter accuracy and interval coverage. No held-out source data are needed for this first study;
  the separate EB holdout remains sealed.
  - [x] New experimental PyTorch sequence classifier (approved 2026-11-10): the user selected a
    from-scratch PyTorch candidate-sequence model, independent of historical classifier weights.
    Frozen the sequence and baseline protocols in `configs/sequence_model_development.v1.toml` and
    `configs/model_baseline_development.v1.toml`. Both exclude every test partition; baseline uses
    source-grouped 5-fold CV over train, with separate calibration and selection sources, no
    synthetic augmentation, and no label-conditioned transforms. Corrected trainer metadata so a
    fresh fit is not mislabeled as retrained from the historical model. The 986-product development
    set has now passed SHA-256 verification (986/986, zero missing or mismatched files); current
    acquisition report has 986 successes and zero failures. The first serial baseline run was
    stopped after 200 sources because it was too slow; no artifacts were written. The trainer now
    uses four spawned CPU workers in stable input order with per-source checksum checks and explicit
    error records. The frozen baseline config records CPU execution and four workers. The restarted
    fresh baseline completed: 771 usable train, 104 calibration, and 108 selection sources; three
    feature failures were recorded. Random Forest was selected at source-grouped 5-fold CV
    macro-F1 0.698 / transit-like recall 0.571, versus HistGradientBoosting 0.673 / 0.571 and
    logistic regression 0.614 / 0.589. These are development model-selection results on catalog
    proxies, not independent performance estimates. Evidence is in the external workspace
    `models/morphology_fresh_baseline_v1/`; no test FITS were opened.
    Implemented candidate-centered sequence construction and an isolated PyTorch training command
    in `scripts/train_sequence_model.py`; it excludes test FITS before reading, uses fresh
    initialization, calibration-only temperature scaling, and selection-only epoch choice. The
    v1 and v2 runs exposed unsupported deterministic CUDA pooling gradients and produced no model
    artifacts; both protocols are preserved. V3 uses explicit reshape-and-mean pooling and writes
    a checksummed sequence cache. Its fresh CNN trained successfully at epoch 6, calibrated with
    temperature 1.349, and recorded selection accuracy 0.630 / macro-F1 0.608. There were 986
    usable sequences, all four classes remained represented in train/calibration/selection, and
    extraction had zero failures. Reloading the weights reproduced the saved selection metrics
    exactly. These selection metrics are model-selection evidence, not independent performance.
    The artifact and report are in external workspace
    `models/candidate_sequence_cnn_v3/`; no test FITS were opened. This remains research-only because
    labels are catalog proxies, not candidate-local ground truth. `uv sync --locked --all-extras`
    succeeded; PyTorch 2.15.0+cu132 and CuPy 14.2.0 both detect the RTX 3050, and a CUDA matrix
    multiplication succeeded. `CandidateSequenceCNN` in
    `src/transiteye/classification/sequence_model.py` is isolated and not imported by the default
    backend or desktop.
  - Fit change control: a historical failure from the already-open v1 test source is reproducible
    with initial depth 764,878 ppm versus the 500,000 ppm optimizer limit. Extend the physically
    allowable normalized depth to 1.0 and explicitly reject impossible initial depths. This source
    is a debugging case only, not an independent evaluation. New fit features can change; old
    bundles remain readable, historical classifier metrics cannot validate the changed pipeline,
    and taxonomy, splits, and result schema are unchanged.
    Implemented and verified: a noiseless 760,000 ppm synthetic eclipse is recovered, impossible
    initial depths are rejected, and replaying the previously opened source now extracts a finite
    feature vector, with optimizer convergence in 31 evaluations. The full headless suite passed
    136 tests after the fit change; this debugging replay is not evidence of model accuracy or
    independent performance.
  - Optimizer change control: non-converged least-squares runs must not be silently treated as
    successful fits. New trapezoid fits will fail explicitly through existing candidate-error
    fields; failed optional Batman refinements will be unavailable. This can alter new features,
    but does not rewrite bundles or change schemas or splits, and historical scores do not validate
    this behavior.
    Implemented; the previously opened debugging case still converges, and 34 focused fit,
    uncertainty, and exposure-study tests passed, including explicit optimizer nonconvergence.
  - Result-flag change control: mark new candidates carrying the five-sample Batman model with
    `physical_quadrature_unvalidated` in the existing JSON/Parquet string-list flags. This additive
    flag leaves schema, taxonomy, splits, and historical bundles unchanged. Trapezoid parameters
    and uncertainty coverage remain independently unvalidated.
    Implemented and verified in JSON, Parquet, bundle reload, and a headless desktop check; the
    desktop also explains the flag in plain language. Thirty related fit/reporting/desktop tests
    passed, followed by the two focused UI/bundle checks after adding the explanation.
  - Physical-fit failure reporting change control: when the optional physical fit was actually
    attempted and returns no model, add `physical_fit_failed` plus an existing `fit_failed`
    missing reason; intentional skips do not carry this failure. This is additive in the existing
    v3 fields, leaves old bundles and splits intact, and does not invent an exception cause.
    Implemented and verified through both attempted/failed and screened/skipped runs, Parquet
    persistence and the desktop explanation. Three focused tests, Ruff and diff check passed.
  - Frozen development grid `configs/exposure_convergence.v1.toml` compares Batman exposure
    supersampling with trapezoidal integration of the same instantaneous model; see
    `docs/exposure-convergence-v1.md`. On the 9 usable geometries, five samples failed the 1 ppm
    criterion in all cases, with maximum 601.89 ppm; even 101 samples passed only 5/9. Three
    grazing geometries emitted native convergence warnings and are explicitly unusable. Results,
    code/config hashes and diagnostics are in external workspace `remediation/r4-exposure-v1-final-current`.
    The nonzero study exit records the failed criterion. After adding incomplete-worker and native
    diagnostic checks, 23 focused characterization/convergence tests, Ruff and diff check
    passed. This is a numerical warning, not parameter-accuracy or uncertainty validation; no
    production sampling setting has been changed.
  - Next slice: freeze and run a separate Simpson-quadrature comparison including geometries on
    either side of the exact warning-producing contact. Exploratory calculations suggested improved
    convergence but are not validation; the new frozen study must retain and report native warnings.
    Frozen configuration: `configs/exposure_simpson.v1.toml`. This is a numerical development
    study, not an evaluation protocol or a change to the production model.
    Frozen run complete: 51-point Simpson integration meets the 1 ppm criterion on 21/21 usable
    geometries (maximum 0.136 ppm); three exact-contact geometries produce native warnings and are
    unusable. The command correctly exits nonzero, with all 120 rows retained. Evidence and hashes:
    external workspace `remediation/r4-exposure-simpson-v1`; procedure and limits:
    `docs/exposure-simpson-v1.md`. An exploratory model timing was roughly ten times slower than
    the current method; it is not the required 100-target benchmark. Twenty-four focused tests,
    Ruff and diff check passed. Production integration remains unchanged and unvalidated.
  - In progress: frozen synthetic-only injection pilot `configs/injection_pilot.v1.toml`, including
    separate noise-only controls, exact-truth matching and conditional interval coverage. This small
    white-noise pilot is not the real-substrate study or independent real-data evaluation required
    for R4; the EB holdout remains sealed.
    The pilot recovered 12/20 injected signals, observed 0/10 alerts on pure white-noise controls,
    and recorded three optimizer nonconvergences. Period/depth/duration errors and conditional
    interval coverage, per-stratum counts, caveats and reproduction instructions are in
    `docs/injection-pilot-v1.md`; all trial bundles and checksums are in external workspace
    `remediation/r4-injection-pilot-v1-corrected`. The first run's table export exposed an unsigned
    seed overflow; after a tested fix, the new run completed and all 30 checkpoints matched the first
    run in the same environment. A resume replay reused the checkpoints. Six focused tests passed.
    This small pilot does not complete the broader injection-recovery or real-data evaluation gates.
  - Next development slice: `configs/real_substrate_injection_pilot.v1.toml` freezes selection of
    four train-partition source groups before inspecting their light curves, including a recorded
    two-minute exposure assumption. Compare injected trials separately from unchanged controls;
    controls on real stars cannot be assumed signal-free and do not estimate a false-alert rate.
    The four selected TRAIN FITS files passed their manifest checksums. The corrected pilot completed
    16 injections plus four controls: 4/16 top-candidate matches, candidates in all four controls,
    and no reported trial/fit failures. All 20 bundles verified, corrected trial records matched the
    earlier run, and a same-environment resume reused the checkpoints. Synthetic provenance now
    distinguishes injected-array checksums from parent FITS paths and explicitly enforces TRAIN.
    Results, conditional measurement/coverage counts and limitations are in
    `docs/real-substrate-injection-pilot-v1.md`; artifacts and frozen selection are in
    external workspace `remediation/r4-real-injection-pilot-v1-final`. Nine focused tests and
    Ruff passed. Four correlated source groups are not sufficient for a performance claim.
    Provenance correction completed: the first run passed its assumed two-minute exposure as an
    explicit value, causing v3 to say `user_supplied`. The final run uses the existing
    `fixed_assumption_unverified` fallback; all 20 scientific trial records matched the prior run,
    and all final bundles passed integrity and provenance checks. Earlier artifacts are unchanged.
    Matching source snapshots for both injection pilots and archive SHA-256 records are saved in
    external workspace `remediation/code-snapshots`, alongside the frozen study inputs and outputs.
  - Benchmark audit: the legacy `run_sector_benchmark` uses coordinator-only `RUSAGE_SELF` for peak
    memory and divides total targets by elapsed time, including failures or previously completed
    tasks. Its output cannot establish whole-job memory or fresh-work throughput. A replacement
    must measure the worker process tree and distinguish completed, failed and replayed work before
    the frozen 100-target run; historical benchmark figures are not current evidence.
    Change control: preserve existing report field names but make new `peak_rss_mb` sampled summed
    process-tree RSS and throughput newly completed targets per elapsed second; record cache hits,
    pre-existing work, sampling limits, full settings and pause/resume evidence in `details`.
    Existing benchmark reports are not rewritten. RSS sums can double-count shared pages and polling
    can miss brief peaks; neither number is an exact unique-memory high-water mark.
    Completed frozen development run: 100/100 targets, zero target failures or cache hits,
    193.33 seconds and 1,491.6 MiB sampled peak summed RSS with four CPU workers. A controlled
    pause drained to 13 completed and resumed; all 100 unique task records had one attempt and all
    bundles passed integrity, input and code checks. The independent flag audit found no fit failures
    and 82 unvalidated-physical-model flags. `docs/development-benchmark-v1.md` records the precise
    settings and limitations; report, preflight, selection, verification and matching source snapshot
    are in the configured external workspace. The benchmark is operational evidence, not a model
    or detection-accuracy evaluation. The separate release/reproduction gate remains open.
  - Development label audit (v1 TRAIN only, no sealed holdout opened): 597 labels include 150
    eclipse-like deep-false-positive proxies and 142 ATL-derived variability proxies. The legacy
    builder assigns `verified`/`reviewed` strings automatically; those strings do not establish
    independent human morphology review. The ATL is a list of targets with predicted oscillation
    detectability, not a list of measured variable signals. Legacy builder code also fabricates
    review-placeholder DRN identifiers; none occurred in this 597-source TRAIN subset, and 37 such
    v2 placeholders were excluded from v1. Preserve old manifests; disable the legacy builder entry
    points and historical real-cohort evaluator so those paths cannot create new unsupported labels
    or rerun an already-open evaluation. This is a compatibility change, not a taxonomy or split
    change; audited replacements and a frozen independent evaluation are still required.
    The guards are implemented and tested before any input read. A checksummed, TRAIN-only audit
    tool reproduces the counts and rejects other manifest paths; the detailed audit and primary
    catalog links are in `docs/label-audit-development-v1.md`. Nine focused guardrail, audit and
    benchmark tests passed, along with Ruff. Pre-change historical scripts are archived with hashes
    outside Git. This is a development audit, not independent label adjudication.
  - In progress: freeze `configs/search_null_pilot.v1.toml` for an experimental full primary-BLS
    search permutation test on synthetic white noise and a correlated-noise diagnostic. This will
    test an IID/exchangeability assumption only; it does not validate TESS red noise, detrending,
    iterative search, or any production significance field, which remains unavailable.
    Completed synthetic diagnostic: 5/100 IID alerts at p ≤ 0.05 met the predeclared white-noise
    checks, while applying the same IID shuffle to correlated AR(1) noise produced 40/40 alerts.
    All 140 trials completed without errors. `docs/search-null-pilot-v1.md` records the fixed grid,
    99-permutation resolution, assumptions and limits; artifacts and the matching source snapshot
    are in the external workspace. Four focused tests passed. This failure under correlation means
    the production significance field remains unavailable and the full R4 significance gate open.
  - Clean-environment check: the installed wheel passed 165 tests and reproduced all 20 fixed
    real-substrate trial records and the aggregate summary exactly; every reproduced bundle passed
    integrity checks. Three offscreen desktop suites and native X11 close/session checks passed.
    The user confirmed the complete manual workflow on the current checkout. Details and wheel/lock
    hashes are in `docs/clean-environment-check-v1.md`. This is a separate environment on the same
    host, not an independent person or machine, and does not complete the release gate.
- [x] Scientific integrity follow-up: review source-vetting missing-data behavior. Absence of
  centroid/crowding measurements and zero-signal difference images currently risk false reassurance.
  Correct those cases to unresolved/not assessed without rewriting prior bundles; retain nullable
  v3 artifact fields and record the changed behavior through the code fingerprint.
  - New analyses now return unresolved for missing centroid/crowding or a zero-signal difference
    image, and leave the unsupported SAP/PDCSAP ratio unavailable. Ten focused tests covering
    vetting, process recovery, and the desktop search controls passed. Missing reasons are saved
    in the result bundle. The full headless suite passed (121 tests). Old bundles are unchanged.
- [x] R3 Durable batch and desktop workflows: replace unbounded queued execution and ineffective
  timeouts with supervised, bounded workers; implement persisted job controls and accurate target
  accounting, then connect the controls to the desktop. Preserve existing SQLite jobs and v2/v3
  bundle readability. Complete headless and manual Linux verification before marking the phase done.
  - 2026-11-01: the user confirmed the session now restores, and separately confirmed that vetting
    sheet export, a batch run, batch control (pause/resume or retry), and visible failure reporting
    worked manually. On follow-up, the user confirmed that both pause followed by resume and retry
    of a failed target worked. The automated suite separately covers these paths.
  - Current slice: prevent a batch start from raising an unhandled exception when another backend
    operation is already running. The desktop disables batch start during another analysis, guards
    keyboard/programmatic starts, and restores controls afterwards. A QProcess-backed desktop test
    also ran an invalid FITS job, retried it, and verified the failed target and error remained
    visible with two attempts. Both focused tests passed.
  - Current slice: the inspector used SNR 7.1 while desktop batch used 6.0 and ignored the selected
    period range. Align new desktop runs to the backend's 6.0 default and pass the inspector range to
    new batch jobs. This changes desktop defaults only; old result bundles and saved jobs are intact.
    With TOI-1437 sector 86, actual Linux desktop single and batch runs with 1–20 day bounds and
    SNR 6.0 produced identical candidates, classifications, intervals, missing reasons, seeds, code
    and model fingerprints, and device. Evidence: external workspace job
    `batch_bbdc4d0934cc465db0c140a8b8484900` and `remediation/r3-desktop-smoke/smoke.teye.json`.
    Periodogram clicks now display a fold-only preview without reusing the original fit or presenting
    residuals as if the selected period had been refitted. Focused desktop tests passed. Three
    consecutive headless desktop-suite runs passed (17 tests each); the earlier Qt teardown crash
    has not been reproduced and is not considered conclusively resolved. The full headless suite
    passed with 130 tests; Ruff and `git diff --check` passed.
    `docs/manual-linux-smoke.md` now records the remaining hands-on workflow, including session
    restore, non-blocking export, batch pause/resume, and explicit failed-target accounting. It is
    a procedure to execute, not evidence that the manual gate has passed.
  - 2026-11-01 continuation: expose existing backend period-search options in the desktop so users
    can actually run a wider search; retain existing defaults and scientific settings. A real
    SIGKILL integration test confirmed that a worker observes coordinator death, releases its lease,
    and a restarted job completes exactly once. The lease prevents overlapping resumed workers.
    The visible Linux Qt application was exercised automatically using the actual TOI-1437 FITS:
    wider-range analysis produced the ambiguous 9.426-day alias, a session reopened with settings,
    a vetting sheet exported, and a one-target batch completed. This is automated interaction,
    not the required manual Linux walkthrough. Focused execution/desktop tests passed (13 tests).
    The inspector now displays candidate flags and reported aliases so a weak two-event result is
    not presented without its ambiguity.
    The user initially reported no candidate with the default 0.30–13.28 day range. A subsequent
    result from the fresh window records the selected 1–20 day range and one weak two-event
    candidate at 9.426 days, with unresolved source assessment. At that checkpoint the manual gate
    remained open: the user reported a save/reopen session failure, while automated save/reopen
    succeeded. The desktop now reports a missing saved bundle and save errors explicitly, confirms
    the saved path, rejects malformed bundles, and clears prior results when opening a session
    without a bundle. On 2026-11-01 the user subsequently confirmed that the session now restores;
    the precise original failure was not independently reproduced. GUI vetting-sheet export is now
    asynchronous, and missing source values render as
    unavailable in exported figures.
    A full run aborted in a PyQtGraph/Qt teardown callback; the desktop suite and subsequent full
    runs passed, most recently 128 tests. This intermittent Qt failure remains under observation.
    An empty batch source now fails rather than reporting success.
  - 2026-11-01: added bounded spawned workers, individually enforced deadlines, per-job coordinator
    locks, idempotent counters, checksummed manifest-cache lookup, and pause/cancel/retry/resume CLI
    controls. The desktop displays actual completion/failure totals and sends controls asynchronously.
    Candidate-level errors remain visible in persisted task records. Initial focused tests cover
    timeouts, checksum mismatch, retry, pause and resumption. Saved target rows and errors can be
    reloaded in bounded pages. Focused scheduler, desktop, and CLI checks passed (24 tests); an
    automated native Linux display startup also exited successfully. The full headless suite passed
    (116 tests), as did Ruff and `git diff --check`. This is not a manual workflow smoke test.
    At that checkpoint the manual Linux workflow smoke test and investigation of orphaned workers
    remained open; see the continuation above for the subsequent crash-recovery work.
  - 2026-11-01 user-reported TOI-1437 sector 86 case: its default search covered periods only to
    13.28 days and durations only to 1.8 hours; the published signal has period about 18.84 days and
    duration several hours. A development-only exploratory run with a 1–20 day period range and
    10 bootstrap replicates returned a weak two-event candidate at 9.426 days, approximately a
    half-period alias, not an independent recovery. The desktop now states its searched period
    range, renders missing class probabilities as unavailable, and places logarithmic periodogram
    markers correctly. The default duration-grid limitation requires separate scientific remediation
    and validation; search defaults have not been changed on this example.

## R2 implementation notes

- [x] R2 Shared scientific configuration and models: preserve explicit model identity, configuration,
  workspace, and input/code provenance across individual and batch runs; verify cache identity and
  feature-path equivalence before expanding into exposure-aware fitting. Existing schema-v2 bundles
  remain readable; any incompatible output changes require a new schema.
  - 2026-11-01 slice: batch workers now receive the selected workspace and explicit model; cache
    keys include full job configuration, actual model bytes, backend source fingerprint and device;
    cached bundles are read and checked against input checksum, schema, status and candidate count.
    A corrupt bundle is recomputed and zero-candidate cache hits remain zero. Analysis IDs use UUIDs
    to avoid collisions. Ten focused execution/CLI tests passed. One development-partition FITS was
    analyzed through the CPU CLI and batch using the same selected model; candidates, classifications,
    and intervals were identical. Both results are smoke tests, not validation claims.
  - Batch scheduling and recovery remain R3 work; scientific coverage and independent model evaluation
    remain R4 work.
  - Approved additive schema-v3 change: record effective runtime options, package/source identity,
    and classifier identity in new results. Existing schema-v2 bundles remain readable; no historical
    artifact or frozen evaluation protocol is modified. New v3 bundles also include an integrity
    manifest for the three components; old bundles without one remain readable but are not cache hits.
    The selected FITS flux product is also recorded, with matching single-target and batch options.
  - 2026-11-01 continuation: individual and batch services now share immutable `AnalysisSettings`;
    v3 records settings and code/model hashes and marks the existing two-minute exposure assumption
    unverified. Inputs below the detector's cadence or time-baseline requirements now report
    `insufficient_data`. A real development-partition FITS run with the same model produced identical
    CLI and batch candidate, classification, uncertainty, settings, and provenance fields; both v3
    bundle integrity manifests verified. This does not validate physical accuracy.
  - Exposure integration compatibility: accept explicit per-cadence duration, preserve raw FITS frame
    headers, and propagate the chosen duration to fitting/bootstrap. Missing metadata does not imply
    exposure; the historic two-minute fallback stays visibly marked as an unverified assumption.
    Physical-model integration changes the deployed feature path, so the historical v1 evaluation
    cannot be represented as an independent validation of the updated pipeline.
    A numerical check found that the five-sample trapezoid can be inaccurate at long exposure;
    its remediation is an exact boxcar integral with independently checked limiting cases. Batman
    retains five integration samples; its convergence and bootstrap coverage remain unverified.
  - Candidate failure schema decision: v3 `candidate_errors` records per-candidate fit exceptions;
    failed fits retain detector estimates with `fit_failed` and no model/physical classification.
    Multi-candidate regression verifies one failed fit cannot discard another candidate, and the
    incomplete result is not reused through the batch cache.
  - Configuration decision: expose and record quality, gap, BLS period/duration grid, detrending,
    screening and bootstrap settings through the shared immutable options; defaults preserve the
    existing behavior and cache identities include explicitly chosen values.
    Explicit bounds are enforced after period-grid construction because Astropy can extend a grid
    by one step beyond the requested maximum.
    Requested and actual compute devices are recorded separately to preserve `auto` semantics and
    make batch CPU spillover visible. Candidate-specific bootstrap seeds are recorded in v3 output.
  - R1 follow-up: the old fixed-phase statistic remained callable as the public
    `compute_empirical_significance` helper. Disable that API explicitly until a genuine search-wide
    surrogate calculation is implemented; compatibility impact is a deliberate runtime error.
- [ ] Freeze and execute one independent evaluation for `morphology_real_catalog_v3` after auditing
  source-local EB label validity. Do not reuse the previously opened v1 test sources; the new EB
  holdout remains sealed.

## Backlog

See the working-product remediation sequence in `PLAN.md`. Independent visual review of
catalog-derived labels, validated exposure integration and the larger evaluation remain future work.

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
| 2026-10-25 | Added Claret (2018) limb-darkening interpolation, multi-sector stitching, arcsec vetting offsets, worker timeout, desktop interactive peak picking & overlays, CI/CD, and runnable examples | `./scripts/verify_clean_release.sh` passes 84/84 tests | Passed |
| 2026-10-26 | Expanded dataset to 1,000 target IDs (cohort_v2), trained morphology_v2, and ran its benchmark | The cohort builder synthesizes FITS signals for missing products; the reported score is not a real-data generalization estimate | Revisit |
| 2026-10-26 | Scaled dataset to 5,000 target IDs (cohort_v3), trained morphology_v3, and ran its benchmark | The cohort builder writes synthetic class-shaped signals; the current evaluation also applies true-label feature modifications, so reported scores are invalid for model validation | Revisit |
| 2026-10-29 | Built cohort_v3 with 5,000 catalogue IDs and synthetic class-shaped FITS signals; trained morphology_v3 and ran its locked benchmark | Code/test artifacts exist, but evaluation leaks test labels into feature vectors; scores are not valid estimates of real-data model performance | Revisit |
| 2026-10-29 | Acquired a real-data catalog cohort from MAST without synthetic fallback; kept one latest-sector product per source for the model-ready subset and recorded all completed real sectors | 863/863 selected FITS products validate TIC ID, sector, table presence, and SHA-256; source-grouped label/split/product manifests created | Passed; model retraining/evaluation pending |
| 2026-10-29 | Trained `morphology_real_catalog_v1` from real MAST light curves using source-grouped train/calibration/selection partitions | 767/767 sources yielded finite features (597 train, 84 calibration, 86 selection); grouped CV selected HistGradientBoosting (macro-F1 0.630, transit-like recall 0.577); bundle load and Ruff passed; locked test not opened | Training complete; independent locked evaluation pending |
| 2026-10-30 | Froze and ran `real_catalog_cohort_v1_locked_eval` once using the fixed model, feature path, and checksummed real-cohort manifests | 96/96 test sources accounted for; 95 yielded candidate features, 1 trapezoid fit failed and was counted as an abstention; conditional macro-F1 0.682, transit-like precision 0.652/recall 0.600; end-to-end macro-F1 0.678; grouped bootstrap and report saved outside Git | Complete; catalog-label proxy limitation remains |
| 2026-10-31 | Cleaned workspace datasets; removed synthetic/experimental cohort data and redundant feature cache while preserving original sources | Removed 942 synthesized/unverified v2 FITS files, all 5,000 synthetic v3 FITS files, and the 1.7 MB v3 feature cache; preserved and checksummed 58 MAST downloads separately; pilot and real catalog cohort unchanged | Complete; historical manifests retained |
| 2026-10-31 | Re-trained `morphology_real_catalog_v2` as a deterministic reproducibility rerun from the preserved real cohort | 767/767 train/calibration/selection sources passed checksum and feature extraction with zero exclusions; identical grouped-CV selection (HistGradientBoosting macro-F1 0.630, transit-like recall 0.577); model bundle loaded and Ruff passed; test partition not opened | Complete as reproducibility rerun; no new independent performance estimate |
| 2026-10-31 | Acquired and froze `real_catalog_cohort_v2` from new NASA Exoplanet Archive TOIs and MAST light curves | 2,028 new TOI proxy labels considered; 918 MAST products discovered; 911 passed FITS/TIC/sector checks with SHA-256; source-grouped manifests cover 541 transit-like, 335 artifact-proxy, and 35 eclipse-proxy sources; `audit-cohort` and Ruff pass | Dataset ready; model fitting blocked by insufficient usable eclipse-like samples for grouped CV (14 usable train examples) |
| 2026-10-31 | Searched the MAST TESS-EBs v1.0 catalog and acquired an additional bright-source tranche | 246 unique, checksum-verified SPOC curves matched to EB catalog labels; fixed feature pipeline yielded 217/219 development feature vectors; 27 source groups assigned to an unopened EB holdout; acquisition and feature-audit manifests saved | Dataset ready for development-only combination; full 4,442-source remainder not queried after MAST service stalled |
| 2026-10-31 | Froze combined real-data cohort v3 and trained `morphology_real_catalog_v3` | 1,109 labels / 1,127 product records; 772 train, 104 calibration, and 108 selection feature vectors; two new EB fits failed and were reported; grouped 5-fold CV selected RandomForest (macro-F1 0.692, transit recall 0.617); Brier 0.435 / ECE 0.077 are calibration-fit diagnostics; model bundle reload, cohort audit, CLI help, Ruff, and 86 pytest tests passed; both test portions were not opened | Training complete; no independent evaluation claim; prior cohort test cannot be reused and EB holdout remains sealed |
| 2026-11-10 | Downloaded the v3 real-catalog training dataset through the MAST Portal Download API | 986/986 train/calibration/selection FITS products passed SHA-256 and FITS TIC/sector/table/column checks; 123 test-assigned sources excluded; 1.815 GiB stored in the ignored workspace; acquisition report and trainer-compatible subset manifest written; Python compilation and `git diff --check` passed | Dataset ready for training; catalog labels remain proxies and no model was trained or evaluated |
| 2026-11-10 | Prepared and trained the fresh PyTorch candidate-sequence model | Locked all optional dependencies; PyTorch 2.15.0+cu132 and CuPy detect the RTX 3050. Fresh source-grouped scikit-learn baseline: 771 train sources, macro-F1 0.698 in 5-fold CV; 3 extraction failures. CNN v3: 986 development sequences, 0 extraction failures, best epoch 6, calibrated selection macro-F1 0.608; checkpoint reload reproduced reported selection metrics exactly. No test FITS opened. Ruff, Python compilation, and diff checks passed; no test suite run. | Experimental artifact saved outside Git; proxy labels and selection metrics do not support an independent accuracy claim |
| 2026-11-11 | Completed the user-requested 50,000-target MAST corpus acquisition | 50,000/50,000 unique TIC light curves downloaded; no failures; every FITS passed structure, TIC/sector, and time/flux column checks; local SHA-256 recorded; 95,773,392,000 bytes in the ignored workspace. FITS PRIMARY CHECKSUM valid; LIGHTCURVE/APERTURE CHECKSUM invalid; DATASUM absent. Manifest labels remain unset. | Complete as an unlabeled corpus; not a supervised training or evaluation set |
| 2026-11-11 | Cross-referenced the corpus against the current ExoFOP-TESS TOI catalog | Exact TIC-ID join and same-sector gate: 139 products / 157 TOI events matched; 111 unambiguous `PC`/`PC` transit-like proxies assigned; 28 ambiguous and 49,861 unmatched products remain unlabeled. Label sidecar, source catalog, event matches, and SHA-256 provenance stored outside Git. Ruff, Python compilation, and `git diff --check` passed; no tests run. | Label overlay complete; catalog proxies do not establish independent morphology or planet confirmation |

### Colab training from downloaded TESS FITS (2026-11-19)

Staged 21,148 development product records across 6,255 sources for the raw-FITS
training notebook. Verified the notebook structure and code-cell syntax, and read,
quality-filtered and detrended a checksummed sector 1 observation for TIC 24704862
from the staged folder. Full extraction and model training on Colab remain pending.
