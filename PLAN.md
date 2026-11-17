# PLAN.md

## Roadmap contract

This is the authoritative five-phase implementation sequence. Work within a phase follows task order
unless a task explicitly states otherwise. A phase is complete only after its verification gate
passes and `PROGRESS.md` records the evidence.

## Working-product remediation (approved 2026-11-01)

The prior phase gates must be revalidated because the legacy evaluator has label-conditioned features
and non-independent recovery/coverage references. Historical artifacts remain unchanged and must not
be used for scientific claims. This new remediation sequence supersedes the ordering of the
post-release evaluation task below; the EB holdout stays sealed until a new protocol is frozen.

- [x] **R1 Scientific integrity guardrails:** disable the legacy locked evaluator; remove
  label-conditioned feature transformations and calibration augmentation from the supported pilot
  trainer; disable the unvalidated fixed-phase significance output; stop implicit synthetic model
  training and expose unavailable classification without blocking physical analysis. Document the
  behavioral compatibility changes and add regression tests for leakage and missing-model handling.
- [x] **R2 Shared scientific configuration and models:** make configuration, model identity, code,
  and input provenance explicit across single and batch workflows; verify model checksums and
  unify the feature path. Add exposure-aware fitting and candidate-level failure reporting.
  Emit additive provenance and component checksums in schema-v3 result bundles while retaining schema-v2 read compatibility;
  historical bundles and frozen evaluation artifacts are not rewritten.
  Record the selected flux product and apply the same flux preference in individual and batch runs.
  Apply and record shared quality, gap, BLS period/duration grid, detrending, screening and bootstrap
  settings through the same typed configuration; default values preserve the historical search policy.
  Record requested versus resolved device so `auto` and batch CPU spillover are auditable.
  Record candidate-specific bootstrap seeds alongside the seed strategy in v3 output.
  Preserve raw FITS exposure keywords; allow an explicit exposure duration and propagate it to both
  fitting models and bootstrap refits. Mark the historical two-minute fallback as an assumption.
  Use an exact boxcar average for the piecewise-linear trapezoid; retain instantaneous mode and
  independently verify the integral. The Batman approximation remains separately qualified.
  A candidate fit failure must be recorded in v3 results and displayed without dropping other
  candidates; no failed fit may be represented as a successful physical inference.
- [x] **R3 Durable batch and desktop workflows:** implement bounded scheduling, real timeouts,
  pause/resume/cancel/retry, accurate accounting, cache validation, responsive desktop controls,
  saved sessions, and consistent CLI/UI results. Desktop single-target and batch runs share the
  inspector period bounds and the backend's 6.0 detection-SNR default; saved jobs remain unchanged.
  The detached marker lifecycle fix passed clean-wheel headless and native checks; the user repeated
  the manual Linux workflow on the current build successfully.
- [x] **R3 integrity follow-up:** correct source vetting that can treat absent centroid/crowding or
  zero-signal pixel data as evidence of an on-target source. Preserve v3 nullable field compatibility,
  explicitly report insufficient evidence, do not fabricate a SAP/PDCSAP depth ratio, and verify
  legacy bundles are not rewritten.
- [ ] **R4 Independent validation and release:** freeze development benchmarks; perform real-data,
  injection, clean-install, headless Qt, and manual Linux verification; then run the exposure/noise
  inference study and generate an auditable report. A 100-target benchmark, external reproduction,
  and independent model evaluation are not complete until their artifacts and gates exist.
  - [x] **New experimental PyTorch sequence classifier (approved 2026-11-10):** freeze a
    source-grouped candidate-sequence protocol and train a fresh tabular baseline on the same
    development sources before fitting a neural model. Then train a deterministic PyTorch model
    from scratch using detected-candidate light-curve sequences. Do not load or fine-tune historical
    classifier weights. Use train for fitting, calibration for probability calibration, and selection
    only for a predeclared choice; keep every test partition unopened. Record fit/extraction failures,
    class supports, seeds, checksums, code/dependency versions, and label limitations. This is
    experimental and does not change default desktop behavior, class taxonomy, split manifests, or
    result schemas. Gate neural training on a successful reproducible baseline and frozen sequence
    specification. Catalog-proxy labels do not establish candidate-local ground truth, so this
    experiment cannot support an independent accuracy claim.
  - [x] **User-requested 50,000-target MAST corpus:** froze and downloaded 50,000 distinct Sector
    1–13 TESS SPOC light curves through the MAST Portal. Excluded every target already present in
    project manifests or workspace files, kept labels explicitly unset in the frozen product
    manifest, recorded per-product hashes and all failures, and stored data outside Git. This added
    an unlabeled input corpus only; no
    existing partition or classifier training protocol changes, and the corpus is not approved for
    supervised use. Acquisition report: 50,000 successful, zero failed; 50,000 distinct TICs; 95.77
    GB stored in the ignored workspace. All files passed FITS structure, TIC/sector, and light-curve
    column checks. The PRIMARY FITS CHECKSUM is valid, while LIGHTCURVE/APERTURE CHECKSUM values are
    invalid and DATASUM is absent; per-product statuses are retained in the report.
  - [x] **User-requested TOI label cross-reference:** joined exact TIC IDs to the current ExoFOP-TESS
    TOI table (8,175 rows, source CSV SHA-256 in the report) and required same-sector evidence.
    Matched 139 products to 157 TOI events. Assigned 111 `transit_like` catalog proxy labels where
    TESS and TFOPWG dispositions were both `PC`; retained 28 products with ambiguous/conflicting
    dispositions and 49,861 without a TOI match as unlabeled. No entries were converted to negatives.
    Wrote a separate 50,000-row Parquet label sidecar and all 157 event matches as JSONL. The frozen
    product manifest, model features, and data partitions are unchanged; proxies are not independent
    morphology truth.
  - [x] **User-requested confirmed-exoplanet TESS cohort:** snapshotted 6,464 confirmed planets from
    NASA Exoplanet Archive `pscomppars`; 6,157 records have TIC IDs representing 4,561 host TICs.
    MAST inventory matched 4,242 host TICs to 22,880 SPOC products; 319 TIC hosts have no available
    product. All 22,880 products downloaded with zero failures. A separate event-time overlay assigned
    16,989 `transit_like` catalog proxies across 3,290 TICs when a confirmed transiting planet's
    frozen transit window overlapped the product interval. The other 5,891 products remain unlabeled
    (no confirmed transiting planet, incomplete ephemeris, or event outside interval). Preserve
    planet-to-host relationships and group all products by host. Host confirmation alone is not a
    morphology label; the proxy is not flux-shape adjudication. Original manifest remains unchanged.
  - [x] **User-requested EB catalog labels:** created a separate sidecar for all 32,057 MAST SPOC
    products. Exact TIC joins, explicit sector evidence, FITS TIC/sector checks, and primary-eclipse
    timing against product intervals yielded 10,595 `eclipse_like` catalog proxies. The remaining
    21,462 stay unlabeled with reasons; the known-EB table without sector data supplies context only.
    All 32,057 catalog-match audit records and source-group IDs are retained. The frozen product
    manifest, model features, and partitions are unchanged. The labels do not confirm curve morphology
    or source localization. Sidecar and report are in the external workspace; generator is
    `scripts/label_mast_eclipsing_binary_catalogs.py`.
  - [x] **Transit-like proxies for the EB cohort:** exact-joined all 32,057 products to the frozen
    ExoFOP-TESS TOI snapshot and applied the existing same-sector disposition policy. Found 248
    `transit_like` TOI proxy products. Of these, 105 also carry an EB catalog proxy; those remain
    product-level conflicts in the combined sidecar. The unambiguous combined overlay has 10,490
    `eclipse_like` and 143 `transit_like` products; 21,424 products remain unlabeled. Event records,
    raw dispositions, snapshot hash, and cross-catalog overlaps are retained. Candidate-level
    morphology still requires matching detected signals to event ephemerides.
  - [x] **Labeled-data candidate-sequence experiment (approved 2026-11-14):** clean and combine only
    unambiguous transit-like and eclipse-like sidecars from the confirmed-planet, EB/TOI, and
    50,000-target cohorts. Deduplicate products, resolve TIC-level class conflicts, exclude identities
    assigned outside TRAIN in prior frozen splits, then freeze deterministic source-grouped
    train/calibration/selection/unopened-holdout assignments before reading FITS data. Apply the blind
    candidate search and require detected ephemerides to match a catalog event; unmatched detections
    remain counted but unlabeled. Establish a tabular baseline on the exact same matched development
    candidates, then train a separate binary PyTorch sequence CNN from scratch. Do not infer negatives
    from unlabeled products, train the unsupported variability/artifact classes, tune on the holdout,
    or integrate the experiment into the desktop. Completed 2026-11-14: froze 23,084 products
    across 6,886 TIC groups, then blind-searched 6,255 development groups; 3,940 matched event
    candidates and 2,315 did not. Trained on 3,162 sequences, calibrated on 392, and selected on 386.
    CNN selection macro-F1 was 0.9254 versus 0.8158 for the same-candidate random-forest baseline.
    After model lock, opened the 631-source holdout once for inference-only final evaluation. Blind
    matching yielded 364 candidates (355 eclipse-like, 9 transit-like); 267 sources had no matched
    candidate. Holdout macro-F1 was 0.7429 for the CNN and 0.6135 for the random-forest baseline;
    balanced accuracy was 0.7693 and 0.8410 respectively. The transit-like support of nine is too
    small for a reliable class-specific conclusion. The holdout is now spent and must not be reused
    for tuning. Labels remain catalog proxies, not independent visual morphology adjudication.
  Correct the recorded deep-eclipse fit failure by extending the normalized trapezoid depth bound
  to 1.0 and rejecting impossible initial depths explicitly. This changes features in new analyses,
  but does not change class definitions, splits, or the existing result schema; old bundles remain
  readable and historical model scores do not validate the new feature path.
  Reject optimizer results that did not converge, reporting trapezoid failures through existing
  candidate errors and leaving failed optional physical refinements unavailable. Existing bundle
  fields and schema remain unchanged, while new fit/feature outcomes may differ.
  Flag candidates that include the unvalidated five-sample physical model using the existing
  candidate flag list, consistently in JSON and Parquet; prior bundles remain unchanged.
  Distinguish an attempted but failed physical refinement from a deliberately skipped one using
  the existing candidate flags and missing-reason map; do not fabricate a detailed cause.

  Development-only requested experiment: train an isolated three-class estimator on the new
  catalog collections with source-group separation, held-out-group exclusion, candidate matching
  and complete failure accounting. Existing four-class application contracts remain unchanged;
  the experiment is not a deployable classifier or independent evaluation. Retraining may add the
  additional v2 source collections under the same source-group and reserved-partition safeguards.
  The user-requested 2026-11-02 expansion freezes 20 new catalog-proxy sources per supported
  class (14 train and 6 test) before downloading or opening the new test observations. It excludes
  prior known source identities, keeps all failed detections, locks the model before the test, and
  reports conditional classification metrics separately from extraction and matching failures.
  This remains a three-class research experiment, not a calibrated desktop model or R4 gate.
  For the next user-requested expansion, freeze a larger source-disjoint development cohort using
  the remaining catalog-backed sources, preserve previous test partitions, and record imbalance,
  all failures and model limitations. Reuse only previously designated training features.
  A further variability-focused expansion freezes new Gaia/TIC cross-matches and holds out source
  groups before downloading their light curves; its one-class test cannot establish multiclass
  accuracy. Earlier test sources remain excluded from training.

  Desktop demo follow-up: explicitly launching the sample selects the historical v1 model only
  after verifying the frozen model files. Display the choice and limitations; leave normal analysis
  opt-in and preserve the existing result contract and independent-validation requirements.

  R4 completion requires all of the following evidence, with failures and missing results counted:
  - [ ] Validate physical-model integration across a declared grid and optimizer behavior; measure
    period, depth, duration, and interval coverage against independent known truth.
  - [ ] Run and document a frozen injection-recovery study and null controls, separate from
    real-data evaluation, with recovery, parameter errors, and uncertainty coverage.
    The development real-substrate pilot's exposure-status provenance was corrected to the existing
    unverified-fallback value and rerun; earlier artifacts are preserved for audit.
  - [ ] Audit catalog-derived labels and freeze source-separated selection and evaluation before
    opening a fresh independent holdout; previously opened test sources are not a fresh holdout.
    Development audit found automatically assigned review strings, false-positive/depth eclipse
    proxies and ATL target-list variability proxies; none establishes source-local morphology.
    Preserve historical manifests and disable legacy synthetic cohort builders and the historical
    evaluation entry point rather than regenerating unsupported labels or reopening old tests.
  - [ ] Develop and validate a search-wide significance calculation against appropriate null data;
    keep the existing unavailable value until validation succeeds.
  - [x] Freeze and run a 100-target development benchmark in the external workspace, recording
    settings, runtime, memory, completion, failures, and interruption recovery.
    Correct the legacy coordinator-only memory and total-target throughput before using it; retain
    old report field names while documenting the new sampled process-tree and fresh-work semantics.
    Historical benchmark artifacts are unchanged and do not establish current throughput or memory.
  - [ ] Verify clean installation, headless and manual Linux workflows, and reproduction in another
    person or environment; publish reproducibility inputs, versions, configurations and limitations.
  - [ ] Produce a scientific report with methods, evidence, failure cases and limitations. Historical
    evaluation outputs are not evidence for the current pipeline.

R1 is a prerequisite for R2–R4. Historical releases and model artifacts remain available for audit,
but the invalid legacy evaluation command and fixed-phase public significance helper must fail
explicitly rather than generate unsupported results.

## Build preferences

- **Mode:** Autonomous implementation in reviewable vertical slices.
- **Verification:** Automated checks at every slice and a mandatory phase gate.
- **Git cadence:** One coherent commit per completed vertical slice when commits are requested.
- **Platform:** Linux with CUDA GPU acceleration (`cupy-cuda13x` / Numba CUDA for detrending, BLS, Lomb-Scargle, bootstrap, and noise-surrogate searches) plus deterministic CPU fallback and numerical parity tests.
- **Data scope:** Archived SPOC/TESS-SPOC light curves with targeted target-pixel vetting.
- **Scientific policy:** Favor defensible baselines; add complexity only after measured failure.

## Phase 1 — Contracts, ingestion, and curated data foundation

**Goal:** Establish trustworthy inputs, typed outputs, dataset provenance, and leakage-safe partitions.

- [x] **1.1 Establish packages, tooling, and configuration**
  - Build: Add constrained runtime/dev groups, backend package boundaries, structured logging, typed
    configuration, and CLI entry point.
  - Acceptance: A clean clone syncs, imports, lints, tests, and exposes CLI help.
  - Verify: `uv sync --locked && uv run ruff check . && uv run pytest && uv run python -m transiteye --help`

- [x] **1.2 Define domain and result contracts**
  - Build: Immutable typed records for light curves, provenance, masks, configurations, candidates,
    classification/source assessments, uncertainties, progress, and complete analysis results.
  - Acceptance: Units, missing-value reasons, versions, no-signal, insufficient-data, and failure
    states serialize and round-trip without Qt imports.
  - Verify: Contract/schema unit and compatibility tests.

- [x] **1.3 Implement FITS ingestion and validation**
  - Build: SPOC/TESS-SPOC adapters for time, PDCSAP/SAP, flux errors, cadence, quality, centroids,
    crowding metadata, and FITS provenance.
  - Acceptance: Reference fixtures preserve time standard, units, cadence, flags, arrays, and source
    metadata; malformed and unsupported inputs fail clearly.
  - Verify: Fixture-based FITS tests and manual CLI inspection of representative products.

- [x] **1.4 Implement MAST discovery and immutable manifests**
  - Build: Product discovery, bounded downloads, checksums, retry/resume, caching, and offline replay.
  - Acceptance: A frozen pilot manifest reproduces the same inventory; interrupted downloads resume;
    unavailable products are accounted for.
  - Verify: Mocked network tests plus online pilot acquisition recorded by manifest checksum.

- [x] **1.5 Create label policy, pilot cohort, and locked split manifest**
  - Build: Curate a high-confidence cohort of ~60–80 TIC sources (~15–20 per class across
    `transit_like`, `eclipse_like`, `stellar_variability`, and documented `other_artifact`, plus ~8
    `sector_shift` and ~4 explicit `ambiguous=True` audit targets) with verifiable public HTTP(S)
    evidence URLs (NASA Exoplanet Archive, Villanova TESS EB Catalog, AAVSO VSX, and documented
    TFOP/SPOC artifact notes). Create source-grouped 60/10/10/20 `train`/`calibration`/`selection`/`test`
    partitions plus `sector_shift` in `manifests/pilot_labels.v1.json`, `manifests/pilot_splits.v1.json`,
    and frozen MAST product inventory `manifests/pilot_products.v1.json`.
  - Acceptance: Every label has provenance; conflicts and uncertainty are explicit; no astrophysical
    source or derivative crosses partitions; the test set remains sealed.
  - Verify: `uv run transiteye audit-cohort --labels manifests/pilot_labels.v1.json --splits manifests/pilot_splits.v1.json`
    plus automated label/leakage audit tests in `tests/test_datasets.py`.

**Phase gate:** Frozen pilot inputs and labels are auditable, schemas are versioned, and leakage tests
pass. No model training may begin before this gate. (PASSED 2026-10-24)

## Phase 2 — Preprocessing, detection, and characterization baseline

**Goal:** Produce scientifically interpretable candidates and parameter estimates without learned
classification.

- [x] **2.1 Build quality assessment and dual light-curve representations**
  - Build: `src/transiteye/preprocessing/quality.py` and `src/transiteye/preprocessing/representation.py`
    with documented TESS `QUALITY` bitmask filtering (default SPOC conservative mask `0x0BEE` / bits
    `1,2,3,4,6,8,10,11,12`), NaN/non-positive flux/error exclusion, per-cadence exclusion reason bitmask,
    gap/orbit segmentation (`gap_threshold_hours = 6.0`), per-segment median normalization, and robust
    scatter/CDPP diagnostics, producing an immutable `PreparedLightCurve` with minimally processed arrays.
  - Acceptance: No interpolation crosses gaps; negative excursions remain; every removed point has a
    recorded reason.
  - Verify: Synthetic edge cases and known FITS fixtures in `tests/test_preprocessing.py`.

- [x] **2.2 Implement duration-aware detrending**
  - Build: `src/transiteye/preprocessing/detrend.py` and `src/transiteye/execution/compute.py`
    (`ComputeBackend` abstraction for `cuda` / `cpu` / `auto` via `cupy-cuda13x` and `wotan`) implementing
    Tukey biweight time-window filtering in a two-pass strategy: pass 1 uses a configurable baseline window
    (`0.75–1.0 d`) per gap-bounded segment for initial search; pass 2 masks candidate in-transit cadences
    (`1.5 * duration` buffer) and recomputes the trend using a duration-scaled window
    `max(3 * duration, 0.5 d)`. Retain both minimally processed and detrended representations, the trend
    curve, and the transit mask.
  - Acceptance: Injection tests quantify recovery and parameter bias across durations, variability,
    cadence, and gaps; GPU and CPU paths agree within `rtol=1e-5, atol=1e-7`; defaults are frozen from
    development data only.
  - Verify: Signal-preservation test matrix and CPU/GPU parity tests in `tests/test_detrending.py`.

- [x] **2.3 Implement blind BLS and variability searches**
  - Build: `src/transiteye/detection/bls.py`, `src/transiteye/detection/variability.py`, and
    `src/transiteye/detection/peaks.py` with CUDA-accelerated batched Box Least Squares and Lomb-Scargle
    kernels (`cupy-cuda13x`) backed by `astropy.timeseries.BoxLeastSquares` and `LombScargle` CPU reference
    implementations (uncertainty-weighted, durations `max(0.5 h, 2 cadences)` to `12 h`, periods `0.3 d` to
    `0.5 * baseline`). Include local fine-grid peak refinement, `[P/2, P, 2P]` alias testing, odd/even depth
    comparison, secondary eclipse phase scan, minimum 2-event enforcement (`WEAK_TWO_EVENT` flag when
    `N_events == 2`), and iterative in-transit masking for up to 3 distinct dip candidates.
  - Acceptance: Known and injected signals recover within defined tolerances; GPU and CPU periodograms
    match within floating-point tolerance; sparse/short light curves return explicit states
    (`INSUFFICIENT_DATA`, `NO_SIGNIFICANT_SIGNAL`); maximum candidate count (`<= 3`) is enforced.
  - Verify: Detection unit tests (`tests/test_detection.py`), CPU/GPU parity checks, injection recovery,
    and blind pilot run.

- [x] **2.4 Refine candidates and estimate parameters**
  - Build: `src/transiteye/characterization/fitting.py` with bounded `scipy.optimize.least_squares`
    exposure-integrated symmetrical trapezoid fits `(T0, P, duration, depth, ingress_fraction, baseline)`,
    optional `batman-package` quadratic limb-darkened refinement for transit-like candidates (`SNR >= 7.0`),
    alternate-period `(P/2, P, 2P)` BIC/$\chi^2_\nu$ comparison, and explicit reporting of both measured
    depth and `CROWDSAP` dilution-corrected depth (`depth_ppm / CROWDSAP`).
  - Acceptance: Results report epoch/time standard, period, depth, duration, event count, phase
    coverage, fit diagnostics, aliases, and measured versus dilution-corrected depth distinctly.
  - Verify: Simulation bias/error tests and regression fixtures in `tests/test_characterization.py`.

- [x] **2.5 Estimate SNR, significance, and uncertainty**
  - Build: `src/transiteye/characterization/uncertainty.py` and `src/transiteye/reporting/bundles.py`:
    (1) `detection_snr` with Pont et al. (2006) duration-binned red-noise correction $\sigma_{\text{red}}(\tau)$,
    (2) `parameter_intervals` (68% and 95%) via GPU-batched / CPU-fallback seeded phase-folded residual
    block-bootstrap (`N = 200`),
    (3) `search_significance` via GPU-batched / CPU-fallback seeded block-permuted / Fourier
    phase-scrambled noise surrogates (`N = 200`, resolution $1/(N+1)$) on shortlisted candidates (`<= 3`),
    and (4) structured result bundle writer (`result.json`, `arrays.npz`, `candidates.parquet`) plus CLI
    subcommand `transiteye analyze-fits --device {auto,cuda,cpu}`.
  - Acceptance: The three score types are separately named; 68%/95% intervals and finite-surrogate
    resolution are exposed; failed and multimodal fits remain visible; GPU and CPU surrogate/bootstrap
    runs are deterministic from recorded seeds.
  - Verify: Coverage experiments, null tests, and deterministic seed reproduction in `tests/test_uncertainty.py`.

**Phase gate:** The CLI (`transiteye analyze-fits`) analyzes local FITS end to end, produces versioned
result bundle directories (`result.json`, `arrays.npz`, `candidates.parquet`), and meets frozen
development thresholds for recovery, false alerts, parameter error, and interval coverage.

## Phase 3 — Features, multiclass classification, and source vetting

**Goal:** Classify candidate morphology with calibrated confidence and assess likely contamination.

- [x] **3.1 Implement the audited feature registry**
  - Build: `src/transiteye/features/registry.py` and `src/transiteye/features/extractors.py` defining ~24
    audited features across 6 families: Morphology (`v_shape_ratio`, `duration_to_period_ratio`,
    `boxiness_index`, `asymmetry_score`, `batman_vs_trapezoid_delta_bic`), Consistency (`odd_even_depth_sigma`,
    `per_transit_depth_mad_ratio`, `phase_coverage_fraction`, `in_transit_duty_cycle`, `observed_event_count`),
    Eclipse (`max_secondary_snr`, `secondary_to_primary_depth_ratio`, `secondary_phase_offset_from_half`,
    `ellipsoidal_amplitude_ratio`), Variability (`ls_max_power`, `ls_to_bls_period_harmonic_distance`,
    `out_of_transit_rms_to_depth_ratio`, `autocorrelation_at_period`), Noise/Quality (`red_noise_beta_factor`,
    `detection_snr`, `in_transit_flagged_fraction`, `gap_proximity_fraction`), and Source Evidence
    (`centroid_shift_sigma`, `sap_to_pdcsap_depth_ratio`, `crowdsap_value`).
  - Acceptance: Features contain no prohibited label/identity/split data and remain stable across
    serialization and batch execution.
  - Verify: Automated leakage audit (`tests/test_features.py`), invariance tests, schema snapshot, and
    feature-distribution review.

- [x] **3.2 Create controlled synthetic augmentation and recovery datasets**
  - Build: `src/transiteye/datasets/injection.py` injecting exposure-integrated planetary transits,
    grazing/blended eclipsing binaries, and rotational/pulsational sinusoids strictly into `train`-partition
    quiet/variable real light-curve substrates with deterministic seeds (`seed = hash(source_group_id, variant_idx)`).
  - Acceptance: No `calibration`, `selection`, or `test` substrate or source appears in augmentation;
    synthetic examples carry `is_synthetic = True` and `parent_source_group_id` and never enter real-data
    headline metrics.
  - Verify: Lineage checks and stratified recovery summaries in `tests/test_injection.py`.

- [x] **3.3 Train and select the morphology classifier**
  - Build: `src/transiteye/classification/train.py` and `src/transiteye/classification/models.py`
    comparing multinomial `LogisticRegression`, class-weighted `RandomForestClassifier`, and
    `HistGradientBoostingClassifier` using ` StratifiedGroupKFold` grouped by `source_group_id` on `train`
    (comparing real-only vs. real + synthetic training augmentation as a logged ablation). Select by
    macro-F1, then `transit_like` recall and simplicity.
  - Acceptance: One reproducible model bundle directory (`model_metadata.json` with feature schema,
    imputation medians, training manifest ID, code/config versions, and SHA-256 hash + verified estimator
    artifact) records all training and CV evidence.
  - Verify: Grouped CV report, confusion matrices, class metrics, and reproducible model hash in
    `tests/test_classification.py`.

- [x] **3.4 Calibrate confidence and add abstention**
  - Build: `src/transiteye/classification/calibration.py` fitting regularized/shared-slope Platt
    (`sigmoid`) probability calibration on the held-out `calibration` partition and selecting abstention
    thresholds ($\tau_{\text{conf}}, \tau_{\text{margin}}$ and OOD feature guardrails) on the `selection`
    partition to emit `abstained = True, predicted_class = None`.
  - Acceptance: Reliability curves, multiclass Brier score, log loss, ECE, and accuracy-versus-coverage
    support the chosen policy; confidence is explicitly labeled as classifier confidence.
  - Verify: Calibration curves, abstention unit/integration tests, and partition-use audit.

- [x] **3.5 Implement independent source vetting**
  - Build: `src/transiteye/vetting/source.py` and `src/transiteye/io/tpf.py` implementing a two-tier
    source vetting pipeline:
    - **Tier 1 (always run from `_lc.fits`):** In- vs out-of-transit `MOM_CENTR1`/`MOM_CENTR2` centroid
      shift significance ($\sigma_{\Delta c}$), `SAP_FLUX` vs `PDCSAP_FLUX` depth consistency, and
      `CROWDSAP` / `FLFRCSAP` crowding assessment.
    - **Tier 2 (run when companion `_tp.fits` is provided or fetched for shortlisted candidates):**
      In- vs out-of-transit difference-image flux centroid offset and concentric aperture depth growth test.
  - Acceptance: Return `TARGET_CONSISTENT`, `BLEND_SUSPECTED`, or `UNRESOLVED` independently of
    morphology; missing pixel evidence in a crowded field (`CROWDSAP < 0.85`) yields `UNRESOLVED`
    rather than a definitive blend guess.
  - Verify: Known on/off-target cases, synthetic centroid shifts, and evidence-display review in
    `tests/test_vetting.py`.

**Phase gate:** Grouped real-data validation supports the deployed classes, calibration, abstention,
and source-assessment claims. The locked test set remains unopened.

## Phase 4 — Sector execution and PyQt desktop product

**Goal:** Scale the validated backend and expose it through a polished, responsive desktop workflow.

- [x] **4.1 Build persistent sector execution**
  - Build: `src/transiteye/execution/store.py`, `src/transiteye/execution/runner.py`, and
    `src/transiteye/execution/service.py` with SQLite WAL job state (`jobs`, `target_tasks`, `cache_entries`),
    bounded `ProcessPoolExecutor(max_tasks_per_child=25)` with VRAM-bounded GPU worker scheduling
    (`gpu_memory_limit_mb` and semaphore-gated CUDA streams with automatic CPU worker spillover),
    single-coordinator SQLite writes, SHA-256 `(input_sha256, config_sha256, model_sha256, schema_version)`
    cache lookup, cancellation, resume/retry, and fast-screen (skips `batman`/surrogates when pass-1 SNR
    `< screen_snr_threshold`) versus detailed analysis tiers. Expose CLI `transiteye run-batch` and
    `transiteye job-status`.
  - Acceptance: Each target is independently restartable; results are deterministic; concurrency, host RAM,
    and GPU VRAM remain bounded; every skipped/failed target is counted.
  - Verify: Interruption/restart integration tests and increasing-load CPU/GPU benchmarks in
    `tests/test_execution.py`.

- [x] **4.2 Freeze and run a sector benchmark**
  - Build: Freeze a deterministic 100–250 target single-sector slice manifest
    (`manifests/sector_benchmark.v1.json`) alongside support for full-sector local directories, and execute
    it to record throughput, runtime, peak RSS memory, disk footprint, cache hit rates, unavailable inputs,
    and failures in `.transiteye-workspace/benchmarks/`.
  - Acceptance: The full manifest reaches a terminal state with reproducible output inventory and no
    silent losses.
  - Verify: Manifest reconciliation and clean rerun/cache-hit checks.

- [x] **4.3 Establish the separate desktop package and process bridge**
  - Build: `desktop/transiteye_desktop/` PyQt6 Widgets application package, presenter/controller boundary,
    supervised `QProcess` backend runner invoking `python -m transiteye ... --progress-jsonl`, typed
    newline-delimited JSON (`JSONL`) progress event parser, workspace result bundle loader (`result.json`,
    `arrays.npz`, `candidates.parquet`), session persistence, and error recovery.
  - Acceptance: No Qt imports exist in `src/transiteye/`; the UI remains responsive during analysis and
    can cancel/reconnect; CLI and desktop resolve identical result bundles.
  - Verify: Architecture import test (`tests/test_architecture.py`), headless `pytest-qt` process tests,
    and CLI/UI bundle equivalence test.

- [x] **4.4 Implement the interactive scientific workflow**
  - Build: PyQt6 + PyQtGraph workspace panels: FITS/manifest opener, linked delivered versus detrended
    light-curve plots with quality-mask and trend overlays, BLS and Lomb-Scargle periodogram views,
    candidate table and selector (`<= 3` candidates + aliases), phase-folded transit view with
    trapezoid/`batman` model and residual subplot, calibrated class probability bar chart with abstention
    banner, source-vetting evidence panel (`MOM_CENTR` shifts, `CROWDSAP`, difference image when TPF
    present), and parameter/SNR/significance/uncertainty cards (including explicit measured vs
    dilution-corrected depth).
  - Acceptance: Candidate selection synchronizes all views; displayed units/definitions are clear;
    display-only min-max/LTTB downsampling does not alter underlying numerical arrays.
  - Verify: `pytest-qt` UI behavior tests (`tests/desktop/test_workflow.py`) and manual review on
    representative signal classes and failures.

- [x] **4.5 Complete batch controls, exports, and visual polish**
  - Build: Sector batch queue monitor with progress/cancel/resume/retry controls, saved `.teye.json`
    workspace sessions, CSV/Parquet/JSON/NPZ bundle exports, `src/transiteye/reporting/figures.py`
    Matplotlib publication-grade PNG/PDF/SVG summary vetting sheets, high-contrast scientific theme,
    keyboard shortcuts, and desktop entry point (`transiteye-desktop`).
  - Acceptance: A new Linux user can install, analyze one FITS file, run a batch, reopen a session, and
    export a complete result without the UI freezing.
  - Verify: Headless UI suite, packaged-app smoke test, accessibility review, and export round trips.

**Phase gate:** The sector benchmark passes, the packaged Linux desktop completes the full workflow,
and desktop results match the backend CLI.

## Phase 5 — Locked evaluation, scientific report, and release

**Goal:** Establish final evidence, document limitations, and produce a reproducible release.

- [x] **5.1 Freeze the final evaluation protocol**
  - Build: `src/transiteye/evaluation/protocol.py` and `configs/evaluation_locked.v1.toml` freezing
    ephemeris matching rules ($|\Delta P / P| \le 0.01$ or exact $1/2\times, 2\times$ harmonic overlap plus
    in-transit cadence overlap $\ge 0.5$), metrics (macro-F1, per-class precision/recall/F1, ECE, Brier,
    period/depth/duration relative error, interval coverage), source-grouped bootstrap (`N = 1000`),
    robustness strata (SNR, crowding, cadence, sector length), ablations (pass-1 vs pass-2 detrending,
    real-only vs synthetic-augmented training, feature family drop-one), and report layout before reading
    locked `test` labels.
  - Acceptance: Protocol covers detection, classification, parameters, calibration, source assessment,
    end-to-end misses, shift, and runtime; no success metric can silently drop failures.
  - Verify: Independent protocol checklist and configuration SHA-256 checksum test in
    `tests/test_evaluation_protocol.py`.

- [x] **5.2 Run locked and robustness evaluations once**
  - Build: `src/transiteye/evaluation/runner.py` and CLI `transiteye run-locked-evaluation` executing a
    single sealed evaluation across the locked `test` partition, `sector_shift` cohort, synthetic
    injection-recovery matrix, and feature/detrending ablations with target-group 95% confidence intervals.
  - Acceptance: Outputs include support counts, failures, harmonics, abstentions, limitations, and
    distinct real/synthetic conclusions. No tuning follows test inspection.
  - Verify: Reproduction from frozen manifests and checksum reconciliation.

- [x] **5.3 Produce publication and user deliverables**
  - Build: `docs/model-card.md`, `docs/methodology.md`, `docs/data-and-label-statement.md`,
    `docs/result-schema.md`, `docs/user-guide.md`, reproducible Matplotlib figures/LaTeX tables in
    `docs/report/generated/`, and a two-column LaTeX scientific report (`docs/report/main.tex` compiled via
    `pdflatex` to `docs/report/transiteye_report.pdf`) of at most three pages including references.
  - Acceptance: Every headline statement traces to a frozen artifact macro; figures remain readable at
    final size; the report covers problem/data/method, evaluation/results, limitations/conclusion in
    $\le 3$ pages.
  - Verify: Automated provenance/macro check, `pdfinfo` / page-count assertion (`pages <= 3`), visual
    inspection, and documentation smoke test.

- [x] **5.4 Perform clean-room release acceptance**
  - Build: Produce the MIT backend wheel/sdist, GPLv3-compatible desktop distribution, `NOTICE` / license
    inventory, SHA-256 release manifest, `RELEASE_NOTES.md`, and `scripts/verify_clean_release.sh`.
  - Acceptance: From a clean environment, dependencies lock, tests pass, package imports, CLI works,
    desktop launches headlessly, reference analysis matches, exports open, and no large/generated data is
    tracked in Git.
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

## Post-release real-catalog retraining (2026-10-31)

- [x] Acquired a separate TESS-EBs catalog tranche and combined its source groups with existing
  real-catalog development partitions for a morphology-model retraining run.
  - Rationale: the larger TOI expansion had too few usable eclipse-like sources for grouped model
    selection; the curated MAST TESS-EBs catalog provides direct eclipsing-binary evidence.
  - Compatibility: new manifests are versioned as `real_catalog_cohort.v3.*`; both original
    source-group assignments are preserved. Both test partitions remain assigned to `test` and were
    not read during training. The EB label describes catalogued system morphology, not source location.
  - Acceptance: checksummed products, source-grouped manifests, reproducible feature extraction,
    complete train/calibration/selection class support, and a saved model/training report.
  - Verification: grouped five-fold CV and calibration/selection on development partitions;
    independent evaluation remains pending and must use a newly frozen protocol.
- [ ] Freeze an independent evaluation protocol for `morphology_real_catalog_v3`, audit catalog
  label validity and source-local event matching, then perform one locked evaluation.

