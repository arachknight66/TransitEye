# PROJECT.md

## Project identity

**Name:** TransitEye  
**Problem:** AI-enabled detection of exoplanets from noisy astronomical light curves  
**Status:** Working-product remediation in progress; previous release gates require revalidation.
The `morphology_real_catalog_v3` retraining candidate has not had an independent locked evaluation.
**Primary platform:** Linux, CUDA GPU-accelerated with deterministic CPU fallback  
**Distribution:** MIT scientific backend and GPLv3-compatible open-source PyQt6 desktop application

## Working-product remediation (approved 2026-11-01)

The previous release's scientific and operational gates require revalidation. The legacy locked
runner modifies features using labels, and its recovery and coverage calculations do not use
independent truth. Its outputs are historical and invalid for scientific performance claims.
The real-catalog v1 evaluation remains a historical, proxy-label evaluation; its test sources cannot
be reused for tuning or independent evaluation of later models. The separate EB holdout remains sealed.

The approved next release prioritizes a working CPU-first desktop and a resumable batch pipeline
using the same backend. First, disable the invalid runner, remove label-conditioned feature edits,
and prevent unavailable or unverified classifiers and significance estimates from silently producing
scientific claims. Subsequent slices will add explicit model selection, full configuration and cache
identity, exposure-aware fits, candidate-level error handling, supervised jobs, and real-data gates.
Existing result bundles remain readable; any change to emitted contracts will be versioned and
documented before implementation. Existing model artifacts and frozen evaluation results are not
rewritten. No label taxonomy or dataset partition is changed by this remediation.
The public historical `compute_empirical_significance` helper is also disabled: its fixed-phase
statistic did not repeat the search and must not be mistaken for a search-wide significance.

Desktop demo correction: explicitly starting the sample workflow also selects the locally stored
historical v1 classifier, after checking its three files against the frozen historical protocol.
The selection and current-pipeline validation limitation remain visible. Ordinary analysis still
requires explicit model selection. No classifier, label policy, result schema, or split changes;
the historical results remain unsuitable for current-pipeline performance claims.

Scientific-integrity correction: source vetting must not replace absent crowding, centroid, or
pixel-difference measurements with reassuring values. New analyses abstain or report unresolved
when those data cannot support localization. This changes source-assessment behavior, not class
definitions or splits. Existing v3 bundles retain their historical values and are not rewritten;
new results retain the nullable v3 artifact fields, with code identity identifying the change.
The purported SAP/PDCSAP depth ratio is unavailable unless both products are independently measured;
a delivered curve divided by its own detrended version is not that comparison.

Label-audit finding: historical catalog dispositions and target-list membership are proxies, not
source-local morphology ground truth. The historical builder automatically assigned review-status
strings and inferred some eclipse-like labels from false-positive disposition and depth; ATL
membership alone is not a measured stellar-variability label. Historical v1 TRAIN contains 150
deep-false-positive eclipse proxies and 142 ATL-derived variability proxies. No historic labels or
splits are rewritten, and no independent classifier evaluation is claimed. The legacy synthetic
cohort builders and historical real-cohort evaluator will be disabled as entry points because they
can produce unsupported labels or rerun an already-open evaluation; old artifacts remain for audit.

R4 fit correction: the trapezoid depth bound changes from 0.5 to 1.0 of normalized baseline to
permit deep eclipses, and impossible initial depths are rejected explicitly. This changes fitted
features for new analyses, with code fingerprints distinguishing them; prior bundles remain
readable and historical classifier performance cannot validate the changed feature path. Class
definitions, dataset splits, and result schema remain unchanged.
New fits will also reject an optimizer result that does not report convergence; failed trapezoid
fits use the existing per-candidate failure representation and failed optional Batman fits stay
unavailable. This may turn formerly reported fits into explicit failures and can alter downstream
features; old bundles are unaffected and no historical model score validates the changed behavior.
Whenever a new result includes the five-sample Batman model, its candidate will carry the additive
`physical_quadrature_unvalidated` flag in both JSON and Parquet. The existing string-list flag field
and v3 schema are unchanged; historical bundles are not modified. This makes the known numerical
limitation visible without claiming that the separate trapezoid-derived parameters are validated.
The real-substrate injection pilot must use the existing two-minute fallback as an unverified
assumption; passing that assumption as an explicitly supplied exposure incorrectly labels it
`user_supplied`. The pilot was rerun with the existing v3 unverified-fallback provenance;
earlier pilot outputs remain unchanged and are not the accepted evidence.
When optional Batman refinement was attempted but returned no model, new results will also carry
`physical_fit_failed` and an existing `fit_failed` missing-reason value for the physical fit. A
deliberately skipped refinement receives neither. This is an additive flag and missing-reason entry
in v3, leaving historical bundles, taxonomy, and split definitions unchanged.

Benchmark reporting correction: new benchmark reports measure sampled summed resident memory of
the Linux coordinator/worker process tree and count newly completed work separately from failures,
cache hits and previously completed tasks. Sampling may miss short-lived peaks and RSS may count
shared pages more than once. Existing report fields remain readable but their peak-memory and
throughput semantics change in newly generated reports; `details` records the measurement method,
settings, and recovery evidence. Historical benchmark reports are not rewritten or validated.

Desktop consistency correction: new single-target and batch runs use the shared backend's 6.0 SNR
default and the period range selected in the target inspector. Previously the desktop single-target
threshold was 7.1 and its batch run ignored that range. Existing results and saved jobs keep their
recorded settings; the changed desktop default is not a validated improvement in sensitivity.

R2 contract change: new analysis bundles use schema v3 and include the effective runtime analysis
settings, backend package version and source SHA-256, and explicit classifier identity/fingerprint.
These fields explain which code and options produced a result; schema-v2 bundles remain readable,
but do not gain provenance they never recorded. New bundles also record component checksums for
cache integrity. This does not change class definitions or splits.
The actual FITS flux column is recorded in input provenance, and batch and single-target commands
accept the same flux preference. This is additive for new results; old v2 bundles remain readable.
For `auto` device selection, results record both the requested setting and the device actually used,
including CPU spillover in mixed-device batch jobs.
New results record a per-candidate seed generated by the deterministic bootstrap seed strategy.
Quality filtering, gap segmentation, BLS search ranges/grid and bootstrap settings are shared typed
options across CLI and batch. Defaults retain the existing search policy; selected options affect
cache identity and are recorded in the v3 bundle.

R2 exposure change: preserve the FITS frame/exposure header values without treating total `EXPOSURE`
or sampling spacing as a per-cadence integration time. The analysis accepts an explicitly supplied
per-cadence exposure, applies it to trapezoid and physical-model fitting and bootstrap refits, and
records whether it was supplied. Missing exposure retains a clearly flagged historical assumption;
it is not advertised as a measured value. The old v1 classifier was not independently evaluated
under the corrected physical-model integration, so its historical metrics are not deployment metrics.
Schema-v3 candidate failure reporting will retain detected candidates whose fits fail, attach the
failure reason and `fit_failed` flag, and keep other candidates. Such candidates carry unrefined
detector estimates rather than inferred physical fits; they have no model curve or classification.
For numerical accuracy at long exposures, the trapezoid forward model uses an exact periodic boxcar
integral rather than a five-point sum. `supersample <= 1` retains instantaneous-sample behavior;
`supersample > 1` selects the exact integral. This changes fits from older releases and invalidates
any direct application of old classifier evaluation scores to the updated feature pipeline. Batman
still uses five samples and requires a separate convergence study.

## Product objective

TransitEye will ingest TESS FITS light curves, preserve and denoise scientifically relevant signals,
detect periodic dips and variability, characterize candidate events, classify their morphology,
assess possible source contamination, and make the evidence inspectable through a polished desktop
application.

The product supports individual FITS analysis and resumable processing of archived sector-level
SPOC and TESS-SPOC light curves. It does not claim planetary confirmation; it ranks and explains
candidate signals for scientific review.

## Current repository state

The repository implements the complete five-phase TransitEye scientific backend and PyQt6 desktop workstation:

```text
TransitEye/
├── AGENTS.md
├── PROJECT.md
├── PLAN.md
├── PROGRESS.md
├── README.md
├── LICENSE
├── NOTICE
├── RELEASE_NOTES.md
├── RELEASE_MANIFEST.sha256
├── pyproject.toml
├── uv.lock
├── configs/
│   ├── evaluation_locked.v1.toml
│   └── phase1.example.toml
├── docs/
│   ├── data-and-label-statement.md
│   ├── methodology.md
│   ├── model-card.md
│   ├── phase1-data-policy.md
│   ├── result-schema.md
│   ├── user-guide.md
│   └── report/
│       ├── main.tex
│       ├── transiteye_report.pdf
│       └── generated/
├── manifests/
│   ├── README.md
│   ├── pilot_labels.v1.json
│   ├── pilot_products.v1.json
│   ├── pilot_splits.v1.json
│   └── sector_benchmark.v1.json
├── src/transiteye/
│   ├── __init__.py
│   ├── __main__.py
│   ├── config.py
│   ├── logging.py
│   ├── characterization/
│   ├── classification/
│   ├── datasets/
│   ├── detection/
│   ├── domain/
│   ├── evaluation/
│   ├── execution/
│   ├── features/
│   ├── io/
│   ├── preprocessing/
│   ├── reporting/
│   └── vetting/
├── desktop/transiteye_desktop/
│   ├── __init__.py
│   ├── __main__.py
│   ├── bridge.py
│   ├── bundle_loader.py
│   ├── decimation.py
│   ├── main_window.py
│   └── widgets/
├── scripts/
│   ├── download_pilot_dataset.py
│   └── verify_clean_release.sh
└── tests/
```

Python is constrained to 3.11, Hatchling builds the package (`uv build`), and `uv` manages the
locked environment. The package exposes version `0.1.0`, a Qt-free CLI (`transiteye`), and a
separate desktop entry point (`transiteye-desktop`).

## Target technology stack

| Area | Planned technology | Purpose |
|---|---|---|
| Environment | Python 3.11, uv, Hatchling | Reproducible builds and packaging |
| Astronomy I/O | Astropy, Astroquery, Lightkurve | FITS, time/units, MAST, TESS products |
| Numerical methods | NumPy, SciPy, Wotan, CuPy (`cupy-cuda13x`) / Numba CUDA | Arrays, fitting, robust detrending, and GPU-accelerated kernels with CPU fallback |
| Detection | Astropy / CUDA-batched BLS and Lomb-Scargle | GPU/CPU periodic dip search, noise-surrogate grids, and stellar variability |
| Transit modeling | Trapezoid fits, batman | Candidate refinement and exposure integration |
| Machine learning | scikit-learn | Baselines, random forest, calibration, metrics |
| Data storage | PyArrow/Parquet, NumPy, SQLite | Results, arrays, and resumable jobs |
| Desktop | PyQt6 Widgets, PyQtGraph | Linux desktop workflow and interactive plots |
| Publication figures | Matplotlib | Reproducible export-quality graphics |
| Quality | pytest, pytest-qt, Ruff | Scientific, integration, and UI verification |

Versions will be constrained and locked when each dependency is introduced. QLP support follows
validation of the SPOC/TESS-SPOC path.

## Target architecture

```text
TESS FITS / MAST manifests
          │
          ▼
I/O and provenance ──► quality assessment
          │
          ▼
dual light curves: minimally processed + detrended
          │
          ├──► BLS dip search ───────────────┐
          └──► Lomb-Scargle variability ─────┤
                                              ▼
                                candidate refinement and features
                                              │
                           ┌──────────────────┴──────────────────┐
                           ▼                                     ▼
               morphology classifier                   source vetting
                           └──────────────────┬──────────────────┘
                                              ▼
                           SNR, significance, uncertainty, exports
                                              │
                                   ┌──────────┴──────────┐
                                   ▼                     ▼
                                  CLI              PyQt desktop
```

### Target repository layout

```text
src/transiteye/
├── domain/             # Typed records, units, schemas, result contracts
├── io/                 # FITS readers, MAST adapters, manifests
├── preprocessing/      # Quality masks, normalization, detrending
├── detection/          # BLS, Lomb-Scargle, peaks, residual searches
├── characterization/   # Fits, aliases, SNR, significance, uncertainty
├── features/           # Audited candidate feature extraction
├── classification/     # Training, inference, calibration, abstention
├── vetting/            # Centroid, pixel, crowding, source assessment
├── datasets/           # Labels, grouping, splits, injections
├── evaluation/         # Metrics, robustness, ablations, report inputs
├── execution/          # Service API, caching, jobs, multiprocessing
└── reporting/          # CLI, result exports, scientific figures

desktop/                # Separate PyQt6 application package
tests/                  # Unit, scientific, integration, scale, and UI tests
configs/                # Versioned analysis and training configurations
manifests/              # Small immutable dataset definitions
docs/                   # Methods, schemas, model card, report source
```

## Core scientific behavior

### Inputs

- Local TESS SPOC and TESS-SPOC light-curve FITS files.
- Archived sector product manifests retrieved from MAST.
- Target-pixel files for shortlisted crowded-field candidates.
- Curated, provenance-rich label records.

Real-catalog training snapshots must preserve catalog evidence and label scope. A catalog-identified
eclipsing binary may support an `eclipse_like` signal label, but it does not establish that the
strongest automatically detected event in every SPOC light curve is on-target or even the catalogued
ephemeris. Catalog absence is never a negative label. Retraining snapshots combine only explicitly
assigned source groups and must leave frozen test partitions unopened until a separately frozen
evaluation protocol is approved.

### Outputs

- Zero or more candidates for each target, including explicit no-signal and failure outcomes.
- Period, epoch, depth, duration, event count, phase coverage, and alternate aliases.
- Detection SNR, empirical search significance, and calibrated class probabilities.
- Signal morphology: transit-like, eclipse-like, stellar variability, or other/artifact.
- Independent source assessment: target-consistent, blend-suspected, or unresolved.
- Provenance, quality flags, uncertainty intervals, configuration, and model/schema versions.
- Structured result bundle directory per analysis (`result.json` for typed metadata/candidates/classifications, `arrays.npz` for dual light curves, masks, trend, and periodograms, and `candidates.parquet` for tabular summaries), plus PNG/PDF/SVG figures.

### Default analysis policy

- Search periods from 0.3 days to half the observed baseline.
- Search durations from the larger of 0.5 hours or two cadences through 12 hours, while keeping
  duration shorter than period.
- Require at least two observed events and flag two-event solutions as weakly constrained.
- Retain distinct peaks, test half/double-period aliases, odd/even events, and secondary eclipses.
- Search masked residuals for at most three distinct dip candidates per target.
- Use two-pass robust biweight detrending (`wotan`): pass 1 with a baseline window (`0.75–1.0 d`) for
  initial BLS search, and pass 2 masking detected events with a duration-scaled window
  `max(3 * duration, 0.5 d)`.
- Run block-bootstrap parameter intervals and seeded noise-surrogate empirical search significance
  (`N = 200` surrogates) on shortlisted candidates (`<= 3` per target).
- Run two-tier independent source vetting: always evaluate `_lc.fits` centroid shifts (`MOM_CENTR1/2`),
  `SAP` vs `PDCSAP` depth consistency, and `CROWDSAP`/`FLFRCSAP` dilution, and run difference-image
  plus multi-aperture pixel vetting whenever a companion `_tp.fits` is provided or fetched for
  shortlisted candidates (returning `UNRESOLVED` rather than guessing when pixel data is absent in a
  crowded field).

All defaults are configurable, versioned, and shown in analysis results.

## Desktop product

The PyQt6 application will provide local FITS opening, raw/delivered versus denoised views, analysis
controls, BLS and variability periodograms, candidate selection, phase-folded views, classification
and source evidence, parameter/uncertainty panels, sector job management, saved sessions, and result
exports.

The application launches analysis in a supervised backend `QProcess` streaming newline-delimited JSON
(`JSONL`) progress events over standard output and reading versioned result bundle directories from the
workspace. The UI remains responsive and does not duplicate scientific algorithms. CLI and desktop runs
with the same inputs and configuration must produce equivalent result bundles.

## Quality goals

- Optimize macro-F1 while reporting per-class behavior and transit recall.
- Measure detection completeness and false-alert burden across signal/noise regimes.
- Measure period, depth, and duration error, including harmonic recovery and missed detections.
- Validate confidence calibration and abstention behavior on independent real data.
- Validate signal preservation through injection/recovery tests.
- Complete one frozen sector benchmark with bounded memory, resumability, and full accounting.
- Produce a reproducible scientific report no longer than three pages, including references.

## Explicit exclusions

- Planet confirmation or statistical validation from light curves alone.
- Direct extraction from full-frame images in the initial rebuild.
- Custom aperture/PSF deblending as a core feature.
- Neural denoisers, CNNs, transformers, and distributed infrastructure before baseline evidence
  demonstrates a need.


### Development-only three-class experiment (2026-11-01)

The user-requested collection contains transit-like, eclipse-like and stellar-variability examples
but no independently labeled other/artifact examples. A standalone research estimator may be
trained on those three existing labels; it is not a four-class `ModelBundle`, is not loaded by the
desktop, does not change class definitions or result schemas, and cannot support calibrated or
independent-test claims. Selection must exclude conflicting sources and any groups assigned outside
TRAIN in existing split manifests, using split identities only; no sealed holdout data or labels are
opened. New unassigned groups receive a deterministic development train/validation split. Known
ephemerides may be used only for candidate-to-catalog label checks, never as input features.

The additional v2 collections may be incorporated into that development-only experiment with the
same group identities, reserved-partition exclusions, label checks and four-class limitations.
