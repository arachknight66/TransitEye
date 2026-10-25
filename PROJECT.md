# PROJECT.md

## Project identity

**Name:** TransitEye  
**Problem:** AI-enabled detection of exoplanets from noisy astronomical light curves  
**Status:** Phases 1–5 complete (v0.1.0 release verified; 68-target FITS cohort, 100-target sector benchmark, locked evaluation, and PyQt6 desktop workstation complete)  
**Primary platform:** Linux, CUDA GPU-accelerated with deterministic CPU fallback  
**Distribution:** MIT scientific backend and GPLv3-compatible open-source PyQt6 desktop application

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

