# PROJECT.md

## Project identity

**Name:** TransitEye  
**Problem:** AI-enabled detection of exoplanets from noisy astronomical light curves  
**Status:** Clean foundation; rebuild planned; implementation has not started  
**Primary platform:** Linux, CPU first  
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

The repository currently contains only the development foundation:

```text
TransitEye/
├── .gitignore
├── .pre-commit-config.yaml
├── .python-version
├── AGENTS.md
├── PROJECT.md
├── PLAN.md
├── PROGRESS.md
├── README.md
├── pyproject.toml
├── uv.lock
└── src/transiteye/__init__.py
```

Current runtime dependencies are empty. Python is constrained to 3.11, Hatchling builds the package,
and uv manages the environment. The package imports and exposes version `0.1.0`.

## Target technology stack

| Area | Planned technology | Purpose |
|---|---|---|
| Environment | Python 3.11, uv, Hatchling | Reproducible builds and packaging |
| Astronomy I/O | Astropy, Astroquery, Lightkurve | FITS, time/units, MAST, TESS products |
| Numerical methods | NumPy, SciPy, Wotan | Arrays, fitting, robust detrending |
| Detection | Astropy BLS and Lomb-Scargle | Periodic dips and stellar variability |
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
- CSV/Parquet tables, JSON metadata, compressed arrays, and PNG/PDF/SVG figures.

### Default analysis policy

- Search periods from 0.3 days to half the observed baseline.
- Search durations from the larger of 0.5 hours or two cadences through 12 hours, while keeping
  duration shorter than period.
- Require at least two observed events and flag two-event solutions as weakly constrained.
- Retain distinct peaks, test half/double-period aliases, odd/even events, and secondary eclipses.
- Search masked residuals for at most three distinct dip candidates per target.
- Use robust biweight detrending with duration-aware windows and a second pass that masks events.

All defaults are configurable, versioned, and shown in analysis results.

## Desktop product

The PyQt6 application will provide local FITS opening, raw/delivered versus denoised views, analysis
controls, BLS and variability periodograms, candidate selection, phase-folded views, classification
and source evidence, parameter/uncertainty panels, sector job management, saved sessions, and result
exports.

The application launches analysis in a supervised backend process. The UI remains responsive and
does not duplicate scientific algorithms. CLI and desktop runs with the same inputs and configuration
must produce equivalent result bundles.

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

