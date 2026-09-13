# Changelog

All notable TransitEye project milestones are summarized here. This academic
release preserves deterministic artifact identities rather than mirroring every
development commit.

## 0.1.0 — Final project release

### Foundation and provenance

- Established typed configuration, canonical serialization, deterministic
  identities, seed policy, provenance records, and offline-first tests.

### Catalog and acquisition

- Froze the NASA Exoplanet Archive TOI snapshot and pilot cohort.
- Added auditable TESS-SPOC MAST discovery, selection, download receipts, raw
  checksums, and immutable product expansion.

### Detection and datasets

- Added transit-preserving preprocessing, blind BLS search, deterministic peak
  identities, harmonic event matching, and candidate/event dataset contracts.
- Kept catalog truth and unmatched-candidate semantics separate from features.

### Controlled demo

- Added deterministic planet-like, eclipsing-binary-like, sinusoidal, and
  control variants on real TESS substrates.
- Reused the scientific preprocessing, BLS, matching, dataset, and downstream
  contracts without changing scientific labels.

### Features and modeling

- Added a 65-feature shared schema spanning time-domain, BLS, Lomb–Scargle, and
  FFT groups.
- Compared Logistic Regression, Random Forest, RBF SVM, and ELM with TIC-grouped
  splits and training-only transforms.
- Froze the selected Random Forest and threshold.

### Evaluation

- Added leave-one-TIC-out robustness, group bootstrap intervals, injection
  recovery, feature stability, calibration diagnostics, interpretation,
  candidate scorecards, domain-shift analysis, and scientific-readiness gating.

### Reproducibility and documentation

- Added a 613-artifact portable verification manifest, deterministic final
  tables and figures, offline verification/report/smoke commands, final README,
  technical report, usage documentation, and presentation package.

### Release preparation

- Audited portability, dependencies, credentials, scripts, claims, caches, and
  generated artifacts; added deterministic release metadata and acceptance
  checks while preserving all frozen scientific identities.
