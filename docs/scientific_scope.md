# Scientific scope

TransitEye is a reproducible research pipeline for detecting and vetting
transit-like events in stellar light curves. TESS is the planned primary data
source; no TESS data are included in this repository foundation.

## Confirmed design decisions

- The system will be a two-stage pipeline: a transit-search baseline proposes
  candidates, then machine learning classifies/vets those candidates.
- The intended baseline transit search is Box Least Squares (BLS).
- The main supervised unit will be a BLS candidate event, not an entire raw
  light curve.
- Evaluation will report both BLS recovery and candidate-classification
  performance, including their combined cascade.
- Production logic belongs in `src/`; notebooks are exploratory only.
- Raw astronomical products will remain immutable and traceable by manifest and
  checksum.

## Scope boundaries for the MVP

The future MVP is intended to include provenance-aware data acquisition,
preprocessing, BLS, engineered time/frequency-domain features, classical ML,
ELM/KELM, grouped evaluation, and robustness experiments.

It does not include full-frame-image extraction, pixel-level centroid vetting,
planet confirmation, a web application, or deep learning as a requirement.

## Unresolved scientific decisions

- **TBD:** final target cohort size, sector coverage, cadence policy, and
  product-family policy.
- **TBD:** final BLS period/duration grids and candidate-ranking policy.
- **TBD:** preprocessing constants and detrending method selection.
- **TBD:** whether optional time-frequency features or a 1D CNN are justified
  after classical baselines are complete.
- **TBD:** statistical criteria for reporting any exploratory candidate scores.

Any output of this project is a candidate-vetting result, not a claim that a
new exoplanet has been discovered or confirmed.
