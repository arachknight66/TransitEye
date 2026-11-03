# Additional Gaia variability research data, 2026-11-02

This development experiment adds 29 TESS light curves for newly selected stellar-variability
proxies. It does not produce a validated four-class desktop model. Gaia classifications are
source-level proxies; the detected TESS signal has not been independently localized or matched to
a Gaia event ephemeris. No observations or labels from the separate sealed eclipsing-binary
holdout were opened.

## Discovery and selection

A [Gaia DR3 variability classifier query](https://gaia.aip.de/metadata/gaiadr3/vari_classifier_result/)
returned 2,000 sources with classifier score above 0.5, selected classes, and the magnitude and
declination bounds recorded in `gaia.json`. Discovery was bounded to the first 500 with a unique
DR2 neighbor within 0.5 arcsec. A TIC lookup and MAST observation query found 30 sources with
SPOC light-curve products. Recorded query data and discovery scripts are in
`.transiteye-workspace/remediation/variable-discovery-20261102/`.

The frozen selection required a unique TIC match, a Gaia DR3/TIC angular separation no greater
than one arcsecond, and exclusion of prior known TIC/Gaia identities, all existing split identities,
confirmed planet-host TICs, known Gaia eclipse candidates and previously staged products. One
candidate did not qualify, leaving 29 source groups. Unknown aliases and source confusion remain
possible. Selection hashes and all input paths are recorded in the frozen manifest. Before opening
these light curves, a deterministic Gaia-identity hash reserved five for test and 24 for training.
Earlier experiments' test sources were not used for training.

## Data and model

All 29 sources produced downloaded light-curve FITS files that passed internal FITS checksums and
TIC/sector verification. SHA-256 hashes were computed from the downloaded [MAST public archive](https://registry.opendata.aws/mast-tess/)
bytes; they are not independently supplied checksums. The original inputs are retained. The
existing quality-mask preparation, detrending, blind 0.3–5-day BLS search, fitting and feature
extraction were applied; no Gaia ID, class, or other catalog label was used as a model feature.
All 24 training and five test files produced feature vectors. Because these are source-level
variable-star labels, a usable extraction does not establish that the detected event corresponds
to the catalogued variability or originates on the catalogued star.

A 200-tree random forest with balanced class weights, depth limit 8 and seed 42 was trained on
455 unique source groups: 431 prior training examples plus 24 new variable-star examples. The
class counts were 131 transit-like, 181 eclipse-like and 143 stellar-variability. The model and
selection were locked by SHA-256 before the five test sources were downloaded and processed.
All five were predicted stellar-variability, agreeing with their Gaia proxy labels.

This five-source **single-class** check provides no estimate of overall accuracy, precision,
false-positive rate, calibration or performance on the other classes. The model is uncalibrated,
lacks `other_artifact`, and has not been installed as the desktop classifier. This test is now open
and cannot serve as a fresh independent test. The recorded earlier tests are also not fresh test sets.

The original FITS files, frozen manifest and scripts, model, checkpoints, environment, test report
and `verification.json` are under
`.transiteye-workspace/remediation/variability-expansion-20261102/`. The post-run audit verified all
29 downloaded hashes, unique and disjoint training/test source groups, finite features, the model
hash and reproduction of the five predictions. The manifest records absolute input paths and hashes,
so reproduction elsewhere requires restoring the same inputs or adapting the protocol; an independent
environment has not yet reproduced the study.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  uv run python scripts/expand_variability_cohort.py train \
  .transiteye-workspace/remediation/variability-expansion-20261102 --workers 4
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  uv run python scripts/expand_variability_cohort.py test \
  .transiteye-workspace/remediation/variability-expansion-20261102 --workers 4
```
