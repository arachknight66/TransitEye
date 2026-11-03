# Larger three-class research cohort, 2026-11-02

This development experiment adds 100 source-selected TESS light curves to the prior research
cohorts. It is not a four-class application model. Labels are catalog proxies, source localization
has not been independently adjudicated, and unadjudicated `other_artifact` candidates are not
used as training labels. No observations or labels from the separate sealed eclipsing-binary
holdout were opened.

The selection was frozen before downloads from the previously collected NASA Exoplanet Archive,
Gaia and MAST inventories. It excludes prior selected TIC/Gaia identities, previous experiment
sources (including all 18 of its test sources), existing split identities and previously staged
products. Known Gaia/TIC aliases were excluded, although incomplete mappings cannot rule out
all aliases. Source conflicts were removed. The selected 100 consist of 38 transit-like,
60 eclipse-like and two stellar-variability proxies. In deterministic Gaia-hash order, six, six
and one per respective class were reserved for testing; the remainder were used for training.
The imbalance reflects the limited unused candidates in these inventories and is a serious
limitation; these samples do not represent the sky's class frequencies.

All 100 selected light-curve files were downloaded from the [MAST public TESS archive on AWS](https://registry.opendata.aws/mast-tess/),
passed FITS checksums and TIC/sector validation, and were retained unmodified. The saved SHA-256
was computed from each download rather than obtained as an independent archive checksum. The
existing quality-mask preparation, detrending, blind 0.3–5-day BLS search, fitting and feature
extraction were applied. Catalog period and epoch checks selected candidates consistent with the
proxy label; the catalog information was not used as model features. Four training candidates and
one test candidate did not match their catalog events and remain recorded as excluded failures.

The 200-tree random forest (depth limit 8, balanced class weights, seed 42) was trained on 431
usable source-level vectors: 307 from the earlier research training set, 41 from the preceding
expansion's training partition, and 83 newly extracted examples. Class support was 131
transit-like, 181 eclipse-like and 119 stellar-variability. The earlier 18-source test set was not
used for training. The estimator and selected manifest were locked by SHA-256 before the 13 newly
reserved test observations were opened.

Twelve of the 13 new test sources produced usable, catalog-matched candidates. On those 12, the
predictions matched the proxy classes:

| Catalog-proxy class | Transit-like | Eclipse-like | Stellar variability |
| --- | ---: | ---: | ---: |
| Transit-like | 6 | 0 | 0 |
| Eclipse-like | 0 | 5 | 0 |
| Stellar variability | 0 | 0 | 1 |

The observed 12/12 agreement is conditional on successful extraction and matching. The excluded
13th test case was an eclipse-like catalog mismatch. The test is small and selected, with only
one variability example, so it cannot establish general accuracy, calibration or performance on
unmatched data. The catalog proxies and potential source confusion add uncertainty. A transit-like
classification does not confirm a planet. The model is uncalibrated and lacks the fourth class;
it has not replaced the desktop model.

The frozen manifest, 100 original FITS files, per-source checkpoints, model, test report,
source snapshot and environment record are under
`.transiteye-workspace/remediation/three-class-larger-20261102/`. `verification.json` records
checks of all 100 file hashes, unique source groups, training/test separation, finite features,
the model hash and reproduction of test predictions. The test is now opened and cannot serve
as a fresh independent test. Because the manifest includes absolute paths and hashes of prior
training inputs, exact reproduction requires restoring those inputs and paths; another environment
has not independently reproduced this experiment.

To rerun against the same frozen workspace and its checkpointed inputs:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  uv run python scripts/expand_research_cohort_large.py train \
  .transiteye-workspace/remediation/three-class-larger-20261102 --workers 4
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  uv run python scripts/expand_research_cohort_large.py test \
  .transiteye-workspace/remediation/three-class-larger-20261102 --workers 4
```
