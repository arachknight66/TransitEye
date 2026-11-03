# Three-class research expansion, 2026-11-02

This is a development experiment, not a validated four-class application classifier. It uses
catalog-derived proxy labels for transit-like, eclipse-like and stellar-variability sources. The
`other_artifact` candidates remain unadjudicated and were not used as labels. No existing sealed
holdout observations or labels were opened.

## Frozen selection and data

The frozen manifest selected 20 new sources per class from the existing NASA Exoplanet Archive,
Gaia and MAST discovery inventories: 14 train and six test per class, ordered by a deterministic
hash of Gaia DR2 identity. Prior selected TIC/Gaia identities, all known split identities, and
previously staged products were excluded; cross-class conflicts were discarded. Available Gaia
cross-identifications were used, but unmapped aliases cannot be ruled out. The inventories were
preselected and public, and catalog classes do not establish signal location or independently
adjudicated morphology. This is a small, selected development test, not a representative survey.

The manifest, its SHA-256, source hashes, input files, checksums, processing checkpoints, model
and reports are in `.transiteye-workspace/remediation/three-class-expansion-20261102-v2/`.
The script snapshot is `frozen-script.py`; `environment.json` records package versions. One
abandoned preliminary run is retained separately in `three-class-expansion-20261102/`; no test
observations were opened by it and it produced no model. The corrected protocol was frozen before
these test observations were downloaded. The final model and manifest were checksum-locked before
test extraction. This experiment's 18-source test is now open and cannot be reused as a fresh
independent test; the separate pre-existing eclipsing-binary holdout remains sealed.

All 60 selected new sources yielded downloaded FITS files that passed FITS integrity and TIC/sector
checks. Their SHA-256 values were computed from the downloaded bytes, not independently supplied
by the archive. Original files were retained. The existing reader, quality-mask preparation,
two-pass detrending, blind 0.3–5 day BLS search and feature extraction were applied. Catalog
periods (and transit epochs for planet hosts) were used only to check whether the detected candidate
matched its label; catalog values and IDs were not model features. One new training source failed
this matching check and was excluded. Every result and its failure reason remains in checkpoints.

## Training and test

A 200-tree random forest with depth limit 8, balanced class weights and seed 42 was fitted using
307 previously extracted development-training examples and 41 new usable training sources. The
training class counts were 99 transit-like, 131 eclipse-like and 118 stellar-variability. The
prior feature extraction used the same recorded backend code fingerprint. No old validation
examples were included in training.

All 18 reserved test sources yielded usable, catalog-matched candidates. Their predictions were:

| Catalog-proxy label | Transit-like | Eclipse-like | Stellar variability |
| --- | ---: | ---: | ---: |
| Transit-like | 6 | 0 | 0 |
| Eclipse-like | 0 | 6 | 0 |
| Stellar variability | 0 | 0 | 6 |

This is 18/18 agreement and a macro F1 of 1.0 on these 18 **successfully detected and matched**
examples. It does not establish real-sky accuracy, calibration, performance on unmatched or
no-candidate targets, source localization, or the absent fourth class. This small selected set and
proxy labels can make the result optimistic. The model is uncalibrated and is not compatible with
the desktop's four-class `ModelBundle`; it was not substituted into the application.

An independent post-run audit verified the recorded SHA-256 of each downloaded file, 60 unique
TIC and Gaia source groups, no group overlap between the usable training examples and test,
finite usable features, the locked model checksum, and reproduction of all saved predictions.
`verification.json`, `model-lock.json` and `test-report.json` contain the audit and per-source
records. The downloaded FITS internal checksums were verified during extraction.

To reproduce in the same repository and workspace, restore the frozen script and exact hashed
inputs (the manifest stores absolute paths), then run the train and test commands below. Existing
checkpoints permit resumption. Running the test again reproduces an already-open evaluation; it is
not a new independent test. Use a new frozen protocol and untouched sources for a future test.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  uv run python scripts/expand_research_cohort.py train \
  .transiteye-workspace/remediation/three-class-expansion-20261102-v2 --workers 4
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  uv run python scripts/expand_research_cohort.py test \
  .transiteye-workspace/remediation/three-class-expansion-20261102-v2 --workers 4
```

Data sources: [NASA Exoplanet Archive TAP](https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html),
[Gaia DR3 variability catalog](https://gaia.aip.de/metadata/gaiadr3/vari_classifier_result/), and
[MAST public TESS data on AWS](https://registry.opendata.aws/mast-tess/).
