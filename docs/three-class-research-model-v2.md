# Expanded three-class research model

This retrains the development-only three-class estimator described in
`docs/three-class-research-model-v1.md` on both rounds of downloaded light curves. It is not a
four-class TransitEye application model: no independently labeled `other_artifact` examples are
included, outputs are uncalibrated, and the estimator is not loaded into the desktop.

The six collections contain 450 entries. Four entries from two sources have conflicting collection
labels; 70 entries were excluded because the source had an existing non-training partition
assignment. Split identities were checked without opening sealed holdout observations or labels.
The 376 remaining source selections and their source-group partitions were frozen before feature
extraction. All 151 sources shared with the prior development experiment kept the same group and
partition; their recomputed feature checkpoints matched exactly.

Input checksums, CPU preprocessing and feature definitions, blind candidate search, and
candidate-to-catalog label checks follow the previous report. Catalog periods and source identities
are **not model features**. In particular, detected transit/eclipsing periods must match the catalog
period or a half/double alias, and transit phases must match the catalog epoch. Gaia variability
labels are source-level classifications and have not been independently reviewed for signal
localization. These choices select a favorable subset and do not measure end-to-end detection.

Of the 376 selected sources, 23 failed candidate-to-catalog matching. The other 353 yielded usable
features: 307 training sources and 46 development-validation sources.

| Class | Training sources | Development-validation sources |
|---|---:|---:|
| Transit-like | 86 | 15 |
| Eclipse-like | 117 | 17 |
| Stellar variability | 104 | 14 |

The standalone estimator is a 200-tree random forest with depth limit 8, balanced class weights and
seed 42. Five-fold source-grouped cross-validation on the training sources gave macro-F1 0.908. On
the 46 selected development-validation sources, the confusion matrix was:

| Actual / predicted | Transit-like | Eclipse-like | Stellar variability |
|---|---:|---:|---:|
| Transit-like | 15 | 0 | 0 |
| Eclipse-like | 0 | 17 | 0 |
| Stellar variability | 0 | 1 | 13 |

The resulting validation macro-F1 was 0.978. **Neither number is an independent performance
estimate.** The validation sample is small, was drawn from previously inspected and deliberately
screened collections, and excludes failed or mismatched candidates; labels are provisional and
source-confounding has not been independently audited. The counts and errors are retained in the
manifest and feature results, rather than hidden from the full selected-sample denominator.

The estimator, per-source checkpoints, `manifest.json`, `feature-results.parquet`, `summary.json`,
`partition-audit.json`, `verification.json`, and a backend source archive are under
`.transiteye-workspace/remediation/three-class-research-v3/`. Verification checked 376 unique
results, feature finiteness, model and backend hashes, group separation, prior partition identity,
reloaded predictions and agreement with the stored validation confusion matrix. The estimator is a
standalone scikit-learn joblib artifact; never deserialize a joblib file from an untrusted source.

To reproduce, restore the recorded backend snapshot and six input selections, then run into a new
external output directory:

```bash
uv run python -m transiteye.evaluation.three_class_training prepare \
  .transiteye-workspace/remediation/three-class-expanded-reproduction
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  uv run python -m transiteye.evaluation.three_class_training run \
  .transiteye-workspace/remediation/three-class-expanded-reproduction --workers 4
```

An application-ready model still requires representative, reviewed examples of the fourth class,
source-local label audit, calibration and abstention, and a frozen independent evaluation.
