# Three-class research model trained from the downloaded collections

This is a **development experiment**, not the four-class application model. It recognizes only
`transit_like`, `eclipse_like` and `stellar_variability`; there are no `other_artifact` examples.
Its random-forest outputs are uncalibrated and cannot be loaded as a valid four-class TransitEye
model. The desktop was not configured to use it.

The three user-requested collections contain 225 entries. Four entries from two sources had
conflicting collection labels and were excluded. Another 70 entries were excluded because the
source had an existing non-training assignment in at least one prior split manifest. Split
**identities only** were checked to protect source grouping; no held-out observations or holdout
labels were opened for this training run. The remaining 151 entries were frozen before feature
extraction: 139 development-training sources and 12 development-validation sources. All 12
validation sources happened to be eclipse-like; that set cannot measure multiclass performance.

Each input was checked against its recorded SHA-256 and TIC. Features were computed on CPU from
quality-masked TESS flux using the existing preprocessing, detrending, blind BLS search and fit,
with a fixed search range of 0.3–5 days and detection threshold 1.5. For the transit-like and
eclipse-like classes, detected candidate periods had to match the catalog period or a half/double
alias within 2%; for transit-like sources, the event phase also had to match the published epoch
within the larger of 0.5 hours or half the candidate duration. These catalog values were used for
**label selection only**, never as model features. Stellar-variability labels are provisional
Gaia source classifications; the procedure does not establish that the strongest detected event
originates on that source. The catalog-based selection and earlier screen introduce selection
bias, and neither class label nor sky location was independently adjudicated.

Thirteen of 151 selected sources did not pass candidate-to-catalog matching; they remain listed
with reasons. The model used 129 training feature vectors (30 transit-like, 56 eclipse-like and
43 stellar-variability). Nine eclipse-like sources remained in the one-class development-validation
partition. There were no other feature-extraction error categories in this run.

The estimator is a 200-tree, balanced random forest with depth limit 8 and seed 42. Five-fold
grouped cross-validation on the 129 training sources produced macro-F1 0.820 and accuracy 0.837.
These are **exploratory development figures**, not an independent evaluation or a reliable estimate
of real-sky performance: the sources were previously selected and inspected, catalog labels are
proxies, and only successful matched candidates enter the metric. The 9-source one-class validation
result is not a multiclass performance measure. All exclusions and class counts appear in the
external `manifest.json`, `feature-results.parquet`, and `summary.json`.

The artifact and inputs are stored under
`.transiteye-workspace/remediation/three-class-research-v2/`. `research-estimator.joblib` is a
standalone scikit-learn estimator, with class order in its `classes_` attribute. It is not an
application `ModelBundle` and must not be loaded from an untrusted source: joblib deserialization
can execute code. `verification.json` records the model hash, feature checks, counts, package
versions, and source-group overlap check; a backend source snapshot is retained alongside it.

To repeat after restoring the exact source snapshot and input collections, use a new output folder:

```bash
uv run python -m transiteye.evaluation.three_class_training prepare \
  .transiteye-workspace/remediation/three-class-research-reproduction
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  uv run python -m transiteye.evaluation.three_class_training run \
  .transiteye-workspace/remediation/three-class-research-reproduction --workers 4
```

A real application model still needs source-local label review, representative examples of the
fourth class, calibrated probabilities and abstention, and a frozen independent evaluation before
any performance or exoplanet-confirmation claim.
