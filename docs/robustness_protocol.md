# B046–B051 robustness protocol

Status: frozen development-demo policy. This is diagnostic analysis, not model tuning or
scientific performance evaluation.

## Confirmed decisions

- The official model is `model-4a58f313bc50b9c3dc41`; its Random Forest, 65-feature
  contract, threshold `0.325`, split, and locked-test metrics remain immutable.
- Robustness uses leave-one-TIC-out refits over the six original train and validation TICs.
  The two official test TICs are excluded. Every fold refits median imputation and the
  frozen Random Forest policy on the other five TICs. The frozen threshold is used because
  nested threshold selection with six groups would be unstable.
- Uncertainty uses 1,000 deterministic TIC bootstrap replicates of out-of-fold predictions.
  Candidate rows are never resampled independently.
- Injection analysis reuses the frozen D001–D008 grid on development TICs. BLS recovery and
  classifier correctness conditional on recovery are reported separately.
- Feature-removal replicas retrain only on the official training TICs and are evaluated on
  validation. Missing-feature and ±0.5% candidate-period stresses use the unchanged frozen
  model. None participates in selection.
- Domain shift compares official gold training features with all unlabeled scientific
  features using medians, IQRs, missingness, range/support violations, and KS distance.
  Scientific labels and scientific performance metrics are neither required nor produced.
- Error categorization is evaluation-only. Synthetic truth is joined after prediction and
  never enters transformations or model features.

## Limitations and TBD

- Intervals from six TICs are coarse and are not population-level confidence intervals.
- The frozen demo grid is compact rather than a full factorial sensitivity surface.
- A domain discriminator is not implemented: eight paired source TICs are too few for a
  credible grouped diagnostic benchmark. Reconsider after cohort expansion (TBD).
- Calibration transfer and scientific correctness cannot be established without an
  independently labeled scientific candidate cohort (TBD).
