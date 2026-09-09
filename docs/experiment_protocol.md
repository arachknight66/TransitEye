# Experiment protocol

## Confirmed design decisions

- Each experiment will use a resolved configuration, deterministic
  configuration hash, source revision, dataset version, master seed, derived
  seeds, and environment metadata.
- Equivalent scientific runs will be recognized by a deterministic run
  fingerprint; individual executions will also receive timestamped experiment
  identifiers.
- Candidate classification will be compared with BLS-only recovery and with the
  combined BLS-to-ML cascade.
- Planned primary metrics include precision, recall, F1, ROC-AUC, PR-AUC,
  confusion matrices, false-positive rate, and false-negative rate.
- All model selection occurs without using the locked test partition.

## Planned experiment sequence

1. Synthetic pipeline checks.
2. BLS-only baseline.
3. Time/BLS-feature classical baselines.
4. Frequency-feature ablations.
5. ELM/KELM comparison.
6. Robustness experiments.
7. Optional advanced model only after the preceding work is complete.

## TBD decisions

- **TBD:** exact model hyperparameter search spaces.
- **TBD:** primary optimization metric and operating threshold policy.
- **TBD:** confidence-interval procedure and number of resamples.
- **TBD:** robustness grid and perturbation levels.
- **TBD:** final reporting template and thesis figure selection.
