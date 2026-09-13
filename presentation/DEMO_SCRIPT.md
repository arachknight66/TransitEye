# TransitEye 5–8 minute demonstration script

## 1. Problem — 30 seconds

“TESS light curves contain transit-like signals, stellar variability,
instrumental systematics, and aliases. TransitEye turns blind BLS peaks into
traceable candidate events and vets them with a shared feature and classical-ML
pipeline.”

Open `README.md` and point to the project status: software complete, demo
complete, scientific readiness blocked.

## 2. Architecture — 45 seconds

Show `results/figures/figure_1_system_architecture.png`. Explain that real TESS
and controlled demo inputs join before the shared feature/model infrastructure,
while their label sources remain separate. A candidate is the ML unit; a TIC is
the split unit.

## 3. Scientific path — 30 seconds

Show the real branch in `results/figures/figure_2_pipeline_flow.png`: frozen
TESS-SPOC products → preprocessing → blind BLS → candidates → features → frozen
scores. Its 75 candidates are unlabeled. The model scoring every row above
threshold is an observed transfer behavior, not a scientific classification.

## 4. Why demo mode exists — 40 seconds

Explain that the pilot lacks independent real candidate labels. Demo mode
injects controlled signals into real TESS substrates without changing the
scientific path. It provides known event truth only after blind candidate
generation.

## 5. Run the compact demo — 60 seconds

```bash
uv run python scripts/reproduce_project.py --demo-smoke
```

Point out that the command is offline. It preprocesses a deterministic 500-point
box-transit fixture, runs existing BLS/peak extraction, and performs frozen
inference on two demo feature rows. The expected 2.5-day period is recovered at
about 2.5025 days.

## 6. Injection and blind recovery — 45 seconds

Show `results/figures/figure_6_injection_recovery.png`. Separate three bars:
BLS recovery, classifier correctness conditional on recovery, and end-to-end
success. Recovery ranges from 72.7% to 100% by synthetic family.

## 7. Features and model — 40 seconds

Show `results/figures/figure_9_feature_importance.png`. The frozen 65-feature
schema includes time, BLS, Lomb–Scargle, and FFT groups. The selected Random
Forest and threshold 0.325 were frozen from validation; scientific scores never
selected the model.

## 8. Evaluation — 50 seconds

Show `results/figures/figure_5_grouped_robustness.png`. Lead with six-TIC LOTO:
pooled PR-AUC about 0.974 and F1 about 0.871. Per-TIC F1 reaches as low as 0.625.
Mention the perfect official test only with its two-held-out-TIC,
development-demo context.

## 9. Scientific limitations — 50 seconds

Show `results/figures/figure_8_domain_shift.png`, then
`figure_10_scientific_readiness.png`. Sixty-seven of 75 real candidates violate
robust demo support on at least one feature. Seven of eight readiness criteria
fail. This is why no real performance or discovery claim is made.

## 10. Reproducibility — 35 seconds

```bash
uv run python scripts/reproduce_project.py --verify
```

Explain that the portable manifest verifies 613 immutable artifacts and their
lineage. `results/index.json` points to every dataset, feature matrix, model,
evaluation, table, figure, and reproduction package needed for review.
