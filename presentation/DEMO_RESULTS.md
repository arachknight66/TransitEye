# TransitEye demonstration results

## Pipeline at a glance

- Scientific path: 8 TICs, 15 TESS-SPOC products, 75 unlabeled candidates.
- Controlled demo: 8 source TICs and 375 candidates (70 positive, 60 negative,
  245 unlabeled).
- Frozen schema: 65 model features across time, BLS, Lomb–Scargle, and FFT.
- Compared models: Logistic Regression, Random Forest, RBF SVM, and ELM.
- Selected model: Random Forest `model-4a58f313bc50b9c3dc41`, threshold 0.325.

## Controlled performance

Six-development-TIC LOTO gives pooled PR-AUC 0.974 and F1 0.871. Per-TIC
PR-AUC ranges 0.846–1.000 and F1 ranges 0.625–1.000. The official locked test is
perfect on PR-AUC and F1, but contains only two held-out TICs and is
development-demo evidence.

| Synthetic family | BLS recovery | Conditional correctness | End-to-end success |
|---|---:|---:|---:|
| Eclipsing binary | 100.0% | 100.0% | 100.0% |
| Easy planet-like | 100.0% | 90.9% | 90.9% |
| Medium planet-like | 81.8% | 100.0% | 81.8% |
| Sinusoidal variability | 72.7% | 100.0% | 72.7% |

These values describe controlled injected events, not real-population
classification.

## Transfer and readiness

All 75 scientific candidates remain unlabeled. The demo-trained frozen model
scores all 75 above threshold, and 67 have at least one robust-support violation.
The score distribution and feature support therefore indicate substantial
demo-to-scientific transfer risk.

```text
software_pipeline_status = complete
demo_pipeline_status = complete
scientific_readiness = blocked
```

The next scientific requirement is a larger independently labeled real cohort
with both classes across enough TIC groups for grouped recovery and classifier
evaluation. All presentation sources are indexed by `results/index.json`; the
portable reproducibility version is `repro-f43b9410e759001971ae`.
