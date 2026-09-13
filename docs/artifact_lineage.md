# Artifact lineage

`results/index.json` is the canonical documentation entry point. The table below
summarizes major nodes from its referenced project manifest; it intentionally
omits the hundreds of file-level inventory records.

| Artifact | Version/ID | Role |
|---|---|---|
| TOI catalog snapshot | `toi-20260909T175435Z-b65e9b5d5a5c17f0904e` | Frozen catalog facts and dispositions |
| Pilot cohort | `pilot-toi-20260909T175435Z-b65e9b5d5a5c17f0904e-8c1d8d1a1e5581bfe536` | Eight-TIC development cohort |
| MAST product snapshot | `mast-products-cc80ce4b85ed6ab29d84` | Frozen TESS-SPOC discovery result |
| Expansion manifest | `expansion-be59541b9388b685e1f9` | Selected raw products and receipts |
| Scientific dataset | `dataset-4b84e8acaa6f6b334c2f` | Real, unlabeled candidate dataset |
| Demo manifest | `demo-manifest-9459aec7c2e26e406301` | Deterministic injection definitions |
| Demo dataset | `dataset-demo-a74b6aad7c21faf15b0d` | Controlled candidate labels and recovery |
| Scientific features | `features-4e9f6d54b84780a2be42` | Shared features for unlabeled inference |
| Demo features | `features-6278a328b8f624b2759b` | Shared features for controlled modeling |
| Official model | `model-4a58f313bc50b9c3dc41` | Frozen Random Forest, threshold 0.325 |
| Robustness evaluation | `evaluation-e1d75a22b6e9f3e57ca8` | LOTO, recovery, stability, and shift |
| Final evaluation | `final-evaluation-a4d1c707d285fa72a87f` | Calibration, interpretation, scorecards, readiness |
| Reproducibility package | `repro-f43b9410e759001971ae` | Portable manifest and full checksum inventory |

## Major graph

```text
catalog -> cohort -> MAST snapshot -> expansion -> raw -> processed -> blind BLS
                                                        |                |
                                                        |         scientific dataset
                                                        v                |
                                                  demo manifest      scientific features
                                                        |                |
                                                   demo dataset -> demo features -> model
                                                                             |
                                  robustness evaluation -> final evaluation -> repro package
```

The full manifest also records that robustness consumes both feature domains and
that final evaluation consumes the frozen model and robustness evaluation. All
file paths in the portable inventory are relative to the repository root.
