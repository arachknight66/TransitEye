# TransitEye 0.1.0 release notes

TransitEye 0.1.0 is the final academic project release of a reproducible,
candidate-level transit detection and vetting pipeline for TESS light curves.
It combines immutable TESS-SPOC acquisition, transit-preserving preprocessing,
blind BLS candidate generation, shared time/BLS/frequency features, classical
ML, grouped controlled evaluation, and deterministic reporting.

## Capabilities

- Frozen catalog, cohort, product discovery, raw-product, and preprocessing
  provenance.
- Blind BLS candidates with post-detection event and harmonic matching.
- Controlled injections on real TESS substrates without contaminating the
  scientific path.
- A shared 65-feature contract and four classical model baselines.
- TIC-grouped splits, LOTO robustness, group-bootstrap uncertainty, injection
  recovery, calibration, interpretation, scorecards, and domain-shift analysis.
- Portable verification of 613 immutable artifacts and offline report/smoke
  workflows.

## Status and results

- Software pipeline: complete.
- Demo pipeline: complete.
- Scientific readiness: blocked.
- Six-TIC pooled LOTO PR-AUC: approximately 0.974.
- Six-TIC pooled LOTO F1: approximately 0.871.
- Per-TIC PR-AUC: approximately 0.846–1.000.
- Per-TIC F1: approximately 0.625–1.000.
- Controlled BLS recovery: 72.7%–100%, depending on synthetic family.

The official development-demo test has PR-AUC and F1 of 1.0 on only two held-out
TICs. It is not scientific generalization evidence. All 75 scientific candidates
remain unlabeled; 67 violate robust demo support on at least one feature.

## Reproduce locally

```bash
uv sync --all-groups --locked
uv run python scripts/reproduce_project.py --verify
uv run python scripts/reproduce_project.py --report
uv run python scripts/reproduce_project.py --demo-smoke
```

These three reproduction modes work offline and do not retrain or modify frozen
artifacts.

## Frozen identities

- Scientific dataset: `dataset-4b84e8acaa6f6b334c2f`
- Demo dataset: `dataset-demo-a74b6aad7c21faf15b0d`
- Scientific features: `features-4e9f6d54b84780a2be42`
- Demo features: `features-6278a328b8f624b2759b`
- Official model: `model-4a58f313bc50b9c3dc41`
- Official threshold: `0.325`
- Robustness evaluation: `evaluation-e1d75a22b6e9f3e57ca8`
- Final evaluation: `final-evaluation-a4d1c707d285fa72a87f`
- Reproducibility package: `repro-f43b9410e759001971ae`

## Known limitations

The real pilot lacks independent candidate labels and enough TIC groups for
defensible grouped scientific evaluation. Demo-to-scientific feature shift is
substantial, synthetic morphologies are limited, candidates within a TIC are
correlated, and Random Forest scores are not calibrated scientific
probabilities. Three non-blocking Pandas concatenation `FutureWarning`s remain;
silencing them could alter accepted frozen dtypes or checksums.
