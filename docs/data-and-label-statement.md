# TransitEye Data and Label Statement

> **Audit correction (2026-11-01):** Historical catalog-derived labels are proxies, not
> independently adjudicated signal morphology. The old builder automatically populated review
> strings, including `verified`; those strings alone do not prove a human reviewed a source or its
> selected signal. In the previously opened v1 TRAIN partition, 150 eclipse-like labels are
> deep-false-positive proxies and 142 variability labels come from membership of the Asteroseismic
> Target List. The [ATL catalog description](https://cdsarc.cds.unistra.fr/viz-bin/ReadMe/J/ApJS/241/12?format=html&tex=true)
> describes targets selected by predicted oscillation detectability, not confirmed variability in
> each observed light curve. [TOI dispositions](https://exoplanetarchive.ipac.caltech.edu/docs/API_TOI_columns.html)
> distinguish false positives from other dispositions but do not establish that every deep false
> positive is an eclipsing binary. Old manifests are retained as historical artifacts, not relabeled.
> The v1 test was already opened; the separate EB holdout remains sealed pending a new protocol.

## 1. Provenance and Partition Isolation
TransitEye enforces strict astrophysical target isolation across all dataset splits. Astronomical targets (stars) observed across multiple TESS sectors (e.g. Continuous Viewing Zones) share a unique `source_group_id` (e.g. `tic-150428135`).

### Frozen Partitions
All targets belong to exactly one of four non-overlapping partitions:
1. `train`: Used solely for training feature extractors, synthetic augmentations, and classifier base models.
2. `calibration`: Dedicated held-out partition used strictly for fitting Platt scaling and temperature parameters.
3. `selection`: Used exclusively for tuning decision policy thresholds ($\tau_{\text{conf}}, \tau_{\text{margin}}$) and out-of-distribution (OOD) feature envelopes.
4. `test`: Reserved for a predeclared independent evaluation. The v1 test was already opened and
   cannot be treated as a fresh independent test of the current pipeline.

## 2. Prohibition of Information Leakage
The feature extraction registry enforces automated AST and runtime audits:
- **Prohibited Terms**: Catalog labels, target identifiers, known ephemerides and split metadata must
  not enter model features. FITS stellar temperature can influence limb-darkening and a derived
  fit-comparison feature; it is not a classifier column, but that indirect influence is an input.
- **Unlabeled Targets**: Targets absent from exoplanet archives are treated as **unlabeled**, never as negative training examples.
- **Controlled Synthetic Augmentation**: Synthetic transit injections (`batman` models with realistic limb darkening and SNR distributions) are restricted exclusively to `train` targets.

## 3. Catalog Ingestion & Provenance Tracking
Curated labels are snapshotted in `manifests/pilot_labels.v1.json` with immutable audit trails:
- `label_source`: Claimed origin of the evidence; catalog membership or a label alone is not proof
  of source-local ground-truth morphology.
- `evidence_uri`: Direct HTTP/HTTPS link to published archive entry or refereed publication.
- `review_status`: Stored review string (`verified`, `reviewed`, `provisional`). In historical
  generated cohorts it was assigned automatically and must not be read as evidence of human review.
- `ambiguous`: Boolean flag identifying unresolved blends or low-SNR catalog classifications.
