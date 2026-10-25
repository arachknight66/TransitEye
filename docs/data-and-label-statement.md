# TransitEye Data and Label Statement

## 1. Provenance and Partition Isolation
TransitEye enforces strict astrophysical target isolation across all dataset splits. Astronomical targets (stars) observed across multiple TESS sectors (e.g. Continuous Viewing Zones) share a unique `source_group_id` (e.g. `tic-150428135`).

### Frozen Partitions
All targets belong to exactly one of four non-overlapping partitions:
1. `train`: Used solely for training feature extractors, synthetic augmentations, and classifier base models.
2. `calibration`: Dedicated held-out partition used strictly for fitting Platt scaling and temperature parameters.
3. `selection`: Used exclusively for tuning decision policy thresholds ($\tau_{\text{conf}}, \tau_{\text{margin}}$) and out-of-distribution (OOD) feature envelopes.
4. `test`: Locked single-shot evaluation set. Unopened until evaluation protocol is finalized.

## 2. Prohibition of Information Leakage
The feature extraction registry enforces automated AST and runtime audits:
- **Prohibited Terms**: Features derived from catalog metadata (`tic_id`, `ra`, `dec`, `tmag`, `teff`, `stellar_radius`, `known_ephemeris`) are strictly forbidden from the feature matrix.
- **Unlabeled Targets**: Targets absent from exoplanet archives are treated as **unlabeled**, never as negative training examples.
- **Controlled Synthetic Augmentation**: Synthetic transit injections (`batman` models with realistic limb darkening and SNR distributions) are restricted exclusively to `train` targets.

## 3. Catalog Ingestion & Provenance Tracking
Curated labels are snapshotted in `manifests/pilot_labels.v1.json` with immutable audit trails:
- `label_source`: Authority proving the ground truth (e.g., `exoplanet_archive`, `tess_eb_catalog`, `variable_star_catalog`, `manual_review`).
- `evidence_uri`: Direct HTTP/HTTPS link to published archive entry or refereed publication.
- `review_status`: Explicit human review flag (`verified`, `reviewed`, `provisional`).
- `ambiguous`: Boolean flag identifying unresolved blends or low-SNR catalog classifications.
