# TransitEye manifests

This directory stores small, immutable, reviewable definitions of inputs and partitions. It must not
contain downloaded FITS products, feature matrices, model artifacts, or evaluation outputs.

Phase 1 provides schemas and validation code for:

- Product manifests: exact MAST product inventories, query parameters, checksums, and local names.
- Label snapshots: source-level morphology labels with public evidence, review state, ambiguity, and
  catalog snapshot date.
- Split manifests: one source group per partition, including all sectors and derivatives.

The frozen Phase 1 pilot cohort consists of:

- `pilot_labels.v1.json`: 68 curated source-level morphology labels (17 per class across `transit_like`,
  `eclipse_like`, `stellar_variability`, and `other_artifact`, with 4 explicit ambiguous audit cases),
  each with public evidence HTTP(S) URIs, sectors, and review status.
- `pilot_splits.v1.json`: Source-grouped partition manifest assigning each source group to `train` (36),
  `calibration` (8), `selection` (8), `test` (8), or `sector_shift` (8), ensuring zero source leakage.
- `pilot_products.v1.json`: Immutable MAST discovery product inventory for all 68 targets across their
  observed sectors (78 FITS products).
