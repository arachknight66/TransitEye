# TransitEye manifests

This directory stores small, immutable, reviewable definitions of inputs and partitions. It must not
contain downloaded FITS products, feature matrices, model artifacts, or evaluation outputs.

Phase 1 provides schemas and validation code for:

- Product manifests: exact MAST product inventories, query parameters, checksums, and local names.
- Label snapshots: source-level morphology labels with public evidence, review state, ambiguity, and
  catalog snapshot date.
- Split manifests: one source group per partition, including all sectors and derivatives.

Create a real pilot only after label evidence has been independently reviewed. The repository does not
ship a scientifically usable label cohort or a locked test set. Doing so without that review would
misrepresent catalog entries as validated model ground truth.
