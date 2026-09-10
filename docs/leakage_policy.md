# Leakage prevention policy

## Confirmed design decisions

- Dataset partitions will be grouped by `object_id`; all sectors, products, and
  candidates belonging to one object must remain in one partition.
- Duplicate raw products will be detected through their portable product
  identity and checksum.
- Label-source fields, catalog dispositions, catalog ephemerides, target names,
  and split assignment are forbidden model features.
- Data-dependent transforms such as imputation, scaling, feature selection,
  calibration, and threshold selection must be fitted without test data.
- Robustness perturbations and synthetic augmentations must be applied only
  after partitioning; augmented descendants remain with their parent object.
- The final test partition is locked before model-selection decisions.

## Dataset-boundary checks (B031--B034)

- One TIC is the indivisible split group. Observation groups, products, raw
  checksums, processed products, candidates, and catalog events inherit that
  TIC's assignment.
- Candidate detection measurements and catalog evaluation metadata use
  separate schemas. Catalog period, epoch, duration, disposition, and match
  fields are forbidden from a future model-input namespace.
- Unmatched candidates are explicitly unlabeled, never gold negative.
- Contradictory matched gold dispositions produce an explicit ambiguous,
  non-training state.
- Dataset validation rejects cross-split identities, duplicate candidates or
  events, inconsistent candidate/event lineage, and invalid label semantics.

## TBD decisions

- **TBD:** final train/validation/test proportions and grouped-stratification
  method.
- **TBD:** handling of closely related astronomical systems beyond a shared TIC
  identifier.
- **TBD:** exact policy for optional stellar metadata experiments.
