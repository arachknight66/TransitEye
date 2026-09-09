# Data dictionary

This document defines the foundation-stage records. It does not define
astronomy-data columns or feature vectors yet.

| Record | Purpose | Core lineage fields |
|---|---|---|
| `RawProductRecord` | One immutable observational product | object, product identity, checksum, source reference |
| `ProcessedCurveRecord` | One future derived curve | source observation, preprocessing configuration, input/output checksums |
| `CandidateRecord` | One future search candidate | observation group, BLS configuration, rank |
| `LabelRecord` | A candidate label and its source | candidate, label state, source reference, policy hash |
| `SplitRecord` | Object-level dataset split assignment | object, dataset version, split-policy hash |
| `CatalogSnapshotRecord` | Frozen external catalog provenance | source, query, retrieval time, raw/derived checksums |

## Identifier concepts

- `object_id`: normalized TIC-style astronomical object identifier, currently
  represented as `tic-<integer>`.
- `observation_id`: `obs-<hash>` of normalized object ID, sector, author,
  cadence, product ID, and raw-product SHA-256.
- `observation_group_id`: `og-<hash>` of the ordered list of observation IDs.
  Ordering is significant because it represents the analysis input order.
- `candidate_id`: `cand-<hash>` of observation-group ID, BLS configuration
  hash, and positive candidate rank.
- `dataset_version`: `dataset-<hash>` of catalog snapshot, raw manifest,
  preprocessing configuration, detection configuration, and label-policy
  hashes.
- `run_fingerprint`: `run-<hash>` of configuration hash, source revision, and
  master seed; it has no timestamp.
- `experiment_id`: `exp-<UTC timestamp>-<run suffix>`; it identifies one
  execution while its run fingerprint identifies equivalent executions.

All hashes are SHA-256 digests of canonical JSON with sorted object keys,
compact separators, and no filesystem paths. Current IDs use 20 hexadecimal
characters (80 bits) after their human-readable prefix. Raw checksums retain
the full 64-character SHA-256 digest.

Canonical construction rules and field validation are implemented in
`transiteye.identifiers` and `transiteye.schemas`.

## TBD additions

- **TBD:** raw FITS metadata columns.
- **TBD:** processed light-curve columns and quality-mask semantics.
- **TBD:** candidate measurement fields.
- **TBD:** feature definitions, units, missing-value policy, and model outputs.
