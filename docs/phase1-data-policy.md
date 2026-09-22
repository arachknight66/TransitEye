# Phase 1 data and label policy

## Labels

TransitEye trains on source-level labels only when each record has a public evidence URI, stable
identifier, snapshot date, review state, and sector coverage. Supported morphology labels are
transit-like, eclipse-like, stellar variability, and other/artifact.

Planet-candidate catalog presence alone does not prove a transit-like label. Absence from a catalog is
never an other/artifact label. Ambiguous records remain marked ambiguous and are excluded from training
until a documented review resolves their use.

## Partitioning

Every target sector, candidate, injected derivative, and noise substrate from an astrophysical source
belongs to one source group and one partition. The required partitions are train, calibration,
selection, and locked test. A sector-shift cohort is kept separate for robustness evaluation.

The test labels stay unopened while models, thresholds, feature sets, and preprocessing choices are
selected. Synthetic injection outcomes are reported separately from real labeled-data metrics.

## Acquisition

Product manifests freeze MAST discovery output before download. Downloaded files are checksummed and
stored in the configured external workspace. Interrupted files use a `.part` suffix and are never
treated as valid outputs. Every unavailable or failed product must remain in an acquisition record.
