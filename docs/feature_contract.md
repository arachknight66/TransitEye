# Shared feature contract (B035–B039)

Status: frozen MVP development policy. Numerical settings in
`configs/features/mvp.yaml` are provisional and require later sensitivity analysis.

One extractor operates on blind-BLS candidate measurements, processed cadence measurements,
and frozen periodograms. It does not receive catalog ephemerides, synthetic injection truth,
labels, match results, split names, or dataset roles. Dataset-specific handling is limited to
resolving already-recorded artifact lineage and copying label/evaluation values to the separate
`labels.parquet` companion table.

Every output column has an explicit role in `features/registry.py`: identity, feature, label,
evaluation, or provenance. Future model matrices must select only entries registered as
`feature`; numeric dtype is never a selection rule. Forbidden catalog truth, synthetic truth,
labels, and dataset shortcuts are rejected when a registry is validated.

Time-domain statistics use valid detrended cadences and the blind candidate period, epoch, and
duration. Lomb–Scargle operates directly on irregular valid timestamps. FFT uses only a
temporary segment-local regular grid: gaps exceeding the configured short-gap allowance start a
new block and are never bridged. Neither branch modifies the processed cadence table.

Undefined quantities remain null. This stage performs no imputation, scaling, clipping learned
across candidates, feature selection, or label-informed transformation. Those operations are
explicitly outside B035–B039 and, if later approved, must be fitted using training data only.
