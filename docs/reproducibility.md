# Reproducibility

TransitEye treats accepted scientific products as immutable artifacts. The
canonical package is `repro-f43b9410e759001971ae`, referenced through
`results/index.json`.

## Identity chain

Deterministic identities bind scientifically meaningful inputs and policies:

- scientific dataset: `dataset-4b84e8acaa6f6b334c2f`
- demo dataset: `dataset-demo-a74b6aad7c21faf15b0d`
- scientific features: `features-4e9f6d54b84780a2be42`
- demo features: `features-6278a328b8f624b2759b`
- official model: `model-4a58f313bc50b9c3dc41`
- robustness evaluation: `evaluation-e1d75a22b6e9f3e57ca8`
- final evaluation: `final-evaluation-a4d1c707d285fa72a87f`
- reproduction package: `repro-f43b9410e759001971ae`

The official threshold is `0.325`.

## Verification

The project manifest records major lineage nodes. The artifact inventory records
relative path, SHA-256, category, type, parent IDs, policy identity, and frozen
status for every registered artifact. Verification checks existence and checksum
for all entries and checks selected upstream identity links. It never silently
ignores a missing frozen file.

```bash
uv run python scripts/reproduce_project.py --verify
```

## Portable deterministic identity

Identity metadata includes content fingerprints, parent versions, and declared
policies that can change a scientific result. Paths are relative to the project
root. Moving the repository to another filesystem leaves the reproduction
version unchanged.

Non-identity provenance records the Python and package versions, dependency
versions, OS/platform, and Git state. Hostname, username, absolute path, and
execution timestamp do not influence scientific identity. They can describe an
execution environment without changing what the artifact means.

## Offline reproduction modes

`--verify`, `--report`, and `--demo-smoke` work offline. Report generation reads
frozen machine-readable artifacts and emits deterministic CSV, PNG, PDF, and
JSON outputs. The optional `--full-replay` reuses local pipeline modules and is
more expensive; acquisition remains outside that command.
