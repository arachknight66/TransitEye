# TransitEye

TransitEye is a reproducible, CPU-first scientific backend for analysing noisy TESS light curves.
The current implementation is Phase 1: typed scientific contracts, FITS validation, reproducible MAST
product manifests, and leakage-safe label/split tooling. Detection, classification, sector execution,
and the PyQt desktop application are planned but not implemented yet.

## Start here

```bash
uv sync --locked
uv run pytest
uv run python -m transiteye --help
```

Inspect a local SPOC or TESS-SPOC light curve without preprocessing it:

```bash
uv run transiteye inspect-fits path/to/lightcurve.fits
```

Freeze available MAST light-curve products for a TIC target, then acquire that immutable manifest:

```bash
uv run transiteye discover-manifest TIC_ID --output manifests/tic-TIC_ID.json
uv run transiteye acquire-manifest manifests/tic-TIC_ID.json --config configs/phase1.example.toml
```

Large downloaded products are stored in the configured workspace and are ignored by Git. Read
`AGENTS.md`, `PROJECT.md`, `PLAN.md`, and `PROGRESS.md` before extending the project.
