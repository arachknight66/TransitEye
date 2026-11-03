# Development Simpson-integration study, v1

This frozen follow-up compares composite Simpson integration of instantaneous Batman flux against
the same 2,001-point trapezoidal reference used in the earlier study. The configuration,
`configs/exposure_simpson.v1.toml`, includes impact parameters of 0.899999, 0.9 and 0.900001 to
test the region around the geometry that produced native warnings. It uses the same development
tolerances: at most 0.1 ppm reference-grid difference and 1 ppm model-grid difference. These
thresholds are not a general claim of observational adequacy.

```bash
uv run python -m transiteye.evaluation.exposure_convergence \
  configs/exposure_simpson.v1.toml \
  .transiteye-workspace/remediation/r4-exposure-simpson-v1-reproduction
```

The command exits nonzero because some cases fail or are unusable. On batman-package 2.5.3,
NumPy 2.4.6, and SciPy 1.17.1, 21 geometries were usable; three radius-ratio 0.1, impact-parameter
0.9 geometries emitted native `ellpic_bulirsch` convergence warnings and are excluded from the
usable-case counts. Every geometry and warning remains in `cases.parquet`; the configuration and
code fingerprints are saved in `summary.json`.

| Simpson samples | Usable cases within 1 ppm | Largest error (ppm) |
|---:|---:|---:|
| 5 | 7 / 21 | 107.46 |
| 11 | 16 / 21 | 9.46 |
| 21 | 20 / 21 | 1.97 |
| 51 | 21 / 21 | 0.136 |
| 101 | 21 / 21 | 0.022 |

The largest difference between the 1,001- and 2,001-point references was 0.00147 ppm on usable
cases. The reference and candidate methods use the same underlying instantaneous transit model,
so agreement is evidence about numerical exposure integration, not independent physical accuracy.
This grid is limited and includes no eccentric orbits or limb-darkening variation. It does not
establish parameter accuracy, uncertainty coverage, or behavior on arbitrary observations.

An exploratory single-process timing on 20,000 timestamps found 51-point Simpson integration about
ten times slower per forward-model evaluation than Batman's current five-point averaging, with a
higher process peak memory. This is not the frozen 100-target benchmark. The production fitter
still uses five points: the native warning, broader grid, optimizer behavior and runtime tradeoff
need investigation before a replacement is validated.
