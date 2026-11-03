# Development exposure-sampling study, v1

This numerical experiment compares Batman's finite-exposure forward model with a separately
integrated version of the same instantaneous Batman model. It does not establish the accuracy of
the underlying physical model, fitted parameters, or uncertainty intervals. The frozen grid in
`configs/exposure_convergence.v1.toml` was recorded before the run. Its 1 ppm comparison tolerance
and 0.1 ppm reference-convergence tolerance are development criteria, not universal astronomical
requirements.

Reproduce in the locked environment, writing outside Git:

```bash
uv run python -m transiteye.evaluation.exposure_convergence \
  configs/exposure_convergence.v1.toml \
  .transiteye-workspace/remediation/r4-exposure-v1-reproduction
```

The command returns a nonzero status when a case is unusable or exceeds tolerance. `cases.parquet`
contains every case and its diagnostics; `summary.json` records the config hash, backend hash,
versions and counts. Each geometry runs in a subprocess so native-library messages cannot be
silently ignored. The reference is a composite trapezoidal average over 2,001 instantaneous
samples, checked against 1,001 samples. The run uses fixed times and no random inputs.

On Python 3.11.16, NumPy 2.4.6 and batman-package 2.5.3, 9 of 12 geometries were usable.
The other three, all radius ratio 0.1 and impact parameter 0.9, emitted Batman's native
`ellpic_bulirsch` convergence warning. All five factors for those geometries are marked unusable;
the numerical values are retained for investigation, not counted as passes. Across the usable
geometries the two reference grids differed by at most 0.00147 ppm. Results are maximum absolute
flux errors on the frozen observation-time grid:

| Exposure samples | Usable cases within 1 ppm | Largest error (ppm) |
|---:|---:|---:|
| 5 | 0 / 9 | 601.89 |
| 11 | 1 / 9 | 241.52 |
| 21 | 2 / 9 | 122.19 |
| 51 | 2 / 9 | 49.10 |
| 101 | 5 / 9 | 24.63 |

The current production fit uses five samples, so this experiment fails the stated development
criterion; increasing the factor to 101 is not sufficient on this grid. This result does not by
itself quantify bias in fitted depth, duration or period, or justify a replacement sampling policy.
Further work must investigate the native warnings, choose and verify an efficient integration
method, and measure fitted-parameter and uncertainty behavior independently.
