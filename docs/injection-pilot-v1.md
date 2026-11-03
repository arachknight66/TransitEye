# Synthetic injection pilot, v1

This is a small development experiment, separate from all real-data evaluations. It uses generated
Gaussian white-noise light curves with independently sampled, exposure-integrated ideal trapezoid
signals. It does not measure performance on TESS noise, stellar variability, gaps, blending, or
real transits. The fitting model is also trapezoidal, so this is an optimistic in-model check rather
than a physical-model validation. No real source, catalog label, or holdout was used.

The frozen settings are in `configs/injection_pilot.v1.toml`. Each injected period and depth has
five trials, with 500 ppm point noise, a 12-day baseline, 10-minute sampling/exposure and 2.5-hour
events. There are ten separate noise-only controls. A recovery must have a successful fit, a period
within 2% of the injected period, and a transit phase within 0.5 hours. Half- and double-period
aliases do not count as recovery. Each interval uses 40 bootstrap replicates; this is especially
coarse for a 95% interval and far too small a pilot for reliable coverage claims.

Reproduce in the locked environment (output outside Git):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run python -m transiteye.evaluation.injection_pilot \
  configs/injection_pilot.v1.toml \
  .transiteye-workspace/remediation/r4-injection-pilot-v1-reproduction
```

| Injected period | Depth | Recovered | Candidate fits failed |
|---:|---:|---:|---:|
| 2.0 days | 500 ppm | 3 / 5 | 2 |
| 2.0 days | 2,000 ppm | 5 / 5 | 0 |
| 3.5 days | 500 ppm | 0 / 5 | 0 |
| 3.5 days | 2,000 ppm | 4 / 5 | 1 |

Overall, 12 of 20 injections met the matching rule. The five weak 3.5-day signals produced no
candidate at the fixed search threshold. Three candidate fits failed because their optimizer
exhausted its evaluation budget. The ten pure-noise controls produced zero candidates. Zero out
of ten is a small-sample observation under this particular white-noise model, not a validated
real-sky false-alert rate. No trial-level errors or physical-fit failures were recorded.

For the 12 matched cases, median absolute errors were 0.000491 days in period, 59.9 ppm in depth,
and 0.161 hours in duration. The 68% intervals covered the injected period, depth, and duration
in 9/12, 7/12, and 8/12 cases, respectively; the 95% intervals covered them in 12/12, 10/12, and
11/12. These are **conditional on successful recovery**, not coverage across all 20 injections;
the missing eight are explicitly counted in the denominator information in `summary.json`. The
small sample and bootstrap count cannot establish calibration.

`trials.parquet`, per-trial JSON and complete result bundles retain seeds, input checksums, truth,
outcomes, errors and available coverage. `metadata.json` records configuration, code and runtime
identity. Restart checks that identity and reuses verified checkpoints. The first run completed
all trials but could not export the table because some seeds exceeded signed 64-bit range; that
run's checkpoints remain intact. After fixing the table to store unsigned 64-bit seeds, a new run
completed, and all 30 trial checkpoints matched the earlier run exactly in the same environment.
This is a repeatability check, not reproduction in an independent environment.
