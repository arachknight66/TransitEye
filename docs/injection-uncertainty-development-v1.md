# Synthetic injection and interval-coverage study, v1

This frozen development study evaluates the current pipeline on ideal, exposure-integrated
trapezoids in independently generated Gaussian white noise. It uses no real sources or held-out
labels and is separate from real-data evaluation. The injection and fit shapes are both
trapezoidal, making this an optimistic in-model experiment rather than validation on real TESS
noise, stellar variability, gaps, blending, or physical transit shapes.

The frozen configuration is `configs/injection_uncertainty_development.v1.toml`: 60 injections
(15 each for two periods and two depths), 20 independent noise-only controls, 12-day baselines,
10-minute cadence and exposure, 500 ppm point noise, and 100 requested bootstrap refits per
candidate. Recovery requires a successful candidate fit, period within 2% of truth, and epoch
within 0.5 hours modulo the injected period; half- and double-period aliases do not count.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run python -m transiteye.evaluation.injection_pilot \
  configs/injection_uncertainty_development.v1.toml \
  .transiteye-workspace/remediation/r4-injection-uncertainty-development-v1-reproduction
```

| Period | Depth | Recovered / injected |
|---:|---:|---:|
| 2.0 days | 500 ppm | 9 / 15 |
| 2.0 days | 2,000 ppm | 15 / 15 |
| 3.5 days | 500 ppm | 5 / 15 |
| 3.5 days | 2,000 ppm | 13 / 15 |

Overall 42/60 injections met the recovery rule; 11 candidate trapezoid fits failed and two
optional physical fits failed. One harmonic candidate was recorded. The 20/20 usable noise-only
controls produced zero candidates. This observation is limited by the 20-control sample and the
white-noise assumption; it is not a calibrated real-sky false-alert rate. There were no
trial-level errors. A trial can complete while its candidate fit fails, so completed trial counts
must not be interpreted as successful fits.

The following coverage is conditional on the **42 matched recoveries**, not all 60 injections:

| Parameter | 68% interval contained truth | 95% interval contained truth | Median absolute error |
|---|---:|---:|---:|
| Period | 27 / 42 | 37 / 42 | 0.000818 days |
| Depth | 25 / 42 | 34 / 42 | 51.0 ppm |
| Duration | 28 / 42 | 38 / 42 | 0.108 hours |

All three intervals were available on each of the 42 matched recoveries. These counts do not
establish calibration: the sample is small, selection on recovery changes the population, and
100 bootstrap draws give a coarse estimate of tail quantiles. More importantly, the current
bootstrap silently skips failed refits and exposes an interval once at least ten succeed; it does
not record the successful-refit count. Therefore 100 is the requested number, not evidence that
all 100 refits succeeded. Until that diagnostic and an adequate success criterion are implemented,
these interval results should be treated as exploratory.

All 80 trial records and bundles, a table, summary, frozen settings, backend fingerprint and
source snapshot are stored at `.transiteye-workspace/remediation/r4-injection-uncertainty-development-v1/`.
An independent reconciliation there found 80 unique trials, 80 integrity-valid bundles, matching
configuration/backend hashes, and summary counts matching the table. Resuming from checkpoints
completed in the same environment. This is not reproduction on a separate machine or evidence
about the untested conditions above.
