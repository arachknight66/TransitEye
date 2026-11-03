# Experimental primary-BLS null test, v1

This development experiment tests an IID permutation null for the **primary BLS search only**.
It is not connected to the application's search-significance field. It does not repeat production
detrending, screening, iterative candidate extraction or characterization, and it does not
establish a valid null model for TESS light curves.

The frozen settings are in `configs/search_null_pilot.v1.toml`. For each trial, the statistic is
the maximum power over the same primary BLS period, duration and phase grid on the observed series.
The whole primary grid is rerun for each of 99 random permutations. The one-sided Monte Carlo
p-value is `(1 + number of permuted maxima at least as large as observed) / 100`, so its minimum
possible value is 0.01. The explicit IID/exchangeability assumption is essential: shuffling destroys
time correlation. See the [Astropy BLS description](https://docs.astropy.org/en/stable/timeseries/bls.html)
for the period/duration/phase search and the [SciPy permutation-test documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html)
for the randomized-test counting convention and its assumptions.

The test generates 300 samples at 30-minute spacing with 500 ppm Gaussian noise and searches
periods 0.5–2.5 days. It uses 100 independent white-noise trials and 40 correlated AR(1) trials
with coefficient 0.8. A white-noise pilot pass required no errors, at most 10/100 alerts at p ≤ 0.05,
and an empirical one-sided excess below the predeclared 95% DKW bound. These are limited diagnostic
criteria, not acceptance criteria for real-data significance.

| Null data | Alerts at p ≤ 0.05 | Valid trials |
|---|---:|---:|
| IID white noise | 5 | 100 / 100 |
| Correlated AR(1) noise, incorrectly shuffled as IID | 40 | 40 / 40 |

The white-noise empirical excess was 0.040 versus the 0.136 bound, so the limited IID pilot passed.
The correlated-noise result shows severe failure when the exchangeability assumption is violated;
it is not a calibrated real-sky false-alert estimate. No trial errors occurred. One hundred trials,
one cadence pattern and one search grid cannot establish general calibration.

Reproduce from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run python -m transiteye.evaluation.search_null_pilot \
  configs/search_null_pilot.v1.toml \
  .transiteye-workspace/remediation/r4-search-null-pilot-v1-reproduction
```

The external workspace stores all trial seeds, input checksums, p-values, the frozen configuration,
runtime/code identity and `summary.json`; a matching source snapshot is stored separately. The
production field remains unavailable until a correlation-preserving null, the complete search
procedure, false-alert behavior on suitable independent data and computational cost have all been
validated.
