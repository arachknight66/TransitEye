# Physical-fit exposure-sampling pilot, v1

This is a small development experiment comparing the current physical-model refinement with an
experimental fitter that differs in its exposure integration. It does not change the production
fitter, assess detection, or estimate uncertainty coverage. The frozen inputs are in
`configs/physical_fit_sampling_pilot.v1.toml`.

Eight synthetic cases combine radius ratios 0.03 and 0.1, impact parameters 0.3 and 0.75, and
2- and 30-minute exposures. Each case has two transits, non-overlapping exposure-length cadence,
one independent Gaussian noise realization at 100 ppm, and fixed quadratic limb darkening. The
noise seed is recorded per case. Both methods fit the same noisy flux, starting values and
parameter bounds, with the same robust least-squares loss and evaluation budget. The first calls
the production five-point Batman refinement; the second uses composite Simpson integration with
51 points. A parity test verifies the experimental optimizer against the production fitter when
both are supplied the same five-point forward model.

The synthetic truth is a 1,001-point trapezoidal integration of instantaneous Batman flux; a
501-point calculation checks reference convergence. The largest reference difference was
0.00573 ppm, below the frozen 0.1 ppm threshold. Both truth and fit models share Batman's
instantaneous physics, so this is not independent validation of the physical model. Each pair is
run in an isolated process: a native diagnostic disqualifies both fits in that case, and failed
optimizers remain in the output. Fitted quantities are epoch, period, radius ratio, scaled
semi-major axis and inclination; these are not the pipeline's reported trapezoidal depth or
duration.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run python -m transiteye.evaluation.physical_fit_pilot \
  configs/physical_fit_sampling_pilot.v1.toml \
  .transiteye-workspace/remediation/r4-physical-fit-pilot-v1-reproduction
```

All 16 fits completed without a reported optimizer failure or native warning. For the four
30-minute cases, the median absolute radius-ratio error was 0.00234 for the production model and
0.00139 for the Simpson model; median absolute scaled-semi-major-axis error was 3.51 and 0.572,
respectively. The corresponding medians across the four 2-minute cases were 0.0000825 and
0.0000348 in radius ratio, and 0.0704 and 0.111 in scaled semi-major axis. These are descriptive
statistics on four related settings, with one noise draw each, not estimates of general
performance. Individual outcomes and errors are retained in `fits.parquet`.

Median fitted runtimes on this host were 0.025 and 0.160 seconds for 2-minute exposures, and 0.020
and 0.025 seconds for 30-minute exposures, for production and Simpson respectively. These timings
include optimizer behavior and different cadence counts; they are not a sector benchmark. All
case rows, timing, seeds, native diagnostics, software versions, hashes and a source snapshot are
in `.transiteye-workspace/remediation/r4-physical-fit-pilot-v1/`.

The pilot supports further investigation of exposure integration; it does not establish that the
51-point method converges over arbitrary transit geometries or is suitable for production.
Exact-contact native warnings, broader orbital and noise conditions, repeated realizations,
parameter bias, interval coverage, verified real exposure metadata and operational cost remain to
be addressed.
