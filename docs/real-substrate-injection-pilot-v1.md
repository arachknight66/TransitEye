# Real-substrate injection pilot, v1

This development pilot injects synthetic ideal trapezoids into four real, checksum-verified TESS
light curves. The four sources were selected from the TRAIN partition by the rule frozen in
`configs/real_substrate_injection_pilot.v1.toml`, without consulting labels or inspecting flux.
The selected FITS products, input checksums, split-manifest checksum and product-manifest checksum
are recorded in the external workspace's `selection.json`; `input_audit.json` records preflight.
The original files are unchanged. Cadences, quality flags, errors, units and time standards are
preserved in the injected curves. The synthetic input checksum is distinct from the original FITS
checksum, which is recorded with the original path in provenance.

This is **synthetic injection on real substrates**, not an independent real-data classifier
evaluation. Each source receives four signals: periods 2.0 and 3.5 days crossed with depths 500 and
2,000 ppm, each 2.5 hours long. The unmodified source is also analyzed as a control. The search
returns at most one candidate. A match requires a converged fit, period within 2% of the injected
period, and epoch within 0.5 hours of the injected phase; harmonics do not count. The phase is
deterministically seeded. Both the injection and fitting assume a two-minute exposure, which has
not been independently verified from the products' metadata. Forty bootstrap replicates are used.

Reproduce from the repository root after obtaining the same verified inputs and frozen selection:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run python -m transiteye.evaluation.real_injection_pilot \
  configs/real_substrate_injection_pilot.v1.toml \
  .transiteye-workspace/remediation/r4-real-injection-pilot-v1-reproduction
```

| Injected period | Depth | Top-candidate matches | Trials |
|---:|---:|---:|---:|
| 2.0 days | 500 ppm | 1 | 4 |
| 2.0 days | 2,000 ppm | 1 | 4 |
| 3.5 days | 500 ppm | 1 | 4 |
| 3.5 days | 2,000 ppm | 1 | 4 |

Four of 16 injected trials matched. All four unchanged controls also produced a candidate; those
real stars are not known to be signal-free, so the control count is **not a false-alert rate**.
None of the four control candidates matched any of its four corresponding injected ephemerides
under the frozen matching rule, but existing signals can still compete with or distort an
injection. All 20 analyses completed without a reported fit or trial failure. One source contributes
multiple related trials, so the trials are not independent source-level samples.

Among the four matched injections, median absolute errors were 0.000183 days in period, 36.1 ppm
in depth, and 0.0442 hours in duration. The 68% intervals covered the injected period, depth and
duration in 1/4, 1/4 and 2/4 cases; the 95% intervals covered them in 3/4, 3/4 and 4/4. These
counts are conditional on matching and exclude the other 12 injections; they cannot establish
coverage from four sources and 40 bootstrap replicates.

Every trial and bundle is retained in the external workspace, with `trials.parquet` and
`summary.json` reporting denominators and outcomes. All 20 corrected bundles passed integrity
checks. After correcting the synthetic provenance and the unverified exposure-status field, all
20 trial records matched the earlier run; a same-environment resume reused the earlier checkpoints.
The final bundles explicitly report `fixed_assumption_unverified`. A separate locked virtual
environment on the same Linux host reproduced all 20 trial records and the summary exactly; see
`docs/clean-environment-check-v1.md`. This is not reproduction on a different machine. A larger
predeclared study with source-separated sampling, verified exposure metadata, diverse injections,
calibrated null controls and adequate uncertainty is still required.
