# 100-target development benchmark, v1

This is an operational benchmark, not an independent scientific evaluation. The frozen inputs are
`configs/benchmark_development.v1.toml` and `manifests/benchmark_development.v1.json`. The manifest
selects 100 distinct TRAIN source groups by numeric TIC and the newest locally downloaded sector,
without using catalog labels. All 100 downloaded inputs passed their manifest SHA-256 checks before
execution. The external workspace retains the selection audit, preflight record, SQLite state,
result bundles, report, verification record and a matching source snapshot.

Reproduce from the repository root with the same verified FITS files and an unused output directory:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run python -m transiteye.execution.benchmark_development \
  configs/benchmark_development.v1.toml manifests/benchmark_development.v1.json \
  .transiteye-workspace/remediation/r4-benchmark-development-v1-reproduction
```

On an AMD Ryzen 5 7235HS Linux machine with four CPU workers, the job execution took 193.33 seconds,
excluding 0.38 seconds of input preflight. All 100 targets completed with zero target failures and
zero cache hits; 98 analyses returned candidates and two reported no significant signal. The
throughput was 0.517 freshly computed completed targets per second. This is the rate for these
specific inputs and settings, not a general sector-scale throughput claim.

The sampled maximum **sum of process RSS** was 1,491.6 MiB. Sampling every 0.25 seconds observed
up to six processes, including the coordinator, workers and process-management child. The sum can
count shared pages multiple times, and polling can miss brief peaks, so it is not an exact
unique-memory high-water mark. The recorded output-directory footprint before writing the report
was 82.97 MiB; it excludes the downloaded inputs and separately stored source snapshots.

The run requested a pause after ten finished targets, drained active work and reached a persisted
pause with 13 completed and zero failed, then resumed to completion. The independent post-run audit
found 100 unique task records, one attempt per target, matching input/code fingerprints and valid
bundle integrity for all 100. It found no trapezoid or physical-fit failure flags; 82 candidates
carried the explicit unvalidated physical-quadrature warning. This particular 100-target run tested
pause/resume, not coordinator SIGKILL. A separate R3 integration test exercised SIGKILL recovery.

Settings include one candidate per source, SNR threshold 6, 40 bootstrap replicates, search periods
0.3–5 days, no classifier and the unverified two-minute exposure assumption. The current duration
grid is also limited by the minimum searched period, so this run is not evidence of sensitivity to
all long events. The versioned report records effective settings, package versions, thread settings,
source/config hashes, target failures and analysis-status counts. The first version of this report
was generated before physical-fit flags were added to automatic benchmark summaries; the separate
`verification.json` contains that flag audit. Later reports include those counts automatically.
