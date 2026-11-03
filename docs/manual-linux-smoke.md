# Linux desktop workflow check

This is a manual operational check, not a scientific evaluation. Record the date, TransitEye
version or source commit, machine, and any failed step when carrying it out. Do not mark the R3
gate passed until every applicable step succeeds and the evidence is recorded in `PROGRESS.md`.

1. From the repository root, install and launch the current checkout:

   ```bash
   uv sync --locked --extra desktop
   uv run --no-sync transiteye-desktop
   ```

2. Open the development-partition FITS file
   `.transiteye-workspace/data/real_catalog_cohort_v1/tess2024326142117-s0086-0000000198356533-0283-s_lc.fits`.
   Set the device to CPU and **Search periods** to 1.00–20.00 days, then select **Analyze Target**.
   Check that analysis completes, the searched range is visible, a candidate near 9.426 days is
   shown, and its two-event flag and aliases are visible. This candidate is an ambiguous
   half-period alias; it is not a confirmed measurement of TOI-1437 b's period.
3. Select **Save Session**, choose a writable filename, and record its full path. Select **Open
   Session** and reopen that file. Verify the same result, candidate, source assessment, and period
   settings return. A session without a result must say so; a missing or malformed referenced
   bundle must produce an error rather than silently claiming success.
4. Export the selected candidate's vetting sheet as PNG. Check that the desktop remains responsive,
   the file opens, the candidate and aliases are shown, and unavailable values are identified as
   unavailable.
5. On **Sector Batch Runner**, select the same FITS file, CPU, and one worker. Start the job, record
   the displayed job ID, and check that it finishes with one completed target and no failures. Open
   its result, then use **Load job** to inspect the persisted record. The batch uses the inspector's
   selected period range. Its candidate and analysis settings should match the single-target run.
6. To exercise pause, resume, and failure reporting, prepare a temporary folder outside Git. From
   the repository root run:

   ```bash
   transiteye_smoke_dir=$(mktemp -d)
   transiteye_smoke_fits="$PWD/.transiteye-workspace/data/real_catalog_cohort_v1/tess2024326142117-s0086-0000000198356533-0283-s_lc.fits"
   ln -s "$transiteye_smoke_fits" "$transiteye_smoke_dir/one_lc.fits"
   ln -s "$transiteye_smoke_fits" "$transiteye_smoke_dir/two_lc.fits"
   printf 'invalid FITS input\n' > "$transiteye_smoke_dir/broken_lc.fits"
   printf 'Batch test directory: %s\n' "$transiteye_smoke_dir"
   ```

   Enter that folder in the batch source field, use one worker, and start a new job. Request pause
   while it is running, wait for active work to drain, and resume it. If it completes before the
   pause request, repeat with a fresh workspace or a larger development-only input set. Check that
   the job accounts for every input and that the invalid FITS is shown as a failed target with an
   error. Retry the stopped job's failed targets and confirm the failure remains visible. These
   duplicate inputs are only for operational testing and must not be used as independent data.
7. Record all results, including any failure or interruption. Separately run the headless suite,
   lint, and CLI help commands in `AGENTS.md`; automated checks do not replace the manual steps.

An absent classifier or unavailable search significance is expected until independently validated
artifacts are explicitly selected or implemented. Neither a transit-like signal nor a successful
smoke test confirms an exoplanet.
