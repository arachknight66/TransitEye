# TransitEye

TransitEye is a reproducible, CUDA-accelerated (with deterministic CPU fallback) scientific system and PyQt6 desktop workstation for detecting, characterizing, classifying, and vetting signals in noisy TESS light curves.

**Scientific remediation in progress:** Earlier synthetic-cohort and legacy locked-evaluation
results are invalid because their evaluation used label-conditioned features and non-independent
references. The historical PDF and release claims must not be used as evidence of performance.
The real-catalog v1 evaluation is documented separately in the [model card](docs/model-card.md);
its labels are catalog-derived proxies. New results do not currently report empirical search
significance, and classification is unavailable unless an explicitly selected model is supplied.

## Key Capabilities

- **Quality-Aware Two-Pass Detrending**: Iterative duration-aware Tukey biweight filtering with in-transit masking (Pass 2) and CUDA GPU acceleration (`MAX_BUF = 2048`).
- **Blind Detection & Characterization**: Box Least-Squares (BLS), Lomb-Scargle variability analysis, trapezoid and `batman` fits, red-noise SNR, and moving-block bootstrap intervals. Exposure handling and interval coverage are under review.
- **Optional Morphology Classification**: An explicitly selected model provides four morphology classes with abstention; model performance and calibration depend on its training and validation data.
- **Independent Two-Tier Spatial Vetting**: Decoupled astrometric centroid shift analysis (`MOM_CENTR1/2`, `CROWDSAP`) and companion Target Pixel File (`_tp.fits`) difference-image + multi-aperture growth vetting.
- **Desktop Workstation & Batch Engine**: Responsive PyQt6 + PyQtGraph interface (`transiteye-desktop`) communicating with the headless backend via versioned result bundles (`result.json`, `arrays.npz`, `candidates.parquet`), `.teye.json` sessions, PNG/PDF/SVG vetting sheets, and SQLite WAL batch jobs.

## Quick Start

```bash
uv sync --locked --extra desktop
QT_QPA_PLATFORM=offscreen uv run --no-sync pytest
uv run --no-sync python -m transiteye --help
```

Analyze a local SPOC or TESS-SPOC light curve end to end:

```bash
uv run --no-sync transiteye analyze-fits path/to/lightcurve.fits --device auto
```

When the per-cadence exposure is independently known, pass `--exposure-seconds` to individual or
batch analysis. Otherwise the result explicitly flags the historical two-minute fit assumption.
The supplied duration is an effective boxcar approximation; it does not model separate frame readout
gaps. The historical v1 classifier was not independently evaluated with the corrected physical-model
integration. The trapezoid uses an exact boxcar integral; the physical model uses five sub-exposure
samples. Physical-model convergence and interval coverage remain to be verified before making
scientific accuracy claims.

The default period search stops at half the observed time baseline. The current duration grid also
caps its longest tested transit at one quarter of the minimum searched period; with the default
0.3-day minimum, this is 1.8 hours. A no-candidate result cannot rule out signals outside those
search limits. Both limitations need scientific validation before changing default search policy.
The desktop's “Search periods” controls let you choose a different range before analysis; these
settings are saved with a desktop session and used by new desktop batch jobs. Desktop single-target
and batch runs use the backend's default SNR threshold of 6.0. A longer search may contain too few
events to distinguish the true period from aliases, and does not by itself confirm a planet.

Source vetting is a preliminary assessment, not confirmation of where a signal originates. New
results mark missing centroid/crowding data and a zero-signal difference image as unresolved; the
SAP/PDCSAP depth ratio is unavailable when both products were not independently measured. Older
bundles may contain default values and should be reanalyzed before relying on source assessments.

Run a batch in the configured workspace using
`uv run --no-sync transiteye run-batch path/to/fits --device cpu --workers 2 --config path/to/config.toml`.
The same explicit `--model-dir` and `--model-calibrator-sha256` options are available for batch
analysis. Manifest jobs read products acquired into the configured workspace; a declared SHA-256
is checked before analysis. Each job prints its ID. `job-status ID`, `pause-job ID`,
`cancel-job ID`, `retry-job ID`, and `resume-job ID` accept the same `--config` workspace option.
Pause drains active targets; cancel terminates them. Retry requeues failed targets on a stopped job;
run resume afterwards. A target timeout includes worker startup and defaults to 180 seconds, adjustable
with `run-batch --target-timeout-seconds`. The desktop also accepts a saved job ID for these controls
and loads saved target records in pages, including failures. `job-status ID --tasks` provides the
same records on the CLI, with `--limit` and `--offset` for pagination. Workers watch their coordinator
and a job cannot resume while an earlier worker still holds its lease. After an abrupt shutdown,
retry shortly if that lease is still held. See `PLAN.md` before using batch results for a scientific
benchmark.

To opt in to the locally available historical real-catalog v1 classifier, provide the bundle and
the calibrator checksum recorded by its frozen protocol:

```bash
uv run --no-sync transiteye analyze-fits path/to/lightcurve.fits --device cpu \
  --model-dir .transiteye-workspace/models/morphology_real_catalog_v1 \
  --model-calibrator-sha256 59338074204c8545bad72711a82e45713a29f811ecd78cb3a59194c0d4a5c9a7
```

Only load model bundles from sources you trust: serialized Python models can execute code. The
reference model is stored in the external workspace and is not included in a clean checkout.
In the desktop, select **Use historical morphology model (experimental)** before starting a
single-target analysis or a new batch. This explicitly selects that same local v1 bundle and its
recorded calibrator checksum; leaving it unchecked runs without classification. Resume uses the
saved batch settings. Re-run an existing result to obtain classification. The checkbox is not
saved in a desktop session and must be selected again for a new run. The historical model's
performance and calibration are not validated for the current pipeline, and an abstention remains
possible. A transit-like classification does not confirm a planet.

## Experimental PyTorch sequence model

The repository includes a separate, from-scratch candidate-sequence CNN experiment. Install all
optional dependencies with `uv sync --locked --all-extras`. Restore the frozen development products
with `python3 scripts/download_training_cohort.py --resume-portal --workers 16`. Run the fresh
baseline before the sequence model:

```bash
uv run --no-sync python scripts/train_real_cohort.py \
  --data-dir .transiteye-workspace/data/real_catalog_cohort_v3_dev \
  --products .transiteye-workspace/data/real_catalog_cohort_v3_dev_products.v1.json \
  --labels manifests/real_catalog_cohort.v3.labels.json \
  --splits manifests/real_catalog_cohort.v3.splits.json \
  --output-dir .transiteye-workspace/models/morphology_fresh_baseline_v1 \
  --model-id morphology_fresh_baseline_v1 --workers 4
uv run --no-sync python scripts/train_sequence_model.py \
  --output-dir .transiteye-workspace/models/candidate_sequence_cnn_v3 \
  --workers 4 --maximum-epochs 100 --batch-size 32 --device cuda
```

The frozen protocols are recorded in `configs/model_baseline_development.v1.toml` and
`configs/sequence_model_development.v3.toml`; their checksums are included in model reports.

This is a research artifact only. It is not loaded by single-target or batch analysis. Its training
labels are catalog-derived source-level proxies, not independently adjudicated candidate-level
morphology; its selection metrics do not establish independent accuracy. The test partition stays
sealed, and the model must not be presented as planetary confirmation.

### Newly labeled candidate-sequence experiment

The separately approved 2026-11-14 experiment trains a binary candidate-morphology proxy model from
scratch. To rebuild its cleaned manifest and resume candidate extraction/training, run:

```bash
uv run --no-sync python scripts/prepare_labeled_sequence_cohort.py
uv run --no-sync python scripts/train_labeled_candidate_sequence.py \
  --workers 4 --maximum-epochs 100 --batch-size 32 --device cuda
```

Its configuration is `configs/labeled_candidate_sequence.v1.toml`; artifacts and per-source
checkpoints are written outside Git to
`.transiteye-workspace/models/labeled_candidate_sequence_v1/`. The training command does not open the
631-source fresh holdout. After model lock, its one-time final evaluation was run with:

```bash
uv run --no-sync python scripts/evaluate_labeled_candidate_sequence_holdout.py \
  --workers 4 --device cuda
```

The run matched 3,940 of 6,255 development sources. On the 386-source selection partition, CNN
macro-F1 was 0.9254 versus 0.8158 for the same-candidate random-forest baseline; these are model
selection scores. The final holdout evaluation matched 364 of 631 sources (355 eclipse-like and only
9 transit-like). CNN macro-F1/balanced accuracy were 0.7429/0.7693; the baseline scored
0.6135/0.8410. With only nine transit-like examples, per-class and aggregate scores have substantial
uncertainty, and ordinary accuracy is dominated by the eclipse-like majority. The holdout is now
spent and must not be reused for tuning. Catalog records remain imperfect proxy labels, and neither
output confirms an exoplanet.

Launch the interactive desktop application:

```bash
uv run --no-sync transiteye-desktop
```

The desktop's **Graph Guide** explains the light curves, periodograms, folded profile and
residuals, including the difference between previewing a period and fitting a candidate.

For a sample walkthrough, press `Ctrl+Shift+D` in the desktop or start it with
`uv run --no-sync transiteye-desktop --demo`. This analyzes the locally downloaded TIC 198356533
sector 86 FITS file in `.transiteye-workspace/data/real_catalog_cohort_v1/`, using a 1–20 day
search. The file is not included in a clean checkout. To use another local TESS light curve,
launch with `--demo --demo-file /path/to/light-curve.fits`; this uses the usual automatic search
range. The sample runs through the normal backend and retains its result and input provenance;
results and any ambiguity or analysis failure depend on the actual data. Starting the sample also
selects the historical experimental four-class model, with its selection visible in the desktop.
The model files must be present and match the frozen historical checksums; otherwise the demo
reports an error. Its performance and calibration on the current pipeline remain unvalidated.

Run clean-room release verification:

```bash
./scripts/verify_clean_release.sh
```

Large downloaded products and generated workspaces are stored in `.transiteye-workspace/` and ignored by Git. Read `AGENTS.md`, `PROJECT.md`, `PLAN.md`, and `PROGRESS.md` before extending the project.
