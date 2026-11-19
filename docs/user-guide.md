# TransitEye User & Operator Guide

## 1. Quickstart & Installation
TransitEye requires Python 3.11 and uses `uv` for reproducible environment management:

```bash
# Clone the repository
git clone https://github.com/USICT/TransitEye.git
cd TransitEye

# Sync dependencies locked
uv sync --locked

# Launch the interactive desktop interface
uv run transiteye-desktop
```

## 2. Command-Line Workflows
The backend is completely headless and runs without GUI dependencies.

### Finding and fetching archive light curves

```bash
uv run python -m transiteye lookup-target "TIC 307210830"
uv run python -m transiteye lookup-target "KIC 11904151"
uv run python -m transiteye fetch-target "KIC 11904151" --config path/to/config.toml
uv run python -m transiteye inspect-target-set path/to/fetched/manifest.json
```

Supply `--catalog TIC` or `--catalog KIC` for an unqualified number. The fetch command freezes
its discovered products in a manifest and verifies FITS target identity before accepting each
download. A failed download or mismatched file stops the fetch; rerun the same command after
correcting the archive problem. This workflow currently fetches data for review. Kepler analysis,
combined observations, and the desktop lookup interface are subsequent development slices.
`inspect-target-set` checks that every product is present and reports its archive URI, checksum,
provider, cadence estimate, and BJD-TDB coverage, together with cadence, overlap-winner,
selected/excluded, and per-reason counts. Cadence width is a sampling proxy where no exposure is
measured. The observation record retains separate overlap and analysis masks and reasons for
invalid values, quality flags, overlap exclusions, and unresolved overlap. TESS and Kepler use
different default quality masks. Cadence width does not establish exposure duration. With at least
three distinct timestamps, overlap width is the lower median positive spacing; with two, `TIMEDEL`
is used only if within 20% of observed spacing; singleton `TIMEDEL` is unknown. Intersecting spans
with an unknown width remain unresolved.

The backend also has an explicit BLS maximum trial-period budget and a standalone joint trapezoid
fit that shares event timing and shape while estimating depth independently for each observation.
These components are not yet connected to an end-to-end combined-analysis command or the desktop;
`inspect-target-set` remains an inventory validator, not a signal analysis. Existing single-file
behavior and result-bundle schemas are unchanged, and v1 manifests remain readable.

A separate exposure resolver accepts a user-supplied duration or computes `INT_TIME * NUM_FRM` as
a frame-sum boxcar approximation. It reports exposure as missing when those header values are
unavailable, and does not use Kepler `TIMEDEL` or aggregate `EXPOSURE`/`LIVETIME` as an exposure
measurement. This resolver is not yet connected to combined analysis; its focused tests pass
(7 tests).

`prepare_observation_set` independently normalizes and detrends each observation, retains selected
rows' original cadence indices, and uses one common BJD-TDB origin for sorted numerical arrays. It
rejects unresolved overlap (6 focused tests pass). It does not run BLS or the joint trapezoid fit;
those standalone components are not yet wired into an end-to-end combined-analysis service or the
desktop.

### Analyzing a Single TESS Target
```bash
uv run python -m transiteye analyze-fits path/to/tess_lc.fits \
    --tpf path/to/tess_tp.fits \
    --device auto \
    --max-candidates 3 \
    --snr-threshold 7.1
```

### Running Sector Batch Processing
Process an entire TESS sector (directory or manifest):
```bash
uv run python -m transiteye run-batch manifests/sector_benchmark.v1.json \
    --workers 4 \
    --device cuda \
    --max-candidates 3
```

### Checking Job Status
Query the SQLite WAL state database for ongoing or completed jobs:
```bash
uv run python -m transiteye job-status <job_id>
```

### Running Locked Evaluation Protocol
Execute the single-shot evaluation suite and produce report artifacts:
```bash
uv run python -m transiteye run-locked-evaluation
```

## 3. Interactive Desktop Application
Launch the desktop UI with:
```bash
uv run transiteye-desktop
```

### Desktop Features
1. **Target Inspector**:
   - Synchronized pan/zoom between raw light curves (with trend overlay) and detrended curves.
   - Interactive BLS periodogram highlighting fundamental period and harmonic aliases ($P/2, 2P$).
   - Phase-folded transit profile with real-time binned median curves, fitted limb-darkened trapezoid/batman curves, and residual plots.
2. **Classification & Vetting Cards**:
   - Real-time calibrated probability bar chart with automatic abstention alerts.
   - Astrometric centroid shift display and companion TPF difference image viewer.
3. **Sector Batch Runner**:
   - Manage multi-worker batch processing with live target status, throughput (targets/sec), and pause/cancel controls.
   - Double-click any completed target to load its diagnostic bundle into the inspector immediately.
4. **Export Options**:
   - Export result bundles as versioned directories, Parquet tables, or publication-grade 300 DPI Matplotlib summary sheets.

## 4. GPU Acceleration Configuration
TransitEye automatically detects NVIDIA GPUs with CUDA capability. To force CPU or GPU:
- CLI flag: `--device cuda` or `--device cpu`.
- Desktop dropdown: Select "cuda" or "cpu" in the main toolbar.
- The system automatically caps VRAM usage and falls back to CPU if GPU memory is constrained.

## 5. Interpreting Machine Learning Outputs

### 5.1 Classification Schema in `result.json`
Every analyzed candidate includes a structured `classification` record:
```json
{
  "predicted_class": "transit_like",
  "probabilities": {
    "transit_like": 0.9412,
    "eclipse_like": 0.0381,
    "stellar_variability": 0.0124,
    "other_artifact": 0.0083
  },
  "source_assessment": "TARGET_CONSISTENT",
  "abstained": false,
  "model_version": "transiteye_morphology_v1"
}
```

### 5.2 Key Fields & Scientific Meaning
- **`predicted_class`**: One of `transit_like`, `eclipse_like`, `stellar_variability`, `other_artifact`, or `null` (if abstained).
- **`probabilities`**: Platt-calibrated posterior probabilities summing to 1.0. These represent frequentist empirical reliability (e.g., across candidates with $P \approx 0.90$, approximately 90% are true positives).
- **`source_assessment`**: Independent spatial vetting assessment (`TARGET_CONSISTENT`, `BLEND_SUSPECTED`, `UNRESOLVED`, or `NOT_ASSESSED`). Morphology and source location are completely decoupled.
- **`abstained`**: Boolean flag. Set to `true` when confidence is $< 0.40$, margin is $< 0.10$, or feature values fall out-of-distribution.

### 5.3 Programmatic Classification in Python
```python
from pathlib import Path
from transiteye.classification.models import ModelBundle

# Load the frozen model bundle
bundle = ModelBundle.load(Path("models/morphology_v1"))

# Classify an extracted feature dictionary
classification = bundle.predict(features)
print(f"Predicted class: {classification.predicted_class}")
print(f"Calibrated probabilities: {classification.probabilities}")
print(f"Abstained: {classification.abstained}")
```
