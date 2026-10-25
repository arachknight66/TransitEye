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
