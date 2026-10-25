# TransitEye

TransitEye is a reproducible, CUDA-accelerated (with deterministic CPU fallback) scientific system and PyQt6 desktop workstation for detecting, characterizing, classifying, and vetting signals in noisy TESS light curves.

## Key Capabilities

- **Quality-Aware Two-Pass Detrending**: Iterative duration-aware Tukey biweight filtering with in-transit masking (Pass 2) and CUDA GPU acceleration (`MAX_BUF = 2048`).
- **Blind Detection & Characterization**: Box Least-Squares (BLS) dip search, Lomb-Scargle rotational variability analysis, exposure-integrated trapezoid and `batman` limb-darkened transit fits, Pont red-noise SNR, and moving-block bootstrap 68%/95% confidence intervals.
- **Calibrated Morphology Classification**: 25 leak-free features across 6 scientific families (`transit_like`, `eclipse_like`, `stellar_variability`, `other_artifact`) with Platt probability calibration and explicit abstention.
- **Independent Two-Tier Spatial Vetting**: Decoupled astrometric centroid shift analysis (`MOM_CENTR1/2`, `CROWDSAP`) and companion Target Pixel File (`_tp.fits`) difference-image + multi-aperture growth vetting.
- **Desktop Workstation & Batch Engine**: Responsive PyQt6 + PyQtGraph interface (`transiteye-desktop`) communicating with the headless backend via versioned result bundles (`result.json`, `arrays.npz`, `candidates.parquet`), `.teye.json` sessions, PNG/PDF/SVG vetting sheets, and SQLite WAL batch jobs.

## Quick Start

```bash
uv sync --locked --all-extras
QT_QPA_PLATFORM=offscreen uv run pytest
uv run python -m transiteye --help
```

Analyze a local SPOC or TESS-SPOC light curve end to end:

```bash
uv run transiteye analyze-fits path/to/lightcurve.fits --device auto
```

Launch the interactive desktop application:

```bash
uv run transiteye-desktop
```

Run clean-room release verification:

```bash
./scripts/verify_clean_release.sh
```

Large downloaded products and generated workspaces are stored in `.transiteye-workspace/` and ignored by Git. Read `AGENTS.md`, `PROJECT.md`, `PLAN.md`, and `PROGRESS.md` before extending the project.
