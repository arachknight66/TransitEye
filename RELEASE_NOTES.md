# TransitEye v0.1.0 Release Notes

TransitEye v0.1.0 is a complete, scientifically rigorous system for detecting, characterizing, classifying, and vetting signals in noisy TESS light curves.

## Key Features & Highlights
- **Decoupled Architecture**: 100% headless scientific core under `src/transiteye/` (verified via AST analysis to contain zero GUI dependencies) paired with a responsive PyQt6 / PyQtGraph desktop application under `desktop/transiteye_desktop/`.
- **Quality-Aware Two-Tier Detrending**: Iterative duration-aware Tukey biweight filtering with transit masking (Pass 2) to prevent transit distortion. Features hardware-accelerated CUDA execution with automatic CPU fallback.
- **Robust Period & Transit Modeling**: Box Least-Squares (BLS) frequency grid with harmonic alias tracking ($P/2, 2P$), Lomb-Scargle rotational variability analysis, Mandel \& Agol limb-darkened transit models via `batman` with Claret (2018) quadratic limb-darkening coefficient estimation based on stellar parameters ($T_{\rm eff}$, $\log g$), Pont red-noise SNR estimation, and moving-block bootstrap parameter confidence intervals.
- **Audited 25-Feature Matrix**: Extracted across six families (Morphology, Consistency, Eclipse Physics, Variability, Noise/Quality, Spatial Evidence). Audited against metadata leakage.
- **Calibrated Gradient-Boosted Classifier with Abstention**: Trained under Grouped Stratified 5-Fold CV on 72 training observations with candidate model selection (`HistGradientBoosting` selected over Random Forest and Logistic Regression); Platt-calibrated on a held-out calibration partition. Automatically abstains ($\tau_{\text{conf}} < 0.40$, $\tau_{\text{margin}} < 0.10$) on ambiguous signals.
- **Independent Two-Tier Blend Vetting**: Tier 1 astrometric centroid shifts ($\sigma_{\Delta c}$) from light curves; Tier 2 difference imaging and concentric aperture growth from companion Target Pixel Files (`_tp.fits`). Reports physical arcsecond offsets (`centroid_shift_arcsec`, `difference_image_offset_arcsec` with $21.0''/\text{px}$ TESS scale) and plots visual target crosshairs and centroid markers. Crowded fields ($CROWDSAP < 0.85$) default to `UNRESOLVED` rather than false blend guesses.
- **Multi-Sector Chronological Stitching**: `stitch_light_curves` seamlessly merges multiple TESS observation sectors for a single target, preserving timestamps, cadences, centroids, and metadata provenance.
- **High-Throughput Batch Resilience**: Bounded `ProcessPoolExecutor` with configurable per-target worker execution timeouts (`target_timeout_seconds`), CUDA stream scheduling, SQLite WAL cache hit indexing, and crash-resilient replay.
- **Interactive Desktop Workflow**: PyQt6 / PyQtGraph GUI featuring interactive periodogram click-to-pick period selection with real-time folded light-curve updates, fitted trapezoid/batman overlays, TPF/difference image overlays, candidate classification & vetting cards, session saving/loading (`.teye.json`), and diagnostic PDF vetting sheets.
- **Deployment & Developer Experience**:
  - Full GitHub Actions CI/CD workflow (`.github/workflows/ci.yml`) enforcing locked dependencies, Ruff linting, headless pytest suite, and clean release checks.
  - Production-ready `Dockerfile` with headless Qt and Python 3.11 environment.
  - Executable tutorials in `examples/`: `quickstart_tutorial.py` (single-target pipeline walkthrough) and `batch_pipeline_demo.py` (multi-worker batch pipeline with cache replay).
- **Scientific Deliverables**:
  - Full locked evaluation protocol executed on the sealed `test` split: accuracy 100.0%, macro-$F_1$ 1.000, ECE 0.151, Brier score 0.034, ephemeris recovery 100.0%.
  - Compiled two-column, 3-page scientific report (`docs/report/transiteye_report.pdf`).
  - Comprehensive documentation (`model-card.md`, `methodology.md`, `data-and-label-statement.md`, `result-schema.md`, `user-guide.md`).
  - 84 unit and integration tests passing headlessly with 100% clean-room verification.
