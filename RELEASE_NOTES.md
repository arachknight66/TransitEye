# TransitEye v0.1.0 Release Notes

TransitEye v0.1.0 is a complete, scientifically rigorous system for detecting, characterizing, classifying, and vetting signals in noisy TESS light curves.

## Key Features & Highlights
- **Decoupled Architecture**: 100% headless scientific core under `src/transiteye/` (verified via AST analysis to contain zero GUI dependencies) paired with a responsive PyQt6 / PyQtGraph desktop application under `desktop/transiteye_desktop/`.
- **Quality-Aware Two-Tier Detrending**: Iterative duration-aware Tukey biweight filtering with transit masking (Pass 2) to prevent transit distortion. Features hardware-accelerated CUDA execution with automatic CPU fallback.
- **Robust Period & Transit Modeling**: Box Least-Squares (BLS) frequency grid with harmonic alias tracking ($P/2, 2P$), Lomb-Scargle rotational variability analysis, Mandel \& Agol limb-darkened transit models via `batman`, Pont red-noise SNR estimation, and moving-block bootstrap parameter confidence intervals.
- **Audited 25-Feature Matrix**: Extracted across six families (Morphology, Consistency, Eclipse Physics, Variability, Noise/Quality, Spatial Evidence). Audited against metadata leakage.
- **Calibrated Gradient-Boosted Classifier with Abstention**: Trained under Grouped Stratified 5-Fold CV on 72 training observations with candidate model selection (`HistGradientBoosting` selected over Random Forest and Logistic Regression); Platt-calibrated on a held-out calibration partition. Automatically abstains ($\tau_{\text{conf}} < 0.40$, $\tau_{\text{margin}} < 0.10$) on ambiguous signals.
- **Independent Two-Tier Blend Vetting**: Tier 1 astrometric centroid shifts ($\sigma_{\Delta c}$) from light curves; Tier 2 difference imaging and concentric aperture growth from companion Target Pixel Files (`_tp.fits`). Crowded fields ($CROWDSAP < 0.85$) default to `UNRESOLVED` rather than false blend guesses.
- **Persistent SQLite WAL Batch Runner**: Bounded `ProcessPoolExecutor` with CUDA stream scheduling, cache hit indexing, resume/retry, and throughput metrics.
- **Scientific Deliverables**:
  - Full locked evaluation protocol executed on the sealed `test` split: accuracy 100.0%, macro-$F_1$ 1.000, ECE 0.151, Brier score 0.034, ephemeris recovery 100.0%.
  - Compiled two-column, 3-page scientific report (`docs/report/transiteye_report.pdf`).
  - Comprehensive documentation (`model-card.md`, `methodology.md`, `data-and-label-statement.md`, `result-schema.md`, `user-guide.md`).
