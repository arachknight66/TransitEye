# TransitEye Model Card: Calibrated Morphology Classifier

## Model Details
- **Architecture**: Ensembled `HistGradientBoostingClassifier` (selected via grouped cross-validation over Random Forest and Logistic Regression) calibrated using Platt scaling (`sigmoid`) on a dedicated held-out calibration partition.
- **Input Features**: 25 audited astrophysical and time-series morphology features across six families (Morphology, Consistency, Eclipse, Variability, Noise/Quality, Source Evidence).
- **Output Classes**: Four mutually exclusive morphological categories:
  - `transit_like`: Symmetrical, flat-bottomed or limb-darkened U-shaped dips.
  - `eclipse_like`: Eclipsing binary profiles, V-shaped dips, or primary/secondary depth alternations.
  - `stellar_variability`: Quasi-periodic sinusoidal, rotational spot modulation, or stellar pulsations.
  - `other_artifact`: Instrumental discontinuities, momentum dumps, thermal ramps, scattered light, or non-periodic noise artifacts.
- **Abstention Policy**: Rejects predictions when top calibrated confidence $\tau_{\text{conf}} < 0.40$ or margin $\tau_{\text{margin}} < 0.10$ (or out-of-distribution feature bounds), emitting `abstained = True` and `predicted_class = None`.
- **Training Pipeline**: Grouped Stratified 5-Fold cross-validation over unique astronomical source groups (`source_group_id`) across 72 training light-curve observations to ensure zero target-level data leakage.

## Quantitative Performance
Evaluated on the locked single-shot test partition ($N = 8$ distinct astrophysical sources) and sector-shift generalization cohort:
- **Test Accuracy**: 100.0% (95% CI: [100.0%, 100.0%])
- **Macro-F1**: 1.000 (95% CI: [0.750, 1.000])
- **Expected Calibration Error (ECE)**: 0.151
- **Multiclass Brier Score**: 0.034
- **Abstention Rate**: 0.0%
- **Ephemeris Period Recovery**: 100.0%
- **Empirical Uncertainty Coverage**: 100.0% (both 68% and 95% nominal intervals contain ground truth)
- **Synthetic Injection Recovery**: 37.5% at 200 ppm, 100.0% at $\ge 500$ ppm (500, 1000, 2500, 5000, 10000 ppm)
- **Sector-Shift Robustness**: 100.0% classification accuracy across all 8 multi-sector targets without retraining.
- **Feature Ablation Impact**: Removing stellar variability features drops Macro-F1 from 1.000 to 0.617 (37.5% degradation).

## Intended Use
- Automated triage, characterization, and preliminary screening of transiting exoplanet candidates in TESS 2-minute and 20-second cadence light curves.
- Scientific inspection via the headless CLI (`transiteye analyze-fits`) and interactive desktop interface (`transiteye-desktop`).

## Out-of-Scope & Misuse
- **Exoplanet Confirmation**: A `transit_like` classification **never confirms an exoplanet**. Astrophysical false positives (e.g. background eclipsing binaries, grazing binaries, hierarchical triples) can mimic planetary transit shapes. Independent radial-velocity or high-resolution imaging follow-up is strictly required for confirmation.
- **Blind Unsupervised Extrapolation**: Applying the model to non-TESS data (e.g., ground-based surveys or FFI custom apertures) without retraining.

## Scientific Limitations
- Highly eccentric systems or multi-planet transit timing variations (TTVs) may exhibit reduced BLS detection efficiency.
- Grazing eclipsing binaries may share degenerate U-to-V shape ratios with giant planets.
- Blend identification in crowded fields (`CROWDSAP < 0.85`) without high-resolution companion Target Pixel Files (TPFs) defaults to `UNRESOLVED` rather than definitive target attribution.
