# TransitEye Model Card: Calibrated Morphology Classifier

Following the framework of Mitchell et al. (2019) (*Model Cards for Model Reporting*), this document details the design, feature architecture, training protocol, calibration, locked evaluation, and scientific limitations of the TransitEye machine learning candidate classifier.

> **Model status (updated 2026-11-01):** `morphology_real_catalog_v1` was historically evaluated
> against catalog-derived proxies (Section 11), but that result does not validate the current
> pipeline or establish independently adjudicated morphology. The newer candidate
> `morphology_real_catalog_v3` has no independent locked evaluation. Sections 1–10 describe an
> earlier synthetic experiment with invalid generalization scores.

> **Execution update (2026-11-01):** The legacy evaluation entry point is disabled. Analysis no
> longer selects or trains a default classifier; the historical real-catalog v1 bundle can be
> explicitly selected as described in the README. Its performance describes the historical fixed
> evaluation path against catalog-derived proxies and does not validate the complete desktop or
> batch workflow. The newer model's holdout remains sealed.

Current training and inference use FITS effective temperature, when present, to select approximate
limb-darkening coefficients for the physical fit. This can indirectly affect a fit-comparison feature;
temperature is therefore an input to that calculation even though it is not a classifier column.
The historical v1/v3 evaluations did not independently validate this updated feature path.

---

## 1. Model Overview

- **Model Identifier**: `transiteye_morphology_v3` (champion) / `transiteye_morphology_v2` (1,000 targets) / `transiteye_morphology_v1` (pilot legacy)
- **Model Type**: Random Forest Classifier (`RandomForestClassifier`, scikit-learn) with post-hoc sigmoid probability calibration (Platt scaling via `CalibratedClassifierCV`).
- **Training Scale**: 5,000 distinct astrophysical targets (10,500 training samples with physical prior augmentations across 3,500 train, 500 calibration, 500 selection, and 500 locked test targets).
- **Primary Task**: Four-class morphological categorization of periodic time-series dip candidates detected in TESS (Transiting Exoplanet Survey Satellite) 2-minute and 20-second cadence light curves.
- **Input Dimensionality**: 25 audited, leak-free continuous numerical features spanning six astrophysical and statistical families (22 morphology features used by the core classifier, 3 spatial evidence features available for source vetting and ablations).
- **Output Space**: Calibrated posterior probability vector $P(y \in \mathcal{C} \mid \vec{x})$ across four classes, with an autonomous abstention mechanism emitting `abstained = True` and `predicted_class = None` when confidence or classification margin falls below tuned scientific thresholds.
- **Code Repository**: `src/transiteye/classification/`

---

## 2. Target Taxonomy & Output Classes

TransitEye strictly separates signal morphology from astrophysical source localization:

| Class Identifier | Scientific Definition | Archetypal Physical Sources |
|---|---|---|
| `transit_like` | Symmetrical, U-shaped, flat-bottomed or limb-darkened dips consistent with planetary transit or low-mass occultor geometry. | Transiting exoplanets, low-mass brown dwarf transits. |
| `eclipse_like` | V-shaped grazing profiles, alternating primary/secondary depths, ellipsoidal out-of-transit variations, or deep mutual eclipses. | Detached, semi-detached, and contact eclipsing binaries (EBs, W UMa). |
| `stellar_variability` | Smooth, quasi-periodic sinusoidal, multi-periodic, or rotational spot modulation without isolated transit flat baselines. | Rotational modulation (spots), $\delta$ Scuti, $\gamma$ Doradus, Cepheid pulsations, flare remnants. |
| `other_artifact` | Non-astrophysical discontinuities, single-event thermal settling, momentum dumps, scattered light ramps, or high red-noise bursts. | Spacecraft momentum dumps, Earth/Moon glint, pointing jitter, detector anomalies. |

> [!IMPORTANT]
> **Scientific Non-Negotiable**: A `transit_like` classification indicates solely that the time-series morphology matches planetary transit geometry. It **never confirms an exoplanet**. Astrophysical false positives (e.g., blended background eclipsing binaries, hierarchical triples) frequently produce `transit_like` shapes. High-resolution speckle imaging and radial-velocity spectrographic follow-up are strictly mandatory for planet confirmation.

---

## 3. Audited Feature Registry (25 Features across 6 Families)

TransitEye uses an audited feature registry ([`src/transiteye/features/registry.py`](file:///home/arachknight/Documents/Projects/USICT/TransitEye/src/transiteye/features/registry.py)). Features are extracted purely from the prepared/detrended light curve and candidate properties. Target identifiers, coordinates ($\alpha, \delta$), TESS magnitudes, sector numbers, ephemeris priors, and split tags are strictly prohibited to prevent data leakage.

### Family 1: Morphology (5 Features)
- `v_shape_ratio`: Fraction of transit half-duration spent in ingress/egress ramps ($0.0 \le \tau_{\text{ing}} \le 1.0$). Distinguishes flat-bottom transits ($\sim 0.1\text{--}0.3$) from grazing V-shaped binary eclipses ($\to 1.0$).
- `duration_to_period_ratio`: Duty cycle $T_{\text{dur}} / P$. Planetary orbits typically exhibit $T_{\text{dur}}/P \sim 0.005\text{--}0.05$; long-duration signals at short periods indicate unphysical transit models or contact binaries.
- `boxiness_index`: Complement of ingress fraction ($1 - \text{ingress\_fraction}$). Quantifies flat bottom prominence.
- `asymmetry_score`: Normalized flux difference between ingress and egress phases ($|\bar{f}_{\text{ing}} - \bar{f}_{\text{eg}}| / \delta$). Detects asymmetric instrumental trends, stellar flares, or cometary tails.
- `batman_vs_trapezoid_delta_bic`: Bayesian Information Criterion difference ($\text{BIC}_{\text{trapezoid}} - \text{BIC}_{\text{batman}}$). Positive values indicate that a physical quadratic limb-darkened Mandel & Agol transit model fits significantly better than a crude geometric trapezoid.

### Family 2: Consistency Across Cycles (5 Features)
- `odd_even_depth_sigma`: Statistical significance of transit depth differences between odd and even cycles:
  $$\sigma_{\text{oe}} = \frac{|\delta_{\text{odd}} - \delta_{\text{even}}|}{\sqrt{\sigma_{\text{odd}}^2 + \sigma_{\text{even}}^2}}$$
  Signals with $\sigma_{\text{oe}} \ge 3.0$ strongly indicate an eclipsing binary observed at half its true orbital period ($P_{\text{true}} = 2 P_{\text{BLS}}$).
- `per_transit_depth_mad_ratio`: Median Absolute Deviation of individual transit depths divided by median depth. High dispersion indicates instrumental glitches or scattered light rather than genuine periodic dips.
- `phase_coverage_fraction`: Fraction of 20 in-transit orbital phase bins containing at least one valid photometric cadence ($0.0 \le C_{\phi} \le 1.0$).
- `in_transit_duty_cycle`: Fraction of expected transit epochs that fall within observed data segments and contain valid observations.
- `observed_event_count`: Integer count of observed transit events with valid data ($N_{\text{tr}} \ge 1$).

### Family 3: Secondary Eclipse & Orbital Physics (4 Features)
- `max_secondary_snr`: Peak signal-to-noise ratio of any secondary dip detected across the out-of-transit phase range ($\phi \in [0.1, 0.9]$).
- `secondary_to_primary_depth_ratio`: Ratio of secondary eclipse depth to primary transit depth ($\delta_{\text{sec}} / \delta_{\text{pri}}$). Ratios $> 0.05$ indicate stellar occultations (eclipsing binaries) rather than planetary occultations.
- `secondary_phase_offset_from_half`: Absolute displacement of candidate secondary eclipse from circular orbit phase 0.5 ($|\phi_{\text{sec}} - 0.5|$). Indicates orbital eccentricity ($e \cos \omega$).
- `ellipsoidal_amplitude_ratio`: Amplitude of the $\cos(4\pi \phi)$ tidal ellipsoidal distortion component relative to transit depth. Massive close-in stellar companions induce strong tidal variations.

### Family 4: Stellar & Residual Variability (4 Features)
- `ls_max_power`: Peak normalized Lomb-Scargle power across frequencies $f \in [0.01, 12.0]\text{ day}^{-1}$. Quantifies dominant periodic stellar rotational modulation or pulsations.
- `ls_to_bls_period_harmonic_distance`: Minimum fractional distance between the Lomb-Scargle peak period and the BLS period or its harmonics ($P, P/2, 2P$):
  $$\Delta_{\text{harm}} = \min_{k \in \{0.5, 1.0, 2.0\}} \frac{|P_{\text{LS}} - k \cdot P_{\text{BLS}}|}{k \cdot P_{\text{BLS}}}$$
  Low distance indicates the BLS algorithm locked onto a stellar rotational harmonic rather than an independent transit.
- `out_of_transit_rms_to_depth_ratio`: Ratio of out-of-transit detrended RMS scatter to candidate transit depth ($\sigma_{\text{oot}} / \delta$). High values indicate marginal or noise-dominated candidates.
- `autocorrelation_at_period`: Normalized light curve autocorrelation at time lag equal to candidate orbital period $R(\tau = P)$.

### Family 5: Correlated Noise & Data Quality (4 Features)
- `red_noise_beta_factor`: Pont et al. (2006) $\beta$ factor measuring correlated red noise over transit duration timescales relative to the theoretical Gaussian white noise expectation ($\beta = \sigma_N / (\sigma_1 / \sqrt{N})$).
- `detection_snr`: Pink/red-noise corrected transit detection signal-to-noise ratio:
  $$\text{SNR}_{\text{red}} = \frac{\delta}{\sqrt{\frac{\sigma_w^2}{n_{\text{in}}} + \frac{\sigma_r^2}{N_{\text{tr}}}}}$$
- `in_transit_flagged_fraction`: Fraction of raw cadences in transit having non-zero SPOC quality flags (e.g., Earth/Moon glint, desat events, cosmic ray hits).
- `gap_proximity_fraction`: Fraction of in-transit cadences located within 1 hour of a data gap (orbit downlink or momentum dump).

### Family 6: Spatial & Source Evidence (3 Features)
*(Extracted independently for vetting, available to the model or isolated in ablations)*
- `centroid_shift_sigma`: Statistical significance of in-transit vs out-of-transit photometric centroid shift ($\sigma_{\Delta c}$). Shifts $\ge 3.5\sigma$ indicate blended background eclipsing binaries (BGEBs).
- `sap_to_pdcsap_depth_ratio`: Ratio of transit depth measured on uncorrected SAP flux to detrended PDCSAP flux. Divergence indicates PDC crowding or trend correction artifacts.
- `crowdsap_value`: Ratio of target flux to total aperture flux ($0.0 \le \text{CROWDSAP} \le 1.0$). Low values indicate high background blending.

---

## 4. Training, Validation & Model Selection

### 4.1 Dataset Partitioning & Leakage Prevention (Cohort v3)
To provide true sector-scale, statistically powerful coverage, the corpus was scaled to a balanced 5,000-star cohort (`manifests/cohort_v3_labels.json`, `manifests/cohort_v3_splits.json`, manifest ID `cohort_v3`) curated from authentic archival catalogs:
- **`transit_like` (1,250 targets)**: NASA Exoplanet Archive TOI table (`tfopwg_disp` in `'CP'`, `'KP'`).
- **`eclipse_like` (1,250 targets)**: NASA Exoplanet Archive TOI false-positive eclipsing binaries (`tfopwg_disp == 'FP'` with transit depth $\delta > 5,000\text{ ppm}$).
- **`stellar_variability` (1,250 targets)**: VizieR Asteroseismic Target List (`J/ApJS/241/12`) and rotational modulation variables (`J/ApJS/250/20`).
- **`other_artifact` (1,250 targets)**: NASA Exoplanet Archive False Alarms (`tfopwg_disp == 'FA'`), shallow noise excursions, and TESS SPOC Data Release Notes instrumental anomalies.

Dataset partitioning strictly isolates astronomical sources (`source_group_id = f"tic-{target_id}"`). All sectors, candidates, augmentations, and noise derivatives from a single star reside exclusively within one partition:
- **`train`**: 3,500 unique source groups (875 per class; 10,500 training samples including deterministic physical prior synthetic variants).
- **`calibration`**: 500 unique source groups (125 per class, held-out for Platt probability calibration).
- **`selection`**: 500 unique source groups (125 per class, held-out for abstention threshold tuning).
- **`test`**: 500 unique source groups (125 per class, sealed locked test set opened strictly during single-shot evaluation).

### 4.2 Candidate Model Comparison (Grouped 5-Fold Cross-Validation)
Three model families were trained and evaluated using `StratifiedGroupKFold(n_splits=5)` on the 10,500 samples of the `train` partition:

| Model Architecture | Hyperparameters / Configuration | Grouped CV Macro-$F_1$ | Transit Recall | Selection Status |
|---|---|---|---|---|
| **RandomForestClassifier** | `n_estimators=200`, `max_depth=14`, `min_samples_split=3`, `random_state=42` | **1.000** | **1.000** | **Selected (Champion)** |
| HistGradientBoostingClassifier | `learning_rate=0.08`, `max_iter=150`, `max_leaf_nodes=31`, `min_samples_leaf=5`, `random_state=42` | **1.000** | **1.000** | Contender |
| LogisticRegression | `C=1.0`, `solver='lbfgs'`, `StandardScaler()`, `max_iter=1000` | 1.000 | 1.000 | Baseline |

`RandomForestClassifier` was selected as the operational champion for `transiteye_morphology_v3` due to its superior variance reduction across diverse noise distributions, out-of-bag ensemble stability, and resistance to overfitting on high-dimensional light curve features.

### 4.3 Calibration Metrics
- **Post-Hoc Calibrator**: Platt sigmoid scaling (`CalibratedClassifierCV(method='sigmoid')`) fitted on the 500-target held-out `calibration` partition.
- **Multiclass Brier Score**: $0.000086$
- **Expected Calibration Error (ECE)**: $0.0080$
- **Abstention Thresholds**: $\tau_{\text{conf}} = 0.40$, $\tau_{\text{margin}} = 0.10$ tuned on the 500-target `selection` partition.

---

## 5. Probability Calibration & Abstention Policy

### 5.1 Platt Sigmoid Calibration
Raw ensemble trees often output uncalibrated, overconfident margins near 0 and 1. TransitEye fits a post-hoc sigmoid (Platt) regularizer using `CalibratedClassifierCV(method='sigmoid')` fitted on the dedicated `calibration` partition:
$$P(y = c \mid \vec{x}) = \frac{1}{1 + \exp(A_c \cdot f_c(\vec{x}) + B_c)}$$
Normalized so that $\sum_{c \in \mathcal{C}} P(y = c \mid \vec{x}) = 1.0$.

### 5.2 Scientific Abstention Policy
TransitEye enforces explicit abstention rather than forcing speculative guesses on ambiguous or pathological signals:
1. **Confidence Threshold**: $\max_c P(y=c \mid \vec{x}) < \tau_{\text{conf}}$ (default $\tau_{\text{conf}} = 0.40$).
2. **Margin Threshold**: $P_{(1)} - P_{(2)} < \tau_{\text{margin}}$ (default $\tau_{\text{margin}} = 0.10$).
3. **Out-of-Distribution (OOD) Guardrail**: If any feature exceeds the training bounding box by more than $5.0 \times \text{span}$ ($x_j < \min_j - 5\Delta_j$ or $x_j > \max_j + 5\Delta_j$), the model automatically abstains.

When any condition triggers:
$$\text{abstained} = \text{True}, \quad \text{predicted\_class} = \text{None}$$

---

## 6. Quantitative Performance on Locked Evaluation

The model was frozen and evaluated single-shot on the locked `test` split ($N = 500$ distinct astrophysical sources, 125 per class) using the frozen evaluation protocol (`configs/evaluation_locked.v3.toml`).

### 6.1 Primary Performance Metrics

| Evaluation Metric | Measured Value | 95% Bootstrap Confidence Interval |
|---|---|---|
| **Effective Coverage** | **71.6%** | [69.2%, 74.0%] |
| **Abstention Rate** | **28.4%** | [26.0%, 30.8%] |
| **Full System Accuracy (penalized for abstentions)** | **59.4%** | [56.0%, 62.8%] |
| **Macro-$F_1$ Score** | **0.638** | [0.607, 0.669] |
| **Expected Calibration Error (ECE)** | **0.091** | [0.075, 0.109] |
| **Multiclass Brier Score** | **0.167** | — |
| **Ephemeris Recovery Rate** | **100.0%** (250/250 periodic candidates) | [100.0%, 100.0%] |
| **Ephemeris Primary Matched Fraction** | **100.0%** | [100.0%, 100.0%] |
| **Ephemeris Harmonic Fraction** | **0.0%** (zero period aliasing) | — |
| **Median Relative Depth Error** | **4.7%** | — |
| **Parameter 68% Interval Coverage** | **100.0%** | — |
| **Parameter 95% Interval Coverage** | **100.0%** | — |

### 6.2 Per-Class Breakdown (Held-Out Test Set, $N = 500$)

| Class | Precision | Recall | $F_1$ Score | Support |
|---|---|---|---|---|
| `transit_like` | 0.645 | 0.856 | 0.735 | 125 |
| `eclipse_like` | 0.969 | 0.248 | 0.395 | 125 |
| `stellar_variability` | 1.000 | 0.272 | 0.428 | 125 |
| `other_artifact` | 0.992 | 1.000 | 0.996 | 125 |

### 6.3 Robustness Strata Breakdown
- **High-SNR Signals ($\text{SNR} \ge 10$)**: Accuracy = **94.7%** ($N = 300$)
- **Moderate-SNR Signals ($\text{SNR} < 10$)**: Accuracy = **70.0%** ($N = 200$)

---

## 7. Synthetic Injection & Recovery Analysis

TransitEye injected synthetic Mandel & Agol transit models into real, quiescent TESS light curves across transit depths from 200 ppm to 10,000 ppm ($\text{SNR} \ge 5.5$):

| Injected Depth (ppm) | Recovery Fraction | Median Relative Depth Error | Classification as `transit_like` |
|---|---|---|---|
| **200 ppm** (Earth/sub-Neptune) | 0.0% (sub-threshold) | — | — |
| **500 ppm** (Super-Earth / Neptune) | 37.5% | 11.4% | 100% (of recovered) |
| **1,000 ppm** (Sub-Saturn) | 62.5% | 10.8% | 100% |
| **2,500 ppm** (Saturn/Jupiter) | 75.0% | 10.4% | 100% |
| **5,000 ppm** (Jupiter) | 87.5% | 10.2% | 100% |
| **10,000 ppm** (Deep transit) | 100.0% | 10.5% | 100% |

The empirical 50% recovery threshold on quiescent TESS SPOC substrates occurs at $\sim 780\text{ ppm}$ in single-sector 2-minute cadence data.

---

## 8. Feature Ablation Studies

To quantify feature family and algorithmic contributions, systematic ablations were evaluated on the held-out test set ($N=500$):

| Model Configuration | Accuracy | Macro-$F_1$ | Relative Degradation | Key Finding |
|---|---|---|---|---|
| **Drop Source Features** | **1.000** | **1.000** | **Baseline** | Light curve morphology alone sustains 100% macro-$F_1$ without spatial centroid features. |
| **Real Data Only (No Injections)** | **1.000** | **1.000** | **0.0%** | Scale of 5,000 real archival targets provides extensive empirical coverage across morphological phase space. |
| **Drop Variability Features** | 0.994 | 0.996 | -0.4% | Removing Lomb-Scargle power introduces minor ambiguity between low-amplitude stellar pulsations and transits. |
| **Pass 1 Detrending Only** | 0.240 | 0.097 | **-90.3%** | **Critical**: Disabling Pass-2 detrending severely distorts transit profiles, resulting in catastrophic failure of the detection and classification pipeline. |

---

## 9. Model Artifacts & Serving Architecture

The model bundle is serialized as a versioned, self-contained directory:
```
.transiteye-workspace/models/morphology_v3/
├── model_metadata.json    # JSON metadata: feature names, bounds, thresholds, checksums
├── estimator.joblib       # Serialized RandomForestClassifier
└── calibrator.joblib      # Serialized CalibratedClassifierCV
```

- **Inference Runtime**: $< 2.0\text{ ms}$ per candidate feature vector on CPU.
- **AST Architecture Guarantee**: Zero GUI dependencies; imports are strictly isolated to `numpy`, `scipy`, and `scikit-learn` in `src/transiteye/`.
- **Integrity Verification**: `estimator_sha256` in `model_metadata.json` ensures tampering detection upon loading.

---

## 10. Caveats, Failure Modes & Ethical Guidelines

1. **Astrophysical Blend Confusion**: Background Eclipsing Binaries (BEBs) blended inside the large 21-arcsecond TESS pixels can produce morphologically flawless U-shaped dips. Users must inspect Tier 2 TPF difference images and `centroid_shift_arcsec` before treating any candidate as plausible.
2. **Grazing Binary Degeneracy**: Grazing eclipsing binaries with small impact parameters can mimic short-duration shallow transits.
3. **Extreme Low SNR ($< 5.5\sigma$)**: Signals near the photometric noise floor exhibit higher parameter scatter and are subject to abstention.
4. **Scientific Responsibility**: Never report candidate discoveries as "confirmed exoplanets" based solely on TransitEye classifications. Always cross-reference ExoFOP-TESS, Gaia DR3 astrometry, and obtain spectroscopic radial velocity confirmation.

---

## 11. Real Catalog Cohort Model and Locked Test Results

The independently evaluated artifact `morphology_real_catalog_v1` is a `HistGradientBoostingClassifier` with the saved calibration and abstention policy. It consumes 22 morphology features and was trained with no synthetic augmentation or label-conditioned feature transforms. Its source-grouped partitions contain 597 training sources, 84 calibration sources, and 86 selection sources. Five-fold grouped cross-validation within the training partition selected the model with macro-F1 **0.630** and transit-like recall **0.577**; this CV estimate is not the locked-test result.

The single-shot test used the frozen protocol [`evaluation_real_catalog.v1.toml`](../configs/evaluation_real_catalog.v1.toml) and the real-cohort split, label, and product manifests. All 96 test sources were accounted for. Candidate features were extracted for 95; one trapezoid fit failed and was retained as an abstention in end-to-end metrics. No source lacked a candidate at the fixed search thresholds.

| Metric | Candidate-conditional classifier (N=95) | End-to-end with failed fit abstaining (N=96) |
|---|---:|---:|
| Accuracy | 0.642 | 0.635 |
| Macro-F1 | 0.682 | 0.678 |
| Transit-like precision | 0.652 | 0.652 |
| Transit-like recall | 0.600 | 0.600 |
| Coverage | 0.853 | 0.844 |
| Abstention rate | 0.147 | 0.156 |
| ECE | 0.176 | Not reported |
| Multiclass Brier score | 0.474 | Not reported |

For the candidate-conditional metrics, source-grouped 95% bootstrap percentile intervals were accuracy **[0.547, 0.727]**, macro-F1 **[0.584, 0.764]**, and transit-like recall **[0.389, 0.786]**. End-to-end intervals were **[0.542, 0.729]**, **[0.578, 0.765]**, and **[0.394, 0.800]**, respectively. Bootstrap samples resampled source groups with replacement.

Per-class candidate-conditional precision / recall / F1 were: transit-like **0.652 / 0.600 / 0.625**; eclipse-like **0.850 / 0.708 / 0.773**; stellar variability **0.800 / 0.833 / 0.816**; and other artifact **0.692 / 0.409 / 0.514**. The weaker other-artifact recall indicates substantial confusion remains in that class.

These are scores against catalog-derived morphology proxies, not independent visual adjudications. They do not establish exoplanet confirmation or generalize to arbitrary TESS targets. The full checksummed per-source output is stored outside Git at `.transiteye-workspace/evaluations/real_catalog_cohort_v1/evaluation_report.json`. The locked test split must not be used to tune this model or its thresholds.

## 12. Real-catalog retraining candidate (2026-10-31; no locked evaluation)

`morphology_real_catalog_v3` is a retraining candidate, not an evaluated replacement for v1. It uses
the same 22 morphology features and four class definitions. It combines the development partitions
of the prior real-catalog cohort with 246 new EB catalog sources from the MAST TESS-EBs v1.0 HLSP.
Catalog TICs overlapping the existing real cohorts were excluded. TESS-EBs provides curated EB
identities, periods, and light-curve morphology; the assigned `eclipse_like` label is still a
system-level proxy and does not establish that the strongest detected event in an individual SPOC
light curve is on-target or matches the catalog ephemeris.

| Partition | Usable sources | `transit_like` | `eclipse_like` | `stellar_variability` | `other_artifact` |
|---|---:|---:|---:|---:|---:|
| Train | 772 | 175 | 325 | 142 | 130 |
| Calibration | 104 | 25 | 44 | 21 | 14 |
| Selection | 108 | 25 | 44 | 21 | 18 |

The saved random forest was selected by five-fold stratified group CV on train sources: macro-F1
**0.692**, transit-like recall **0.617**. Per-class CV F1 was 0.603 transit-like, 0.807 eclipse-like,
0.824 stellar variability, and 0.532 other artifact. These CV metrics guide model selection and are
not independent performance estimates. Calibration-partition diagnostics (multiclass Brier **0.435**,
ECE **0.077**) are fit-data diagnostics and do not establish calibrated generalization. Abstention
thresholds are confidence **0.40** and margin **0.15**.

No test partition was opened. The combined split manifest retains 123 test groups, but the prior
cohort's test was already used for v1 evaluation and must not be reused for this candidate. A new
independent, source-local evaluation protocol is required. The 27 new EB holdout groups are also
closed. Training used no synthetic augmentation or label-conditioned transformations; two new EB
train targets failed fit initialization and are recorded in
`.transiteye-workspace/models/morphology_real_catalog_v3/training_report.json`.
