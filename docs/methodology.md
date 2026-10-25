# TransitEye Scientific Methodology

## 1. Quality Filtering & Time-Series Preprocessing
TESS SPOC and TESS-SPOC light curves contain data quality anomalies caused by Earth/Moon scattered light, thermal transients following momentum dumps, and coarse-pointing events. TransitEye applies a bitmask filter over the FITS `QUALITY` column:
$$\mathcal{M}_{\text{quality}} = (\text{QUALITY} \ \& \ \text{DEFAULT\_BITMASK}) == 0$$
Cadences with non-finite flux or null timestamps are masked. Normalized relative flux is computed using median robust scaling:
$$f_{\text{norm}}(t) = \frac{f(t)}{\text{median}(f)}$$

## 2. Robust Duration-Aware Biweight Detrending
To disentangle stellar rotational variability and instrumental trends from planetary transits, TransitEye implements an iterative, duration-aware biweight filter:
- **Pass 1 (Stellar Trend Estimation)**: A broad Tukey biweight moving window ($W = 1.0\text{--}3.0\text{ days}$) estimates the continuum flux while down-weighting outliers:
  $$w(u) = (1 - u^2)^2 \quad \text{for } |u| \le 1, \quad u = \frac{t - t_0}{c \cdot \text{MAD}}$$
- **Pass 2 (In-Transit Protection)**: Once preliminary transit candidates $(P_i, T_{0,i}, D_i)$ are identified, in-transit windows $[T_{0,i} - D_i, T_{0,i} + D_i]$ are masked from the biweight kernel. This prevents transit signal attenuation and flux distortion.
- **Hardware Acceleration**: For large sector workloads, detrending is accelerated via a custom CUDA kernel executing in PyTorch/CuPy with automatic fallback to CPU `wotan`.

## 3. Period Determination: Box Least-Squares (BLS)
Transit periodicities are searched over a uniformly spaced frequency grid ($f = 1/P$) using Box Least-Squares (BLS). For each trial period $P$ and duration $q$, the BLS power spectrum evaluates:
$$\text{SNR}_{\text{BLS}}(P, q) = \frac{s(P, q)}{\sqrt{r(P, q)}}$$
Subharmonic ($P/2$) and harmonic ($2P$) aliases are explicitly identified and tracked. Concurrently, Lomb-Scargle periodograms quantify sinusoidal stellar rotational periods to detect stellar activity aliases.

## 4. Transit Profile Modeling & Parameter Uncertainty
- **Symmetric Trapezoid & Batman Limb Darkening**: Detected transit dips are characterized by nonlinear least-squares optimization over orbital parameters:
  $$\vec{\theta} = \{ P, T_0, (R_p / R_*)^2, a/R_*, i, u_1, u_2 \}$$
- **Pont Red-Noise Detection SNR**: White noise underestimates uncertainty in correlated time-series. TransitEye computes the Pont et al. (2006) pink/red noise SNR:
  $$\text{SNR}_{\text{red}} = \frac{\delta}{\sqrt{\frac{\sigma_w^2}{n_{\text{in}}} + \frac{\sigma_r^2}{N_{\text{tr}}}}}$$
- **Moving-Block Bootstrap**: 95% confidence intervals for depth, duration, and period are computed via circular moving-block bootstrap with block length $L = 2.0 \cdot D$, preserving correlated noise properties.

## 5. Independent Spatial Source Vetting
TransitEye decouples morphological shape classification from astrophysical source localization:
- **Tier 1 (Light Curve Centroids)**: Computes the astrometric shift significance between in-transit and out-of-transit flux centroids:
  $$\Delta c = \sqrt{(\bar{x}_{\text{in}} - \bar{x}_{\text{oot}})^2 + (\bar{y}_{\text{in}} - \bar{y}_{\text{oot}})^2}, \quad \sigma_{\Delta c} = \frac{\Delta c}{\text{SE}(\Delta c)}$$
- **Tier 2 (Companion Target Pixel Files)**: When `_tp.fits` pixel stamps are available, TransitEye performs in-minus-out difference imaging and concentric aperture depth growth tests. Crowded targets ($CROWDSAP < 0.85$) without pixel evidence are categorized as `UNRESOLVED` to avoid false target attribution.

## 6. Machine Learning Candidate Classification & Confidence Calibration

### 6.1 Audited Feature Vector Construction
For each candidate dip, TransitEye extracts an audited 25-dimensional feature vector $\vec{x} \in \mathbb{R}^{25}$ spanning six physical families:
$$\vec{x} = [\vec{x}_{\text{morph}}, \vec{x}_{\text{consist}}, \vec{x}_{\text{eclipse}}, \vec{x}_{\text{variability}}, \vec{x}_{\text{noise}}, \vec{x}_{\text{source}}]^T$$
- **Morphology ($\vec{x}_{\text{morph}} \in \mathbb{R}^5$)**: Ingress/egress ratio ($v_{\text{shape}}$), duty cycle ($T_{\text{dur}}/P$), flat-bottom boxiness ($1 - v_{\text{shape}}$), asymmetry score, and physical transit $\Delta\text{BIC} = \text{BIC}_{\text{trapezoid}} - \text{BIC}_{\text{batman}}$.
- **Consistency ($\vec{x}_{\text{consist}} \in \mathbb{R}^5$)**: Odd-even transit depth significance $\sigma_{\text{oe}} = |\delta_{\text{odd}} - \delta_{\text{even}}| / \sqrt{\sigma_{\text{odd}}^2 + \sigma_{\text{even}}^2}$, per-transit depth MAD ratio, orbital phase bin coverage $C_{\phi}$, in-transit duty cycle, and observed transit event count $N_{\text{tr}}$.
- **Secondary Eclipse ($\vec{x}_{\text{eclipse}} \in \mathbb{R}^4$)**: Maximum secondary dip SNR, secondary-to-primary depth ratio $\delta_{\text{sec}} / \delta_{\text{pri}}$, secondary phase offset $|\phi_{\text{sec}} - 0.5|$, and tidal ellipsoidal amplitude ratio $A_{\text{ellip}} / \delta$.
- **Variability ($\vec{x}_{\text{variability}} \in \mathbb{R}^4$)**: Peak Lomb-Scargle power $P_{\text{LS}}$, harmonic distance to BLS period $\min_k |P_{\text{LS}} - k P| / (k P)$, out-of-transit RMS to depth ratio $\sigma_{\text{oot}} / \delta$, and lag-autocorrelation $R(\tau = P)$.
- **Noise & Quality ($\vec{x}_{\text{noise}} \in \mathbb{R}^4$)**: Pont red-noise factor $\beta$, detection $\text{SNR}_{\text{red}}$, in-transit quality-flagged fraction, and cadence gap proximity fraction.
- **Source Vetting ($\vec{x}_{\text{source}} \in \mathbb{R}^3$)**: Centroid shift $\sigma_{\Delta c}$, uncorrected-to-corrected depth ratio $\delta_{\text{SAP}} / \delta_{\text{PDCSAP}}$, and aperture crowding metric $\text{CROWDSAP}$.

To guarantee scientific validity and zero leakage, all metadata (TIC IDs, coordinates, sectors, magnitudes, ephemerides, split tags) are strictly blocked from feature construction.

### 6.2 Model Architecture & Grouped Optimization
TransitEye employs a Histogram-based Gradient Boosted Decision Tree (`HistGradientBoostingClassifier`) optimizing the multi-class multinomial deviance (cross-entropy loss):
$$\mathcal{L}(\Theta) = - \sum_{i=1}^N \sum_{c=1}^K y_{i,c} \log p_c(\vec{x}_i; \Theta)$$
where $K = 4$ target classes:
$$\mathcal{C} = \{\text{transit\_like}, \text{eclipse\_like}, \text{stellar\_variability}, \text{other\_artifact}\}$$

To prevent data leakage across observation sectors and synthetic augmentations, model training enforces **Grouped Stratified K-Fold** partitioning on astronomical source identifier (`source_group_id`):
$$\mathcal{S}_{\text{train}} \cap \mathcal{S}_{\text{val}} = \emptyset, \quad \mathcal{S}_{\text{train}} \cap \mathcal{S}_{\text{test}} = \emptyset$$

### 6.3 Post-Hoc Probability Calibration (Platt Scaling)
Raw decision tree margins are calibrated into true posterior probabilities using regularized Platt scaling fitted on a dedicated held-out calibration split:
$$P(y = c \mid \vec{x}) = \frac{\exp(A_c f_c(\vec{x}) + B_c)}{\sum_{j=1}^K \exp(A_j f_j(\vec{x}) + B_j)}$$
Calibration fidelity is audited via Expected Calibration Error (ECE) and multiclass Brier score:
$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|, \quad \text{BS} = \frac{1}{N} \sum_{i=1}^N \sum_{c=1}^K (p_{i,c} - y_{i,c})^2$$

### 6.4 Autonomous Scientific Abstention Policy
To guard against erroneous automated classifications on ambiguous or out-of-distribution signals, the model evaluates three formal abstention gates:
1. **Low Confidence**: $\max_{c \in \mathcal{C}} P(y = c \mid \vec{x}) < \tau_{\text{conf}}$ (default $\tau_{\text{conf}} = 0.40$).
2. **Narrow Classification Margin**: $P_{(1)} - P_{(2)} < \tau_{\text{margin}}$ (default $\tau_{\text{margin}} = 0.10$).
3. **Out-of-Distribution (OOD) Features**: $\exists j \in [1, 25]: x_j < \min_j - 5\Delta_j \lor x_j > \max_j + 5\Delta_j$.

When any gate triggers, TransitEye abstains from class assignment:
$$\text{abstained} = \text{True}, \quad \text{predicted\_class} = \text{None}$$
This preserves high downstream purity by routing ambiguous signals directly to human expert review.
