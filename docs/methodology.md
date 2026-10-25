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
