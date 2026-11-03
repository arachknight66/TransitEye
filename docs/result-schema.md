# TransitEye Result Bundle Schema Specification

Every target analysis produces a self-contained, versioned directory bundle (`.transiteye-workspace/results/<analysis_id>/`).

**Remediation note:** Current schema-v2 output uses `null` for uncomputed search significance and
for unavailable classifier probabilities/confidence in Parquet. The corresponding JSON classification
has an empty probability map, a null prediction and model version, and `abstained: true`, while
retaining independent source assessment. Missing-reason entries distinguish these states. Older v2
bundles remain readable, but their fixed-phase `search_significance` values are not validated as
search-wide significance. Interval coverage also remains under investigation.

New schema-v3 bundles add `analysis_settings`, `code_version`, `code_sha256`, `model_id`, and
`model_sha256` to `result.json`. Model checksums are absent for in-memory models without a persisted
artifact; no-classifier runs record a null model. Existing v2 bundles remain readable.
`analysis_settings.device` records the requested device and `resolved_device` records the CPU or GPU
used for the run, including batch CPU spillover.
`candidate_seeds` maps successfully processed candidates to a deterministic seed; the seed is used
for bootstrap refits when that stage runs, and recorded even when fast screening skips it.
Schema-v3 bundles also contain `integrity.json`, recording SHA-256 checksums of the three data files.
This detects accidental corruption during cache replay; it does not authenticate untrusted bundles.
The provenance metadata records the actual selected FITS flux column as `selected_flux`.
`trapezoid_exposure_days` is used by both fitting models and bootstrap refits. Its `exposure_status`
is `user_supplied` when passed explicitly and `fixed_assumption_unverified` for the historic two-minute
fallback. Raw frame keywords remain in the input metadata; sampling `TIMEDEL` and total `EXPOSURE`
are not automatically interpreted as exposure per cadence.
`candidate_errors` records fitting failures by candidate ID. Candidates with a `fit_failed` flag
retain preliminary detector estimates, have no fitted model, and lack physical classification.
The `physical_quadrature_unvalidated` candidate flag indicates that an optional Batman curve was
computed using the five-sample exposure approximation, whose accuracy is not established. The flag
appears in both JSON and Parquet. It does not validate the separately fitted trapezoid parameters.
The `physical_fit_failed` flag and `candidate_id.physical_fit` missing reason identify an optional
physical fit that was attempted but did not produce a model. A skipped physical fit is not a failure;
the generic flag does not identify the underlying numerical exception.
In new results, unavailable source-vetting measurements are non-finite in the corresponding Parquet
columns and are recorded under `missing_reasons` in JSON. Missing centroid/crowding data and a
zero-signal difference image yield an unresolved assessment. The SAP/PDCSAP depth ratio is not
computed from a single flux product. Historical bundles retain their original source assessments.

## Bundle Directory Layout
```text
bundle/
├── result.json          # Target-level metadata, candidates summary, classifications, and 68%/95% intervals
├── arrays.npz           # Compressed numerical time-series, periodograms, fitted models, and TPF difference stamps
└── candidates.parquet   # Tabular dataframe of all detected transit candidates, probabilities, and vetting metrics
```

### 1. `result.json`
```json
{
  "schema_version": "2",
  "analysis_id": "analysis_261136679_1729753821",
  "target_id": "261136679",
  "status": "complete",
  "configuration_version": "0.1.0",
  "provenance": {
    "target_id": "261136679",
    "sector": 1,
    "product_type": "tess_light_curve",
    "time_system": "TDB"
  },
  "candidates": [
    {
      "candidate_id": "cand_01",
      "period_days": 6.2679,
      "epoch_days": 1325.503,
      "duration_hours": 2.92,
      "depth_ppm": 298.5,
      "observed_event_count": 4,
      "detection_snr": 24.1,
      "search_significance": 0.995,
      "aliases_days": [3.13395, 12.5358],
      "flags": []
    }
  ],
  "classifications": {
    "cand_01": {
      "predicted_class": "transit_like",
      "abstained": false,
      "source_assessment": "target_consistent",
      "model_version": "morphology_random_forest_v1",
      "probabilities": {
        "transit_like": 0.942,
        "eclipse_like": 0.038,
        "stellar_variability": 0.012,
        "other_artifact": 0.008
      }
    }
  },
  "parameter_intervals": {
    "cand_01": {
      "period_days": {
        "68": {"lower": 6.2671, "upper": 6.2687, "coverage": 0.68, "missing_reason": null},
        "95": {"lower": 6.2662, "upper": 6.2696, "coverage": 0.95, "missing_reason": null}
      }
    }
  }
}
```

### 2. `arrays.npz`
Contains compressed NumPy arrays using canonical backend keys:
- `time_days`: float64 1D array of barycentric Julian dates (BJD - 2457000).
- `flux_raw`: float64 1D array of normalized delivered PDCSAP/SAP flux.
- `flux_detrended`: float64 1D array of detrended continuum-normalized flux.
- `trend_flux`: float64 1D array of estimated stellar variability model.
- `flux_error`: (Optional) float64 1D array of 1-sigma normalized flux uncertainties.
- `exclusion_flags`: int32 1D array of per-cadence quality exclusion bitmasks (`0` = valid).
- `bls_periods`: float64 1D array of trial periods searched in Box Least-Squares.
- `bls_powers`: float64 1D array of BLS periodogram powers.
- `ls_periods`: float64 1D array of trial periods searched in Lomb-Scargle.
- `ls_powers`: float64 1D array of Lomb-Scargle variability powers.
- `cand_XX_trapezoid_model`: float64 1D array of fitted exposure-integrated trapezoid flux for candidate `XX`.
- `cand_XX_batman_model`: (Optional) float64 1D array of fitted limb-darkened `batman` transit flux for candidate `XX`.
- `tpf_stamp` / `cand_XX_tpf_stamp`: (Optional) float64 2D array of mean out-of-transit target aperture pixels.
- `tpf_diff` / `cand_XX_tpf_diff`: (Optional) float64 2D array of difference image (`OOT - In-transit`) flux.

### 3. `candidates.parquet`
Apache Parquet columnar table containing all fitted parameters, dilution-corrected depth, calibrated class probabilities (`prob_transit_like`, `prob_eclipse_like`, `prob_stellar_variability`, `prob_other_artifact`), and spatial vetting diagnostics (`centroid_shift_sigma`, `crowdsap_value`, `sap_to_pdcsap_depth_ratio`, `difference_image_offset_pixels`, `aperture_depth_growth_ratio`) for each detected candidate.
