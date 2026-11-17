# TransitEye manifests

This directory stores small, immutable, reviewable definitions of inputs and partitions. It must not
contain downloaded FITS products, feature matrices, model artifacts, or evaluation outputs.

Phase 1 provides schemas and validation code for:

- Product manifests: exact MAST product inventories, query parameters, checksums, and local names.
- Label snapshots: source-level morphology labels with public evidence, review state, ambiguity, and
  catalog snapshot date.
- Split manifests: one source group per partition, including all sectors and derivatives.

The frozen Phase 1 pilot cohort consists of:

- `pilot_labels.v1.json`: 68 curated source-level morphology labels (17 per class across `transit_like`,
  `eclipse_like`, `stellar_variability`, and `other_artifact`, with 4 explicit ambiguous audit cases),
  each with public evidence HTTP(S) URIs, sectors, and review status.
- `pilot_splits.v1.json`: Source-grouped partition manifest assigning each source group to `train` (36),
  `calibration` (8), `selection` (8), `test` (8), or `sector_shift` (8), ensuring zero source leakage.
- `pilot_products.v1.json`: Immutable MAST discovery product inventory for all 68 targets across their
  observed sectors (78 FITS products).

The real-data expansion assembled on 2026-10-29 is documented separately from the synthetic cohort experiments:

- `real_catalog_cohort.v1.labels.json`: 863 catalog-labelled TIC sources with evidence URIs and explicit catalog provenance. Counts are 250 `transit_like`, 221 `eclipse_like`, 208 `stellar_variability`, and 184 `other_artifact`; the set is not class-balanced. The 37 v2 `manual_review` records were excluded because they were generated review placeholders rather than externally sourced labels.
- `real_catalog_cohort.v1.splits.json`: source-grouped train/calibration/selection/test assignments filtered from the v2 split snapshot; every sector for a TIC remains in that TIC's partition.
- `real_catalog_cohort.v1.products.json`: 881 genuine MAST SPOC light-curve products from 863 distinct targets, including 18 additional real sectors downloaded during acquisition. Every entry has a SHA-256 checksum and can be resolved under `.transiteye-workspace/data/real_catalog_cohort_v1/`.

The FITS files are kept outside Git. Each downloaded product was checked for a readable light-curve table and matching TIC ID and sector header. Catalog labels are evidence-backed proxies and are not equivalent to independent visual review of every light curve. The real-cohort model was trained using source-grouped train, calibration, and selection partitions, then evaluated once on the locked test split under `configs/evaluation_real_catalog.v1.toml`. Treat the result as performance against these catalog-derived proxies, not as a final scientifically adjudicated estimate. The synthetic cohort v2/v3 benchmark scores are not valid real-data performance estimates.

Workspace cleanup on 2026-10-31 removed the generated v2/v3 FITS datasets and the v3 feature cache. The v2 build mixed downloaded products with class-signal-synthesized light curves, and v3 was generated from synthetic class-shaped light curves. Fifty-eight original MAST downloads from the v2 acquisition were retained separately, with checksums, in `.transiteye-workspace/data/cohort_v2_mast_originals_v1/` and `cohort_v2_mast_originals.v1.products.json`. The original v2/v3 labels and product manifests remain as historical provenance; they no longer describe complete local datasets. The pilot and real catalog cohort datasets were retained.

A separate real-data expansion was acquired on 2026-10-31:

- `real_catalog_cohort.v2.labels.json`: 911 new TIC source groups from a fresh NASA Exoplanet Archive TOI query; 541 `transit_like` disposition proxies, 335 `other_artifact` false-alarm/false-positive proxies, and 35 deep-false-positive `eclipse_like` proxies. These are catalog disposition/depth proxies, not independently adjudicated light-curve morphology labels. `stellar_variability` is absent from this expansion.
- `real_catalog_cohort.v2.splits.json`: a frozen source-group split (409 train, 183 calibration, 183 selection, 136 test; manifest ID `1860a4c238736c2e`). All targets are new relative to the v1 cohort. The test partition has not been opened.
- `real_catalog_cohort.v2.products.json`: 911 original MAST TESS light curves with SHA-256 and checked TIC/sector headers, stored under `.transiteye-workspace/data/real_catalog_cohort_v2/`. Of 918 discovered products, seven downloads failed and are listed in `real_catalog_cohort.v2.acquisition_report.json`.

Model fitting on this expansion is currently blocked by the rare eclipse-like proxy class. The feature pipeline yielded only 14 eclipse examples in train, eight in calibration, and eight in selection. Fourteen training examples do not support the existing five-fold stratified grouped cross-validation safely, so the trainer's guard stopped before fitting. Do not treat this cohort as a complete four-class retraining or use the 136-source test split for tuning.

A separate catalog-backed EB tranche was acquired from the MAST TESS-EBs v1.0 HLSP on 2026-10-31:

- `real_catalog_tess_ebs.v1.labels.json`: 246 unique TIC source groups labelled `eclipse_like` from the curated catalog, after excluding TICs already present in the real cohorts. These labels identify known EB systems; they do not encode source location or guarantee that the strongest automatic event in a SPOC light curve is the catalogued eclipse.
- `real_catalog_tess_ebs.v1.products.json`: 246 original SPOC light curves with SHA-256 checksums and catalog-matched sectors; files reside under `.transiteye-workspace/data/real_catalog_tess_ebs_v1/`. The raw MAST catalog and checksum are also preserved there. Five targets in the 500-target bright-source tranche had no matching SPOC product; the remaining tranche targets were not queried after the MAST service stalled.
- `real_catalog_tess_ebs.v1.splits.json`: frozen TIC-grouped split (177 train, 20 calibration, 22 selection, 27 test; ID `29402c8c2b85b771`). Its test partition was not opened.
- `real_catalog_tess_ebs.v1.feature_audit.json`: the fixed pipeline produced finite features for 217/219 development sources (175 train, 20 calibration, 22 selection); two train fit initializations failed. The 27-source holdout remains unopened.

The retraining snapshot combines the v1 real-catalog partitions and the new EB tranche while preserving each source group's original assignment:

- `real_catalog_cohort.v3.labels.json`, `real_catalog_cohort.v3.products.json`, and `real_catalog_cohort.v3.splits.json` identify 1,109 sources and 1,127 product records; split ID `ed040a75524d3c72`, product manifest ID `3dc2e59d579c4327`. Combined partitions contain 774 train, 104 calibration, 108 selection, and 123 test sources. `.transiteye-workspace/data/real_catalog_cohort_v3/` contains links only for development products; neither test partition was read during training.
- `morphology_real_catalog_v3` was trained without synthetic augmentation or label-conditioned feature edits. It used 772 finite training vectors, 104 calibration vectors, and 108 selection vectors. Usable supports were train: 325 eclipse, 130 artifact, 142 stellar-variability, 175 transit; calibration: 44, 14, 21, 25; selection: 44, 18, 21, 25. Two EB train fit failures are recorded in the report.
- The selected random forest achieved grouped-CV macro-F1 `0.692` and transit-like recall `0.617`; these are training-partition model-selection results, not independent performance estimates. Calibration-set Brier `0.435` and ECE `0.077` are fit-partition diagnostics only. No test split has been evaluated for this candidate.

To restore the v3 development inputs in a clean workspace, run
`python3 scripts/download_training_cohort.py`. The script requests every product through the MAST
Portal Download API (the portal may redirect internally to MAST object storage), deterministically
selects the latest sector per source, acquires only the 774 train, 104 calibration, and 108 selection
sources, and excludes all 123 test sources before making archive requests. It writes FITS files under the ignored
`.transiteye-workspace/data/real_catalog_cohort_v3_dev/`, a development-only product manifest, and
an acquisition report with each outcome under `.transiteye-workspace/data/`. Downloads are
resumable; each accepted file must match the frozen manifest SHA-256 and its FITS TIC, sector, table,
time, and flux headers. A download failure is retained in the report rather than silently omitted.
This cohort inherits the proxy-label and historical feature-pipeline limitations above.

## Unlabeled 50,000-target MAST corpus

The user-requested `mast_tess_unlabeled_50000_v1` corpus is a separate acquisition from the labeled
model cohort. Its frozen product manifest (`d1f546b027534bb7`) selects 50,000 distinct TIC targets
from MAST's official Sector 1–13 SPOC light-curve bulk inventories using the seed in
`configs/mast_unlabeled_corpus.v1.toml`. It excluded 5,403 target identities already present in
project manifests or local FITS products. The frozen product manifest keeps every morphology label
unset. A separate, versioned TOI overlay was created from the current public ExoFOP-TESS TOI CSV
(retrieved 2026-11-11; source URL and SHA-256 are in the report), joined by exact TIC ID and gated
on the TOI sector list including the product sector. It matched 139 products to 157 TOI events and
assigned 111 `transit_like` catalog proxy labels where both TESS and TFOPWG dispositions are `PC`.
The 28 products with ambiguous/conflicting dispositions and 49,861 without a TOI match remain
unlabeled; no unmatched entry was treated as a negative. Raw dispositions are retained. These
catalog proxies are not independently adjudicated morphology or confirmation that a planet exists,
and they do not change existing model partitions. Products and the 50,000-entry manifest remain outside Git under
`.transiteye-workspace/data/mast_unlabeled_50000_v1/`. `scripts/download_unlabeled_mast_corpus.py`
requests each product through the MAST Portal Download API, validates FITS target/sector/table
headers, records archive SHA-256 and failures in `acquisition_report.v1.json`, and supports resume.
The acquisition completed on 2026-11-11: all 50,000 products downloaded successfully, with zero
failures, 50,000 unique target IDs, and 95,773,392,000 bytes stored outside Git. Each product passed
structural, TIC/sector, and light-curve column checks and has a locally computed SHA-256. The FITS
PRIMARY CHECKSUM validates; LIGHTCURVE and APERTURE CHECKSUM values are invalid, and DATASUM is
absent in these archive products. Per-HDU checksum-header results are retained in the report. The
TOI overlay is stored as `labels_toi_proxy.v1.parquet`, with all TIC-matched events in
`toi_matches.v1.jsonl` and the audit in `toi_crossmatch_report.v1.json`. Catalog IDs, dispositions,
ephemerides, and TIC parameters must not be used as model features.
