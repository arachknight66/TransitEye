# Architecture

TransitEye is an artifact-oriented pipeline with two input paths and shared
candidate-level analysis. The scientific path preserves real observations and
produces unlabeled scores. The demo path creates controlled truth on copies of
real TESS substrates. Both paths use the same preprocessing, detection, feature,
and inference contracts.

![System architecture](../results/figures/figure_1_system_architecture.png)

## Layers

The **catalog layer** freezes a NASA Exoplanet Archive TOI snapshot, typed
dispositions, and a pilot cohort. The **acquisition layer** freezes TESS-SPOC
MAST product discovery, selection receipts, raw bytes, and checksums. Live
external state is never read implicitly during analysis.

The **preprocessing layer** reads immutable FITS products, applies the declared
quality, finite-value, segmentation, normalization, and detrending policies, and
records output lineage. It preserves gaps and downward transit-like excursions.

The **detection layer** performs a blind BLS search. Ranked peaks become stable
candidate identities. Catalog ephemerides or injection truth are not supplied to
the search. Post-detection matching evaluates fundamental and documented
harmonic relationships after candidates have been frozen.

The **dataset layer** separates candidate facts, catalog or synthetic events,
matches, labels, and artifact lineage. The candidate event is the ML unit because
one light curve can contain several proposed periods and each proposal has its
own detector and folded-shape measurements.

The **demo layer** injects deterministic planet-like, eclipsing-binary-like, and
sinusoidal signals into in-memory copies of real TESS substrates. Controls and
unmatched BLS candidates remain unlabeled. The demo never writes synthetic truth
into the scientific dataset.

The **feature layer** extracts registered time-domain, BLS, Lomb–Scargle, and FFT
features. Truth, labels, matches, split assignments, and dataset roles are
excluded from the extractor schema. Undefined values remain missing until
training-only transformations are fit.

The **model layer** uses TIC-grouped train, validation, and locked-test splits.
TIC is the split unit because candidate rows from the same star share noise,
systematics, products, and injected variants. Imputation and scaling are learned
from training TICs only.

The **evaluation layer** preserves stage separation: BLS event recovery measures
whether an injected event became a matching candidate; ML classification
measures correctness only where a labeled candidate exists. Grouped robustness,
calibration, interpretability, domain shift, and readiness gating wrap the
frozen official model.

The **reproducibility/report layer** inventories immutable artifacts, verifies
checksums and lineage, and builds deterministic tables and figures. It does not
duplicate acquisition, preprocessing, BLS, features, modeling, or evaluation.

## Truth firewall

```text
catalog ephemerides / synthetic truth
                 |
                 +----> post-detection matching, labels, evaluation

raw flux ----> preprocessing ----> blind BLS ----> features ----> model
```

Truth metadata may select demo examples for analysis, but it never enters model
features or fitted transformations. An unmatched candidate is unknown, not a
negative example.

## Artifact boundaries

Every material stage records a version or policy hash and parent identities.
The compact major-version graph is maintained in
[artifact_lineage.md](artifact_lineage.md); the complete portable inventory is
in the reproducibility package referenced by `results/index.json`.
