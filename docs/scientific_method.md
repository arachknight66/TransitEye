# Scientific method

## Research objective

TransitEye studies whether blind transit-like candidates in TESS light curves
can be represented and vetted with a reproducible classical machine-learning
pipeline. The pilot separates controlled validation from scientific inference:
synthetic events support measured demo performance, while real candidates remain
unlabeled until independent evidence exists.

## Data sources and catalog policy

The observational source is the TESS-SPOC light-curve product family retrieved
through MAST. Target and event context comes from a frozen NASA Exoplanet Archive
TOI snapshot. Raw responses, normalized records, cohort membership, discovery
receipts, selected products, and SHA-256 values are versioned.

TOI dispositions are preserved and mapped as follows: confirmed planet (CP) and
known planet (KP) support a positive catalog event; false positive (FP) and false
alarm (FA) support a negative catalog event; planet candidate (PC), ambiguous
planet candidate (APC), missing, and unexpected states remain unlabeled. A
catalog disposition labels an ML candidate only after an accepted event match.
Unmatched candidates are never treated as negatives.

The current scientific candidate cohort has no independently validated binary
ML labels. Catalog matching metadata may evaluate recovery where appropriate,
but it is not a substitute for an independent candidate-level scientific test.

## Preprocessing and blind detection

Raw products are immutable. Preprocessing applies frozen quality and
finite-value filtering, gap-aware segmentation, normalization, conservative
artifact handling, and transit-preserving detrending. Each product retains its
source checksum and configuration identity.

BLS searches a declared period-duration grid without catalog ephemerides or
synthetic truth. Local peaks receive deterministic ranks and candidate IDs.
Post-detection matching compares period, epoch, and duration, including declared
integer harmonic relationships. This order prevents truth from guiding candidate
generation.

## Recovery and classification

**BLS event recovery** asks whether a truth event has a compatible detected
candidate. A catalog or injected planet that is absent from the matching
candidate set is a BLS false negative for that recovery analysis.

**ML candidate classification** asks whether a recovered, labeled candidate is
assigned the correct demo class. Conditional classifier correctness excludes
unrecovered events. End-to-end injected-event success requires both recovery and
correct classification, but does not replace either stage metric.

## Demo design

Demo variants inject planet-like signals, eclipsing-binary-like signals,
sinusoidal confounders, or controls into real TESS substrates. All variants from
one source TIC stay in the same statistical group. Candidate truth is assigned
only through post-BLS matching. This design tests the implemented cascade under
controlled perturbations while retaining real instrumental noise and
systematics.

## Features and models

The frozen feature contract contains 25 time-domain, 16 BLS, 11 Lomb–Scargle,
and 13 FFT features. Lomb–Scargle supports irregular timestamps. FFT summaries
use only short-gap, segment-local regularization and never bridge long gaps.

The compared model families are Logistic Regression, Random Forest, RBF SVM,
and a deterministic single-hidden-layer Extreme Learning Machine. Median
imputation, scaling where required, and any all-null feature removal are fitted
on training rows only. Validation PR-AUC selects the model; validation F1 selects
its threshold. The locked test cannot change either decision.

## Grouped evaluation

The eight demo source TICs are split 4/2/2 into training, validation, and locked
test groups. Six development TICs support leave-one-TIC-out predictions and a
1,000-replicate TIC bootstrap. Candidate-level rows are not resampled as if
independent. Calibration bins are descriptive because uncertainty is dominated
by the small number of TIC groups.

## Scientific-readiness criteria

Scientific readiness requires independently labeled real candidates, both real
classes, labels across several TIC groups, acceptable demo-to-scientific feature
support, meaningful real-event BLS recovery, grouped scientific evaluation, no
leakage, and enough independent TICs for defensible statistics. Only the leakage
criterion currently passes. The demo pipeline is complete; scientific readiness
is blocked.
