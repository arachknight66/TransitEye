# TransitEye demo mode

Demo mode validates software and future ML-pipeline behavior using deterministic
synthetic signals embedded in immutable real TESS noise and systematics. It is a
controlled injected-signal benchmark, not scientific validation on the TESS
population and not evidence of real exoplanet-detection accuracy or
generalization.

The demo path reuses the scientific FITS reader, preprocessing implementation,
blind BLS configuration, peak extractor, event matcher, and candidate-oriented
dataset contract. Synthetic truth is supplied only after candidate artifacts are
frozen. Candidate labels are match-based: unmatched and control candidates remain
unlabeled.

Synthetic periods, epochs, durations, depths/amplitudes, event classes, variant
names, truth matches, labels, dispositions, and truth-source fields are
label/evaluation metadata. They are prohibited from future model-input column
selection. All variants derived from one source TIC remain one statistical group.

The frozen development design uses eight source TICs: four training, two
validation, and two locked-test TICs. Grouped leave-one-TIC-out robustness covers
the six development TICs. A larger independently labeled scientific cohort is
still required.

## What demo mode proves

Supported conclusions are that the software pipeline operates end to end,
injection recovery can be measured, candidate features and frozen-model
inference work, and grouped demo robustness can be evaluated.

Demo mode does not establish real-population accuracy, validate a discovery, or
produce a calibrated real-planet probability. Synthetic morphology is limited,
and candidates sharing a source TIC are correlated.
