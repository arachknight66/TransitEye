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

Confirmed development decision: the MVP uses one development partition because
only eight source TICs are available. A statistically meaningful final split and
final scientific cohort remain TBD.
