# Evaluation

TransitEye uses candidate-level metrics with TIC-level separation. Validation
selects the model and threshold; the two-TIC test is locked; leave-one-TIC-out
predictions cover the six development TICs. TIC bootstrap intervals preserve the
correlation of candidates from the same source.

The benchmark hierarchy is: locked test as the declared holdout, grouped
development robustness as the stronger stability summary, TIC-bootstrap
uncertainty, injection recovery, then domain-shift limitations. The perfect
two-TIC test is always reported with its small-group context.

BLS recovery, classifier correctness conditional on recovery, and end-to-end
controlled success are distinct quantities. Calibration is diagnostic and does
not replace the frozen model. Permutation importance is the primary global
interpretation; impurity importance is secondary. Local perturbation
explanations describe features contributing to a score, including for unlabeled
scientific examples, without establishing astrophysical status.

The authoritative generated summaries are `results/tables/`,
`results/figures/`, and the final evaluation referenced by
`results/index.json`. Scientific readiness is assessed against eight explicit
criteria and is currently blocked.
