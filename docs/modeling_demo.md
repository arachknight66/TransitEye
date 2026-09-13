# Development-demo modeling policy (B040–B045)

Status: frozen development policy. These results validate the software path on controlled
injections and are not evidence of scientific generalization.

The split unit is the source TIC. The eight demo TICs are deterministically partitioned 4/2/2
into train, validation, and locked test groups using a named seed derived from the project master
seed. Every candidate, variant, observation group, and raw checksum associated with a TIC follows
that assignment. Only `positive` and `negative` gold demo candidates enter supervised fitting;
unlabeled and ambiguous rows retain their split assignment but are excluded from model fitting and
metrics.

For every feature ablation and model, training medians are learned from training rows only.
Features that are entirely null in training are removed using the frozen `drop` policy. Logistic
Regression, RBF SVM, and ELM additionally receive training-only standardization. Random Forest
receives median-imputed but unscaled values. No missingness indicators, oversampling, class
weighting, or label-driven feature selection are used.

The ELM uses one deterministic random `tanh` hidden layer. With hidden response matrix `H`, binary
targets encoded as −1/+1, and regularization `lambda`, output weights are
`(H.T H + lambda I)^−1 H.T y`. It uses no iterative neural-network framework.

One fixed baseline configuration is evaluated for each of Logistic Regression, Random Forest,
RBF SVM, and ELM. Validation PR-AUC selects the model. The highest score threshold among ties that
maximize validation F1 is then frozen. The two predeclared feature ablations are finalized before
each locked test evaluation. Test results must not cause model, threshold, features, or policy
changes in this run.

The frozen inference function accepts any feature table satisfying the selected registry schema;
it does not inspect dataset role. Scores on the unlabeled scientific-development matrix are only
an interface smoke test and must not be interpreted as detections or performance estimates.
