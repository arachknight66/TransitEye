# Limitations

## Data limitations

The pilot cohort is small. The controlled demo has eight source TICs: four for
training, two for validation, and two for the locked test. Leave-one-TIC-out
robustness uses the six development TICs only. Candidate rows from one TIC share
instrumental behavior and injected substrates, so they are correlated and do
not provide the uncertainty of an equal number of independent stars.

## Scientific label limitations

The 75 scientific candidates are unlabeled at the ML-candidate level. There are
no independently validated scientific positive or negative classes and no
labels spanning multiple TIC groups. Real candidate classification metrics and
grouped real evaluation therefore cannot be computed.

## Domain shift

Sixty-seven of 75 scientific candidates violate robust demo support on at least
one feature. The frozen demo-trained model places all 75 above threshold, with
scores from 0.415 to 0.990. These facts indicate transfer concerns; they do not
establish candidate identity. High-importance/shift analysis flags candidate
duration and an FFT-to-BLS power ratio as especially plausible transfer-risk
drivers, without asserting causality.

## Demo limitations

Injected morphology and parameter distributions are simplified and do not span
the TESS population, all stellar variability, blending, centroid behavior, or
instrumental artifacts. Real TESS substrates improve noise realism but do not
make synthetic event labels equivalent to independent real labels. Controls and
unmatched candidates remain unlabeled.

## Model and calibration limitations

The Random Forest was selected within one compact development design. Its scores
are ranking and triage outputs, not scientific planet probabilities. Development
OOF ECE is about 0.131, validation ECE about 0.169, and descriptive locked-test
ECE about 0.192. Curves and bootstrap ranges are unstable with only six
independent development groups. No calibrated replacement model was fitted.

## Detection limitations

BLS recovery varies by synthetic family from 72.7% to 100%. An unrecovered event
never reaches the classifier and is an end-to-end false negative. Conditional
classifier results must therefore be read alongside recovery and combined
success, rather than as a replacement for the detection stage.

## Claim boundary

Current evidence supports controlled synthetic-injection validation, grouped
demo robustness, injected-event recovery, model-generated scores, and
demo-to-scientific covariate-shift diagnostics. It does not support population
generalization, discovery confirmation, validated real candidate classification,
or calibrated scientific planet probabilities.
