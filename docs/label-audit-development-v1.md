# Exploratory label audit: historical v1 TRAIN partition

This audit examines the 597 TRAIN records in the already-open historical real-catalog v1 cohort.
It does not inspect the separate eclipsing-binary holdout and does not freeze a new evaluation
protocol. Counts are based on the recorded evidence identifiers and notes; they are not a visual
or source-local adjudication of the light curves. The v1 test was previously opened and cannot be
used as a fresh independent test of the current pipeline.

| Stored training class | Records | Audit finding |
|---|---:|---|
| transit-like | 175 | Catalog-disposition proxy; the strongest detected signal still needs source-local matching. |
| eclipse-like | 150 | All 150 carry deep-false-positive proxy notes; depth and an FP disposition alone do not prove an eclipsing binary. |
| stellar variability | 142 | All 142 are tagged as ATL targets; ATL membership is not proof of variability observed in the selected light curve. |
| other/artifact | 130 | 63 notes describe TOI false alarms; 67 describe low-amplitude false-positive proxies. The notes do not independently establish an instrumental cause. |

The historical cohort-builder code populated `verified` or `reviewed` strings automatically.
Those strings do not demonstrate that a person independently reviewed each source or the candidate
selected by TransitEye. The [NASA TOI column definitions](https://exoplanetarchive.ipac.caltech.edu/docs/API_TOI_columns.html)
define the dispositions used by the builder; they do not equate a deep FP with a confirmed binary.
The [ATL catalog description](https://cdsarc.cds.unistra.fr/viz-bin/ReadMe/J/ApJS/241/12?format=html&tex=true)
describes a prioritized target list based on predicted oscillation detectability, not verified
variability in every TESS observation.

The older synthetic builders also generated placeholder `SPOC-DRN-...` review identifiers. None
appears in this 597-source TRAIN subset; the repository's manifest notes say 37 v2 placeholders
were excluded when creating v1. The legacy builders and previously opened real-cohort evaluator now
fail explicitly instead of generating new unsupported cohorts or rerunning the old test. Their
pre-change source and checksums are retained in the external workspace; historical manifests and
results have not been rewritten.

Reproduce the counts from the repository root using the exact checksummed v1 manifests:

```bash
uv run python -m transiteye.evaluation.label_audit \
  manifests/real_catalog_cohort.v1.labels.json \
  manifests/real_catalog_cohort.v1.splits.json \
  .transiteye-workspace/remediation/r4-label-audit-v1/reproduction.json
```

The audit tool accepts only those previously opened v1 manifest paths and hashes, aggregates TRAIN
records only, and rejects duplicate or unpartitioned source groups. Its output is not a replacement
label manifest. Source-local evidence review, an auditable label policy, a frozen independent
selection protocol, and appropriately separated evaluation sources remain required before any new
classifier performance claim or opening the sealed EB holdout.
