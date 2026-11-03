# Physical-model contact diagnostic, v1

This development-only diagnostic investigates the native `ellpic_bulirsch` convergence warnings
seen in the exposure study. It evaluates instantaneous Batman quadratic-limb-darkened transit
flux at mid-transit on a frozen grid of three planet-to-star radius ratios (0.02, 0.1 and 0.3) and
seven projected separations around internal contact, `z = 1 - radius_ratio`. The offsets range
from -0.000001 to +0.000001 stellar radii. Each case runs in a separate process so native output is
retained and a native warning makes the case unusable, even when it returns a finite value.

The comparison integrates the intensity blocked on stellar annuli using SciPy adaptive quadrature;
the fraction of each annulus covered by the planet is calculated geometrically. It uses the actual
projected separation returned by Batman, avoiding comparisons at slightly different floating-point
positions. A separate test checks the annular calculation against the analytic circle-overlap area
for a uniform star. The radial calculation is independent of Batman's elliptic-integral flux
calculation, but shares the assumed geometry and quadratic intensity law.

Reproduce with an unused output directory:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run python -m transiteye.evaluation.physical_contact \
  configs/physical_contact_diagnostic.v1.toml \
  .transiteye-workspace/remediation/r4-physical-contact-v1-reproduction
```

The frozen acceptance threshold was 0.01 ppm. On Python 3.11.16, Batman 2.5.3, NumPy 2.4.6 and
SciPy 1.17.1, all 18 cases without warnings met that threshold; the largest absolute difference was
0.001032 ppm. Each of the three nominal exact-contact cases emitted the native convergence warning
and was marked unusable. Their returned fluxes happened to agree with the annular result within
numerical precision, but that agreement does not establish that the native calculation converged.
The diagnostic consequently exits nonzero. All 21 rows, warnings, versions, configuration hash,
backend fingerprint and a source archive are preserved in the external workspace at
`remediation/r4-physical-contact-v1-final/`.

This is an instantaneous, circular-orbit check at one limb-darkening setting, not validation of
finite-exposure integration, optimizer convergence, physical parameters, interval coverage, or all
geometries. The production five-sample model remains explicitly flagged as unvalidated; no
production fitting behavior or result schema changed. The added evaluation module changes the
backend code fingerprint of subsequently generated results.
