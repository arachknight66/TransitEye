# Fourth-class false-alarm candidate acquisition

The `other_artifact` class requires evidence about the actual detected signal. NASA's TOI
false-alarm (`FA`) disposition means a signal was deemed a false alarm, but it does not identify
its physical cause: stellar variability, noise and instrumental effects can all produce false
alarms. A different TESS sector or the strongest automatically detected signal can also differ
from the TOI event. Therefore TOI `FA` is not an automatically valid `other_artifact` label.

A snapshot query of the NASA Exoplanet Archive TOI table returned 100 `FA` events on 99 distinct
targets. The acquisition excluded known identities in the prior six exploratory collections and
reserved source partitions, then cross-checked Gaia identities and confirmed-planet host entries.
It found 59 remaining targets with public TESS SPOC light-curve products. No sealed holdout
observations or labels were opened. Alias exclusion is limited to the available TIC/Gaia mappings.

All 59 FITS files passed target/sector, FITS data-inclusive checksum and TransitEye reader checks.
Six also matched previously recorded SHA-256 values; SHA-256 for the remaining 53 was calculated
from the downloaded MAST bytes and is not an independent archive checksum. Forty-five chosen
observation windows approximately contain the catalog original event epoch; 41 files have at least
one good-quality cadence near that epoch. The TOI table describes the epoch as BJD without
specifying an exact time scale in the queried columns; the coverage test is approximate, not an
event match. A Gaia variability cross-check gives three possible `SOLAR_LIKE` matches, subject to
cross-match and classification uncertainty.

The files, 59-row list, NASA and Gaia snapshots, acquisition and verification scripts, and
candidate-level coverage flags are in external workspace `data/false_alarm_candidates_v1/`.
These are **unadjudicated review candidates**, not 59 confirmed instrumental artifacts, and they
are not used for training. Before training a fourth class, review each actual detected event,
source localization, ephemeris match, quality flags and competing variability explanations;
freeze a source-grouped selection and evaluation protocol and retain unresolved cases as
unlabeled. Do not treat the absence of a catalog match as a negative label.
