# Preprocessing decisions

## Confirmed design decisions

- Raw observational files will be preserved unchanged.
- Every derived curve will reference its source observation and the exact
  preprocessing configuration used.
- Quality filtering, non-finite values, gaps, normalization, outlier handling,
  detrending, and transit preservation will be recorded as explicit decisions.
- Long gaps will not be silently bridged for transit detection.
- Detrending choices must be tested for transit-signal preservation before use
  in the primary experiment.

## Planned, not yet implemented

- Per-sector and gap-aware segmentation.
- Conservative normalization and artifact treatment.
- A comparison between declared flux-product/detrending variants.
- Synthetic transit-preservation tests.

## TBD scientific choices

- **TBD:** raw flux product and quality-mask policy.
- **TBD:** outlier criteria, including treatment of downward excursions.
- **TBD:** detrending algorithm and all numerical parameters.
- **TBD:** definition of a significant gap and segment-combination strategy.
- **TBD:** whether a second transit-aware detrending pass is needed.
