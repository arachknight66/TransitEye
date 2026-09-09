# Label policy

## Implemented design decisions

- Labels will be sourced from a frozen, versioned catalog snapshot rather than
  from a live mutable catalog during model training.
- The snapshot query, retrieval timestamp, raw response, and checksum will be
  retained as provenance.
- Catalog-event dispositions are mapped by typed code, preserving both the
  original source value and the internal mapping.
- Catalog fields used to construct labels must never be model features.
- Unmatched candidates remain unlabeled unless a future, documented policy
  establishes another label source.

## Current class semantics

The proposed binary task is planet-like catalog-supported event versus
catalog-dispositioned non-planet event. Confirmed and known planets are planned
gold positives; false positives and false alarms are planned gold negatives.

Planetary and ambiguous candidates map to `unlabeled_secondary`, and missing or
unexpected dispositions map to explicit `unknown`. FP and FA both map to a gold
negative label but retain distinct subgroups.

## TBD decisions

- **TBD:** a BLS candidate will receive a gold label only when it can be
  associated with a catalog event using documented ephemeris-matching rules.
- **TBD:** period, epoch, duration, and harmonic matching tolerances.
- **TBD:** policy for multi-event systems and label conflicts.
- **TBD:** whether a later multiclass task is scientifically supported.

No unlabeled target should be assumed to be planet-free.
