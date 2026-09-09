"""Post-detection-only catalog ephemeris comparison."""

from __future__ import annotations

import pandas as pd

from transiteye.config import BlsSettings


def match_candidates(
    candidates: pd.DataFrame, events: pd.DataFrame, settings: BlsSettings
) -> pd.DataFrame:
    """Compare frozen candidates with catalog data; never alters BLS outputs."""
    rows = []
    for _, c in candidates.iterrows():
        for _, e in events.iterrows():
            p = float(c.period)
            q = float(e.period_days)
            ratio = p / q
            kind = "no_match"
            if abs(ratio - 1) <= settings.period_match_tolerance:
                kind = "fundamental"
            elif abs(ratio - 0.5) <= settings.period_match_tolerance:
                kind = "half_period"
            elif abs(ratio - 2) <= settings.period_match_tolerance:
                kind = "double_period"
            phase = abs(((float(c.epoch) - float(e.transit_epoch_bjd)) / q + 0.5) % 1 - 0.5)
            rows.append(
                {
                    "candidate_id": c.candidate_id,
                    "toi_id": e.toi_id,
                    "candidate_period": p,
                    "catalog_period": q,
                    "relative_period_error": abs(ratio - 1),
                    "candidate_epoch": float(c.epoch),
                    "catalog_epoch": float(e.transit_epoch_bjd),
                    "phase_error": phase,
                    "harmonic_ratio": ratio,
                    "match_type": kind,
                    "matched": kind != "no_match" and phase <= settings.phase_match_tolerance,
                }
            )
    return pd.DataFrame(rows)
