"""Deterministic distinct BLS peak extraction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from transiteye.config import BlsSettings
from transiteye.detection.bls import BlsResult
from transiteye.identifiers import make_candidate_id


def extract_peaks(result: BlsResult, settings: BlsSettings) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index in np.argsort(result.periodogram.power.to_numpy())[::-1]:
        row = result.periodogram.iloc[index]
        period = float(row.period)
        relation = None
        ratio = None
        for stronger in rows:
            stronger_period = stronger["period"]
            if not isinstance(stronger_period, (int, float)):
                raise TypeError("Internal peak period must be numeric.")
            r = period / float(stronger_period)
            if abs(r - 1) < settings.local_peak_fraction:
                relation = "near_duplicate"
                break
            for integer in (2, 3):
                if (
                    abs(r - integer) < settings.harmonic_tolerance
                    or abs(r - 1 / integer) < settings.harmonic_tolerance
                ):
                    relation = "harmonic"
                    ratio = r
                    break
        if relation == "near_duplicate":
            continue
        rank = len(rows) + 1
        rows.append(
            {
                "candidate_id": make_candidate_id(
                    observation_group_id=result.observation_group_id,
                    bls_config_hash=result.config_hash,
                    rank=rank,
                ),
                "rank": rank,
                "period": period,
                "duration": float(row.duration),
                "epoch": float(row.epoch),
                "power": float(row.power),
                "depth": float(row.depth),
                "relation_to_stronger": relation,
                "harmonic_ratio": ratio,
            }
        )
        if len(rows) == settings.top_k:
            break
    return pd.DataFrame(rows)
