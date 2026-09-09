"""Typed FITS ingestion and conservative segment-level preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits  # type: ignore[import-untyped]

from transiteye.acquisition.checksums import sha256_file
from transiteye.preprocessing.quality import QUALITY_POLICY_VERSION, quality_mask


@dataclass(frozen=True)
class LightCurveRecord:
    data: pd.DataFrame
    metadata: dict[str, object]


def read_tess_lightcurve(path: str | Path, *, observation_id: str) -> LightCurveRecord:
    """Read FITS columns without changing the source file or substituting flux streams."""
    source = Path(path)
    with fits.open(source, memmap=True) as hdul:
        table = hdul[1].data
        required = ("TIME", "PDCSAP_FLUX", "PDCSAP_FLUX_ERR", "SAP_FLUX", "SAP_FLUX_ERR", "QUALITY")
        missing = [name for name in required if name not in table.names]
        if missing:
            raise ValueError(f"Missing expected TESS columns: {missing}")
        frame = pd.DataFrame(
            {
                "time": np.asarray(table["TIME"], dtype=float),
                "pdcsap_flux": np.asarray(table["PDCSAP_FLUX"], dtype=float),
                "pdcsap_flux_err": np.asarray(table["PDCSAP_FLUX_ERR"], dtype=float),
                "sap_flux": np.asarray(table["SAP_FLUX"], dtype=float),
                "sap_flux_err": np.asarray(table["SAP_FLUX_ERR"], dtype=float),
                "quality": np.asarray(table["QUALITY"], dtype=np.int64),
            }
        )
        frame.insert(0, "cadence_row", np.arange(len(frame), dtype=int))
        header = hdul[0].header
        metadata = {
            "source_observation_id": observation_id,
            "source_raw_sha256": sha256_file(source),
            "tic_id": header.get("TICID"),
            "sector": header.get("SECTOR"),
            "exposure_seconds": header.get("TIMEDEL"),
            "quality_policy_version": QUALITY_POLICY_VERSION,
        }
    return LightCurveRecord(frame, metadata)


def preprocess(
    record: LightCurveRecord, *, gap_days: float, trend_window: int, positive_spike_mad: float
) -> LightCurveRecord:
    """Mask invalid cadences, segment gaps, robustly normalize, and save conservative trends."""
    frame = record.data.copy()
    quality_ok, quality_reason = quality_mask(frame["quality"].to_numpy())
    finite_time = np.isfinite(frame.time)
    finite_flux = np.isfinite(frame.pdcsap_flux)
    finite_err = np.isfinite(frame.pdcsap_flux_err)
    frame["quality_accepted"] = quality_ok
    frame["quality_reason"] = quality_reason
    frame["valid"] = quality_ok & finite_time & finite_flux & finite_err
    frame["rejection_reason"] = np.where(
        ~finite_time,
        "nonfinite_time",
        np.where(
            ~finite_flux,
            "nonfinite_pdcsap_flux",
            np.where(~finite_err, "nonfinite_pdcsap_error", quality_reason),
        ),
    )
    frame = frame.sort_values(["time", "cadence_row"], kind="stable").reset_index(drop=True)
    duplicate = frame.time.duplicated(keep="first")
    frame.loc[duplicate, "valid"] = False
    frame.loc[duplicate, "rejection_reason"] = "duplicate_time"
    gap = frame.time.diff().gt(gap_days).fillna(False)
    frame["segment_id"] = gap.cumsum().astype(int)
    frame["normalization_factor"] = np.nan
    frame["normalized_flux"] = np.nan
    frame["artifact_reason"] = "accepted"
    frame["trend"] = np.nan
    frame["detrended_flux"] = np.nan
    for _, idx in frame.groupby("segment_id").groups.items():
        ids = list(idx)
        valid = frame.loc[ids, "valid"]
        if not valid.any():
            continue
        median = float(frame.loc[ids, "pdcsap_flux"].loc[valid].median())
        frame.loc[ids, "normalization_factor"] = median
        norm = frame.loc[ids, "pdcsap_flux"] / median
        frame.loc[ids, "normalized_flux"] = norm
        residual = norm.loc[valid] - np.median(norm.loc[valid])
        mad = np.median(np.abs(residual))
        if mad > 0:
            spikes = norm > 1 + positive_spike_mad * 1.4826 * mad
            frame.loc[ids, "artifact_reason"] = np.where(
                spikes, "positive_spike", frame.loc[ids, "artifact_reason"]
            )
            frame.loc[ids, "valid"] &= ~spikes
        trend = norm.rolling(
            trend_window, center=True, min_periods=max(3, trend_window // 3)
        ).median()
        frame.loc[ids, "trend"] = trend
        frame.loc[ids, "detrended_flux"] = norm / trend
    return LightCurveRecord(
        frame,
        record.metadata
        | {
            "gap_days": gap_days,
            "trend_window": trend_window,
            "positive_spike_mad": positive_spike_mad,
        },
    )
