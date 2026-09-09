from __future__ import annotations

import numpy as np
import pandas as pd

from transiteye.preprocessing.pipeline import LightCurveRecord, preprocess
from transiteye.preprocessing.qa import plot_preprocessing_qa


def test_synthetic_box_transits_are_preserved() -> None:
    time = np.arange(1000, dtype=float) / 100
    flux = 1_000 + 0.2 * time
    injected = np.zeros(1000, dtype=bool)
    for start in (200, 500, 800):
        injected[start : start + 10] = True
    flux[injected] *= 0.99
    record = LightCurveRecord(
        pd.DataFrame(
            {
                "cadence_row": np.arange(1000),
                "time": time,
                "pdcsap_flux": flux,
                "pdcsap_flux_err": np.ones(1000),
                "sap_flux": flux,
                "sap_flux_err": np.ones(1000),
                "quality": np.zeros(1000, dtype=int),
            }
        ),
        {},
    )
    result = preprocess(record, gap_days=0.5, trend_window=101, positive_spike_mad=8)
    depth = 1 - np.nanmedian(result.data.loc[injected, "detrended_flux"])
    assert abs(depth - 0.01) < 0.003
    assert result.data.loc[injected, "valid"].sum() == 30


def test_nonfinite_and_gaps_are_explicit() -> None:
    record = LightCurveRecord(
        pd.DataFrame(
            {
                "cadence_row": [0, 1, 2],
                "time": [0.0, np.nan, 2.0],
                "pdcsap_flux": [1.0, 1.0, np.nan],
                "pdcsap_flux_err": [1.0, 1.0, 1.0],
                "sap_flux": [1.0, 1.0, 1.0],
                "sap_flux_err": [1.0, 1.0, 1.0],
                "quality": [0, 0, 0],
            }
        ),
        {},
    )
    result = preprocess(record, gap_days=0.5, trend_window=5, positive_spike_mad=8)
    assert {"nonfinite_time", "nonfinite_pdcsap_flux"}.issubset(set(result.data.rejection_reason))


def test_qa_plot_is_written(tmp_path) -> None:
    record = LightCurveRecord(
        pd.DataFrame(
            {
                "time": [0.0, 1.0],
                "pdcsap_flux": [1.0, 1.0],
                "valid": [True, True],
                "normalized_flux": [1.0, 1.0],
                "trend": [1.0, 1.0],
                "detrended_flux": [1.0, 1.0],
            }
        ),
        {},
    )
    assert plot_preprocessing_qa(record, tmp_path / "qa.png").exists()
