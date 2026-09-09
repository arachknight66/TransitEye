"""Reusable non-scientific preprocessing diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from transiteye.preprocessing.pipeline import LightCurveRecord


def plot_preprocessing_qa(record: LightCurveRecord, destination: str | Path) -> Path:
    """Plot raw/masked/normalized/trend/detrended streams without transit annotations."""
    frame = record.data
    fig, axes = plt.subplots(5, 1, sharex=True, figsize=(10, 10))
    series = (
        ("pdcsap_flux", "raw PDCSAP"),
        ("valid", "valid mask"),
        ("normalized_flux", "normalized"),
        ("trend", "trend"),
        ("detrended_flux", "detrended"),
    )
    for axis, (column, label) in zip(axes, series, strict=True):
        axis.plot(frame.time, frame[column], ".", ms=1)
        axis.set_ylabel(label)
    axes[-1].set_xlabel("TIME")
    fig.tight_layout()
    out = Path(destination)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
