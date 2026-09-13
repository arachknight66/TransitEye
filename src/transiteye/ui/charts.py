"""Read-only matplotlib views over selected frozen candidate artifacts."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure


def candidate_figure(
    cadences: pd.DataFrame,
    periodogram: pd.DataFrame | None,
    period: float,
) -> Figure:
    """Create light-curve, phase-fold, and existing-periodogram views."""
    panels = 3 if periodogram is not None else 2
    figure, axes = plt.subplots(1, panels, figsize=(5.2 * panels, 3.4))
    time = cadences["time"].to_numpy()
    flux_column = "detrended_flux" if "detrended_flux" in cadences else "normalized_flux"
    if flux_column not in cadences:
        flux_column = next(column for column in cadences.columns if "flux" in column)
    flux = cadences[flux_column].to_numpy()
    axes[0].scatter(time, flux, s=2, alpha=0.55, color="#63b3ed")
    axes[0].set(title="Processed light curve", xlabel="Time", ylabel="Relative flux")
    phase = ((time - np.nanmin(time) + period / 2) % period) / period - 0.5
    axes[1].scatter(phase, flux, s=2, alpha=0.55, color="#8bd3c7")
    axes[1].set(title="Phase-folded candidate", xlabel="Phase", ylabel="Relative flux")
    if periodogram is not None:
        x_column = next((c for c in periodogram.columns if "period" in c), periodogram.columns[0])
        y_column = next((c for c in periodogram.columns if "power" in c), periodogram.columns[-1])
        axes[2].plot(periodogram[x_column], periodogram[y_column], color="#f6c177", linewidth=1)
        axes[2].axvline(period, color="#ef8354", linestyle="--", label="Selected candidate")
        axes[2].set(title="Frozen BLS periodogram", xlabel="Period (days)", ylabel="BLS power")
        axes[2].legend(frameon=False)
    figure.tight_layout()
    return figure
