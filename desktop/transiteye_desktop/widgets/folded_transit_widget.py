"""Phase-folded transit profile and residuals visualization widget."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from transiteye_desktop.bundle_loader import LoadedCandidate


class FoldedTransitWidget(QWidget):
    """Displays phase-folded flux around mid-transit with backend-fitted model and residuals."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.glw = pg.GraphicsLayoutWidget()
        layout.addWidget(self.glw)

        # Subplot 1: Phase-folded Transit & Model
        self.p_fold = self.glw.addPlot(row=0, col=0, title="Phase-Folded Transit Profile")
        self.p_fold.showGrid(x=True, y=True, alpha=0.3)
        self.p_fold.setLabel("left", "Relative Flux")
        self.p_fold.setLabel("bottom", "Time from Mid-Transit (hours)")

        # Subplot 2: Residuals
        self.glw.nextRow()
        self.p_res = self.glw.addPlot(row=1, col=0, title="Transit Fit Residuals (O - C)")
        self.p_res.showGrid(x=True, y=True, alpha=0.3)
        self.p_res.setLabel("left", "Residual")
        self.p_res.setLabel("bottom", "Time from Mid-Transit (hours)")

        self.p_res.setXLink(self.p_fold)

        # Plot items
        self.scatter_raw = pg.ScatterPlotItem(
            size=4, pen=None, brush=pg.mkBrush(140, 180, 240, 90)
        )
        self.scatter_binned = pg.ScatterPlotItem(
            size=7, pen=pg.mkPen(255, 255, 255, 180), brush=pg.mkBrush(20, 100, 220, 220)
        )
        self.model_curve = self.p_fold.plot(
            pen=pg.mkPen(color=(255, 60, 60), width=2.5), name="Trapezoid Fit"
        )
        self.batman_curve = self.p_fold.plot(
            pen=pg.mkPen(color=(255, 200, 60), width=2.0, style=pg.QtCore.Qt.PenStyle.DashLine),
            name="Batman Fit",
        )

        self.p_fold.addItem(self.scatter_raw)
        self.p_fold.addItem(self.scatter_binned)

        self.res_scatter = pg.ScatterPlotItem(
            size=4, pen=None, brush=pg.mkBrush(160, 160, 160, 90)
        )
        self.res_zero = pg.InfiniteLine(
            pos=0.0,
            angle=0,
            pen=pg.mkPen(color=(120, 120, 120), style=pg.QtCore.Qt.PenStyle.DashLine),
        )
        self.p_res.addItem(self.res_scatter)
        self.p_res.addItem(self.res_zero)

    def set_candidate(
        self,
        candidate: LoadedCandidate,
        times: np.ndarray,
        detrended_flux: np.ndarray,
    ) -> None:
        """Fold detrended light curve and backend-fitted models on candidate period and epoch."""
        if len(times) == 0:
            self.clear()
            return

        p = max(1e-6, candidate.period_days)
        t0 = candidate.epoch_days
        dur_h = candidate.duration_hours

        # Phase in hours from mid-transit: [-P/2, +P/2] * 24
        phase_days = (times - t0 + 0.5 * p) % p - 0.5 * p
        phase_hours = phase_days * 24.0

        # Zoom window: +/- 3.5 * duration
        window_h = max(2.0, 3.5 * dur_h)
        mask = (phase_hours >= -window_h) & (phase_hours <= window_h)
        if not np.any(mask):
            self.clear()
            return

        w_phase = phase_hours[mask]
        w_flux = detrended_flux[mask]

        # Sort by phase
        sort_idx = np.argsort(w_phase)
        w_phase = w_phase[sort_idx]
        w_flux = w_flux[sort_idx]

        self.scatter_raw.setData(w_phase, w_flux)

        # Compute binned phase curve for visual clarity
        nbins = 35
        bin_edges = np.linspace(-window_h, window_h, nbins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_means = []
        valid_centers = []

        for b_lo, b_hi, b_c in zip(bin_edges[:-1], bin_edges[1:], bin_centers, strict=False):
            b_mask = (w_phase >= b_lo) & (w_phase < b_hi)
            if np.any(b_mask):
                bin_means.append(float(np.median(w_flux[b_mask])))
                valid_centers.append(b_c)

        if valid_centers:
            self.scatter_binned.setData(np.asarray(valid_centers), np.asarray(bin_means))
        else:
            self.scatter_binned.clear()

        # Render backend-fitted trapezoid and optional batman curves
        trap_arr = candidate.trapezoid_model_curve
        bat_arr = candidate.batman_model_curve

        if trap_arr is not None and len(trap_arr) == len(times):
            w_trap = trap_arr[mask][sort_idx]
            self.model_curve.setData(w_phase, w_trap)
            primary_model_at_data = w_trap
        else:
            self.model_curve.clear()
            primary_model_at_data = np.ones_like(w_flux)

        if bat_arr is not None and len(bat_arr) == len(times):
            w_bat = bat_arr[mask][sort_idx]
            self.batman_curve.setData(w_phase, w_bat)
            primary_model_at_data = w_bat
        else:
            self.batman_curve.clear()

        res = w_flux - primary_model_at_data
        self.res_scatter.setData(w_phase, res)

        self.p_fold.setXRange(-window_h, window_h, padding=0.05)
        self.p_res.setXRange(-window_h, window_h, padding=0.05)

    def clear(self) -> None:
        """Clear all phase-folded plots."""
        self.scatter_raw.clear()
        self.scatter_binned.clear()
        self.model_curve.clear()
        self.batman_curve.clear()
        self.res_scatter.clear()
