"""Synchronized raw and detrended light curve plot widget using PyQtGraph."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from transiteye_desktop.bundle_loader import LoadedBundle
from transiteye_desktop.downsample import min_max_downsample
from transiteye_desktop.widgets.safe_graphics import SafeInfiniteLine


class LightCurvePlotWidget(QWidget):
    """Displays raw light curve with biweight trend and detrended normalized flux."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # PyQtGraph GraphicsLayoutWidget
        self.glw = pg.GraphicsLayoutWidget()
        layout.addWidget(self.glw)

        # Subplot 1: Raw Flux & Trend
        self.p_raw = self.glw.addPlot(row=0, col=0, title="Raw Flux (PDCSAP/SAP) & Estimated Stellar Trend")
        self.p_raw.showGrid(x=True, y=True, alpha=0.3)
        self.p_raw.setLabel("left", "Flux")
        self.p_raw.setLabel("bottom", "Time (BJD - 2457000)")

        # Subplot 2: Detrended Flux
        self.glw.nextRow()
        self.p_detrend = self.glw.addPlot(row=1, col=0, title="Detrended Normalized Flux")
        self.p_detrend.showGrid(x=True, y=True, alpha=0.3)
        self.p_detrend.setLabel("left", "Relative Flux")
        self.p_detrend.setLabel("bottom", "Time (BJD - 2457000)")

        # Link X axes for synchronized pan and zoom
        self.p_detrend.setXLink(self.p_raw)

        # Plot curves
        self.raw_curve = self.p_raw.plot(pen=pg.mkPen(color=(160, 160, 160), width=1), name="Raw")
        self.trend_curve = self.p_raw.plot(pen=pg.mkPen(color=(255, 80, 80), width=2), name="Trend")
        self.masked_scatter = pg.ScatterPlotItem(size=4, pen=None, brush=pg.mkBrush(255, 50, 50, 120))
        self.p_raw.addItem(self.masked_scatter)

        self.detrend_curve = self.p_detrend.plot(pen=pg.mkPen(color=(70, 140, 240), width=1), name="Detrended")
        self.baseline_line = SafeInfiniteLine(pos=1.0, angle=0, pen=pg.mkPen(color=(120, 120, 120), style=pg.QtCore.Qt.PenStyle.DashLine))
        self.p_detrend.addItem(self.baseline_line)

        # Transit highlight regions
        self._highlight_items: list[pg.LinearRegionItem] = []

    def set_bundle(self, bundle: LoadedBundle) -> None:
        """Render light curve data from a loaded bundle."""
        self.clear_highlights()
        t = bundle.times
        if len(t) == 0:
            self.clear()
            return

        # Downsample for 60fps display if array is large
        t_raw, f_raw = min_max_downsample(t, bundle.raw_flux, target_points=4000)
        t_trend, f_trend = min_max_downsample(t, bundle.trend_flux, target_points=4000)
        t_det, f_det = min_max_downsample(t, bundle.detrended_flux, target_points=4000)

        self.raw_curve.setData(t_raw, f_raw)
        self.trend_curve.setData(t_trend, f_trend)
        self.detrend_curve.setData(t_det, f_det)

        # Show invalid quality cadences as red dots
        bad_mask = ~bundle.quality_mask
        if len(bad_mask) == len(t) and np.any(bad_mask):
            self.masked_scatter.setData(t[bad_mask], bundle.raw_flux[bad_mask])
        else:
            self.masked_scatter.clear()

        self.p_raw.autoRange()
        self.p_detrend.autoRange()

    def highlight_candidate_transits(
        self,
        period: float,
        epoch: float,
        duration_hours: float,
        t_min: float,
        t_max: float,
    ) -> None:
        """Draw semi-transparent transit windows along the time axis."""
        self.clear_highlights()
        if period <= 0 or duration_hours <= 0:
            return

        dur_days = duration_hours / 24.0
        k_min = int(np.floor((t_min - epoch) / period))
        k_max = int(np.ceil((t_max - epoch) / period))

        for k in range(k_min, k_max + 1):
            t_center = epoch + k * period
            if t_min <= t_center <= t_max:
                region = pg.LinearRegionItem(
                    values=[t_center - dur_days / 2.0, t_center + dur_days / 2.0],
                    brush=pg.mkBrush(255, 200, 50, 40),
                    movable=False,
                )
                self.p_detrend.addItem(region)
                self._highlight_items.append(region)

    def clear_highlights(self) -> None:
        """Remove candidate transit windows."""
        for item in self._highlight_items:
            self.p_detrend.removeItem(item)
        self._highlight_items.clear()

    def clear(self) -> None:
        """Clear all curves."""
        self.raw_curve.clear()
        self.trend_curve.clear()
        self.detrend_curve.clear()
        self.masked_scatter.clear()
        self.clear_highlights()
