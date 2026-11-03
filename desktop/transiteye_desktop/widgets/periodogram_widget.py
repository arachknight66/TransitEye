"""BLS and Lomb-Scargle periodogram visualization widget."""

from __future__ import annotations

import math
from typing import Any

import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from transiteye_desktop.bundle_loader import LoadedBundle
from transiteye_desktop.widgets.safe_graphics import SafeInfiniteLine


class PeriodogramWidget(QWidget):
    """Plots BLS power spectra and Lomb-Scargle variability with detection peaks and harmonic aliases."""

    period_selected = pyqtSignal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.glw = pg.GraphicsLayoutWidget()
        layout.addWidget(self.glw)

        # Row 0: BLS Periodogram
        self.p_bls = self.glw.addPlot(
            row=0, col=0, title="Box Least-Squares (BLS) Search Periodogram"
        )
        self.p_bls.showGrid(x=True, y=True, alpha=0.3)
        self.p_bls.setLabel("left", "BLS Power")
        self.p_bls.setLabel("bottom", "Period (days)")
        self.p_bls.setLogMode(x=True, y=False)

        self.power_curve = self.p_bls.plot(pen=pg.mkPen(color=(100, 200, 255), width=1.5))
        self.peak_line = SafeInfiniteLine(
            pos=0,
            angle=90,
            pen=pg.mkPen(color=(255, 60, 60), width=2, style=pg.QtCore.Qt.PenStyle.DashLine),
        )
        self.subharmonic_line = SafeInfiniteLine(
            pos=0,
            angle=90,
            pen=pg.mkPen(color=(255, 180, 60), width=1.5, style=pg.QtCore.Qt.PenStyle.DotLine),
        )
        self.harmonic_line = SafeInfiniteLine(
            pos=0,
            angle=90,
            pen=pg.mkPen(color=(255, 180, 60), width=1.5, style=pg.QtCore.Qt.PenStyle.DotLine),
        )

        self.p_bls.addItem(self.peak_line, ignoreBounds=True)
        self.p_bls.addItem(self.subharmonic_line, ignoreBounds=True)
        self.p_bls.addItem(self.harmonic_line, ignoreBounds=True)

        # Row 1: Lomb-Scargle Variability Periodogram
        self.glw.nextRow()
        self.p_ls = self.glw.addPlot(
            row=1, col=0, title="Lomb-Scargle Stellar Variability Periodogram"
        )
        self.p_ls.showGrid(x=True, y=True, alpha=0.3)
        self.p_ls.setLabel("left", "LS Power")
        self.p_ls.setLabel("bottom", "Period (days)")
        self.p_ls.setLogMode(x=True, y=False)

        self.ls_curve = self.p_ls.plot(pen=pg.mkPen(color=(120, 230, 140), width=1.5))
        self.ls_peak_line = SafeInfiniteLine(
            pos=0,
            angle=90,
            pen=pg.mkPen(color=(80, 255, 120), width=1.5, style=pg.QtCore.Qt.PenStyle.DashLine),
        )
        self.p_ls.addItem(self.ls_peak_line, ignoreBounds=True)

        self._hide_marker_lines()

        # Connect click interactions
        self.glw.scene().sigMouseClicked.connect(self._on_scene_clicked)

    def _on_scene_clicked(self, event: Any) -> None:
        """Handle user clicking on the periodogram to select candidate period."""
        if event.button() != Qt.MouseButton.LeftButton:
            return
        pos = event.scenePos()
        if self.p_bls.sceneBoundingRect().contains(pos):
            view_pt = self.p_bls.vb.mapSceneToView(pos)
            period = float(10.0 ** view_pt.x())
            if period > 0:
                self.set_peak_period(period)
                self.period_selected.emit(period)
        elif self.p_ls.sceneBoundingRect().contains(pos):
            view_pt = self.p_ls.vb.mapSceneToView(pos)
            period = float(10.0 ** view_pt.x())
            if period > 0:
                self.period_selected.emit(period)

    def _hide_marker_lines(self) -> None:
        self.peak_line.setVisible(False)
        self.subharmonic_line.setVisible(False)
        self.harmonic_line.setVisible(False)
        self.ls_peak_line.setVisible(False)

    def set_bundle(self, bundle: LoadedBundle, peak_period: float | None = None) -> None:
        """Render BLS and Lomb-Scargle periodograms from bundle."""
        periods = bundle.bls_periods
        powers = bundle.bls_powers

        if len(periods) == 0 or len(powers) == 0:
            self.power_curve.clear()
            self._hide_marker_lines()
        else:
            self.power_curve.setData(periods, powers)
            if peak_period is not None and peak_period > 0:
                self.set_peak_period(peak_period)
            elif len(bundle.candidates) > 0:
                self.set_peak_period(bundle.candidates[0].period_days)
            else:
                self._hide_marker_lines()
            self.p_bls.autoRange()

        ls_periods = bundle.ls_periods
        ls_powers = bundle.ls_powers
        if len(ls_periods) > 0 and len(ls_powers) > 0:
            self.ls_curve.setData(ls_periods, ls_powers)
            best_ls_idx = int(ls_powers.argmax())
            best_ls_p = float(ls_periods[best_ls_idx])
            if best_ls_p > 0:
                self.ls_peak_line.setPos(math.log10(best_ls_p))
                self.ls_peak_line.setVisible(True)
            self.p_ls.autoRange()
        else:
            self.ls_curve.clear()
            self.ls_peak_line.setVisible(False)

    def set_peak_period(self, period: float) -> None:
        """Update detection marker and harmonic lines."""
        if period <= 0:
            self.peak_line.setVisible(False)
            self.subharmonic_line.setVisible(False)
            self.harmonic_line.setVisible(False)
            return

        self.peak_line.setPos(math.log10(period))
        self.peak_line.setVisible(True)

        self.subharmonic_line.setPos(math.log10(period / 2.0))
        self.subharmonic_line.setVisible(True)

        self.harmonic_line.setPos(math.log10(period * 2.0))
        self.harmonic_line.setVisible(True)

    def clear(self) -> None:
        """Remove plots and markers when a session has no result bundle."""
        self.power_curve.clear()
        self.ls_curve.clear()
        self._hide_marker_lines()
