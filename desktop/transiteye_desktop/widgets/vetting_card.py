"""Spatial source localization, parameter intervals, and blend vetting inspection widget."""

from __future__ import annotations

from typing import Any

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from transiteye_desktop.bundle_loader import LoadedBundle, LoadedCandidate


class VettingCard(QFrame):
    """Visualizes source localization, centroid shifts, CROWDSAP, 68%/95% intervals, and TPF difference images."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet("QFrame { background-color: #242428; border-radius: 8px; padding: 8px; }")

        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        title = QLabel("Source Vetting & Parameter Uncertainties")
        title.setStyleSheet("font-weight: bold; font-size: 14px; color: #E0E0E0;")
        layout.addWidget(title)

        # Assessment Badge
        badge_layout = QHBoxLayout()
        self.lbl_badge = QLabel("NOT ASSESSED")
        self.lbl_badge.setStyleSheet(
            "background-color: #404040; color: white; font-weight: bold; "
            "font-size: 12px; padding: 4px 8px; border-radius: 4px;"
        )
        badge_layout.addWidget(self.lbl_badge)
        badge_layout.addStretch()
        layout.addLayout(badge_layout)

        # Metrics grid
        grid = QGridLayout()
        grid.setSpacing(4)

        grid.addWidget(QLabel("Centroid Shift:"), 0, 0)
        self.lbl_shift = QLabel("-- σ")
        self.lbl_shift.setStyleSheet("font-weight: bold; color: #A0C0FF;")
        grid.addWidget(self.lbl_shift, 0, 1)

        grid.addWidget(QLabel("CROWDSAP:"), 1, 0)
        self.lbl_crowdsap = QLabel("--")
        self.lbl_crowdsap.setStyleSheet("font-weight: bold; color: #A0C0FF;")
        grid.addWidget(self.lbl_crowdsap, 1, 1)

        grid.addWidget(QLabel("SAP/PDCSAP Ratio:"), 2, 0)
        self.lbl_ratio = QLabel("--")
        self.lbl_ratio.setStyleSheet("font-weight: bold; color: #A0C0FF;")
        grid.addWidget(self.lbl_ratio, 2, 1)

        grid.addWidget(QLabel("Measured / Dil. Depth:"), 3, 0)
        self.lbl_depths = QLabel("-- / -- ppm")
        self.lbl_depths.setStyleSheet("font-weight: bold; color: #A0C0FF;")
        grid.addWidget(self.lbl_depths, 3, 1)

        grid.addWidget(QLabel("Period 68% / 95% CI:"), 4, 0)
        self.lbl_period_ci = QLabel("--")
        self.lbl_period_ci.setStyleSheet("font-size: 11px; color: #C0D8FF;")
        grid.addWidget(self.lbl_period_ci, 4, 1)

        grid.addWidget(QLabel("Depth 68% / 95% CI:"), 5, 0)
        self.lbl_depth_ci = QLabel("--")
        self.lbl_depth_ci.setStyleSheet("font-size: 11px; color: #C0D8FF;")
        grid.addWidget(self.lbl_depth_ci, 5, 1)

        grid.addWidget(QLabel("Fit exposure:"), 6, 0)
        self.lbl_exposure = QLabel("not recorded")
        self.lbl_exposure.setStyleSheet("font-size: 11px; color: #FFE080;")
        grid.addWidget(self.lbl_exposure, 6, 1)

        layout.addLayout(grid)

        # TPF Stamp & Difference Image Plots
        self.img_layout = QHBoxLayout()

        self.tpf_glw = pg.GraphicsLayoutWidget()
        self.tpf_glw.setFixedHeight(140)
        self.p_tpf = self.tpf_glw.addPlot(title="TPF Aperture")
        self.p_tpf.setAspectLocked(True)
        self.img_tpf = pg.ImageItem()
        self.p_tpf.addItem(self.img_tpf)

        self.p_diff = self.tpf_glw.addPlot(title="Diff Image (OOT - In)")
        self.p_diff.setAspectLocked(True)
        self.img_diff = pg.ImageItem()
        self.p_diff.addItem(self.img_diff)

        # Target crosshairs and centroid markers
        self.tpf_target_cross = pg.ScatterPlotItem(
            size=14, pen=pg.mkPen((0, 255, 100), width=2), symbol="+"
        )
        self.diff_target_cross = pg.ScatterPlotItem(
            size=14, pen=pg.mkPen((0, 255, 100), width=2), symbol="+"
        )
        self.diff_centroid_marker = pg.ScatterPlotItem(
            size=12, pen=pg.mkPen((255, 60, 60), width=2), brush=pg.mkBrush(255, 60, 60, 100), symbol="o"
        )
        self.p_tpf.addItem(self.tpf_target_cross)
        self.p_diff.addItem(self.diff_target_cross)
        self.p_diff.addItem(self.diff_centroid_marker)

        self.img_layout.addWidget(self.tpf_glw)
        layout.addLayout(self.img_layout)
        self.tpf_glw.setVisible(False)

    @staticmethod
    def _format_ci(param_intervals: dict[str, Any], fmt: str = ".4f") -> str:
        if not isinstance(param_intervals, dict):
            return "--"
        ci68 = param_intervals.get("68")
        ci95 = param_intervals.get("95")
        parts: list[str] = []
        if isinstance(ci68, dict) and ci68.get("lower") is not None and ci68.get("upper") is not None:
            parts.append(f"68% [{float(ci68['lower']):{fmt}}, {float(ci68['upper']):{fmt}}]")
        elif isinstance(ci68, dict) and ci68.get("missing_reason"):
            parts.append(f"68% ({ci68['missing_reason']})")
        if isinstance(ci95, dict) and ci95.get("lower") is not None and ci95.get("upper") is not None:
            parts.append(f"95% [{float(ci95['lower']):{fmt}}, {float(ci95['upper']):{fmt}}]")
        return " | ".join(parts) if parts else "--"

    def set_data(self, bundle: LoadedBundle, candidate: LoadedCandidate | None) -> None:
        """Update vetting card from candidate and bundle TPF arrays."""
        if candidate is None:
            self.clear()
            return

        settings = bundle.metadata.get("analysis_settings") or {}
        exposure_days = settings.get("trapezoid_exposure_days")
        status = settings.get("exposure_status")
        if exposure_days is None:
            self.lbl_exposure.setText("not recorded")
        elif status == "user_supplied":
            self.lbl_exposure.setText(f"{float(exposure_days) * 86400:.1f} s (supplied)")
        else:
            self.lbl_exposure.setText(f"{float(exposure_days) * 86400:.1f} s (unverified assumption)")

        assessment = candidate.source_assessment
        self.lbl_badge.setText(assessment.replace("_", " ").upper())

        if assessment == "target_consistent":
            self.lbl_badge.setStyleSheet(
                "background-color: #2E7D32; color: white; font-weight: bold; "
                "font-size: 12px; padding: 4px 8px; border-radius: 4px;"
            )
        elif assessment == "blend_suspected":
            self.lbl_badge.setStyleSheet(
                "background-color: #C62828; color: white; font-weight: bold; "
                "font-size: 12px; padding: 4px 8px; border-radius: 4px;"
            )
        elif assessment == "unresolved":
            self.lbl_badge.setStyleSheet(
                "background-color: #F57F17; color: white; font-weight: bold; "
                "font-size: 12px; padding: 4px 8px; border-radius: 4px;"
            )
        else:
            self.lbl_badge.setStyleSheet(
                "background-color: #555555; color: white; font-weight: bold; "
                "font-size: 12px; padding: 4px 8px; border-radius: 4px;"
            )

        # Extract features
        shift_sig = candidate.raw_data.get("centroid_shift_sigma", None)
        shift_arc = candidate.raw_data.get("centroid_shift_arcsec", None)
        if shift_sig is not None and not np.isnan(float(shift_sig)):
            if shift_arc is not None and not np.isnan(float(shift_arc)):
                self.lbl_shift.setText(f"{float(shift_sig):.1f} σ ({float(shift_arc):.1f}\")")
            else:
                self.lbl_shift.setText(f"{float(shift_sig):.2f} σ")
        else:
            self.lbl_shift.setText("--")

        crowdsap = candidate.raw_data.get("crowdsap_value", None)
        if crowdsap is not None and not np.isnan(float(crowdsap)):
            self.lbl_crowdsap.setText(f"{float(crowdsap):.3f}")
        else:
            self.lbl_crowdsap.setText("--")

        ratio = candidate.raw_data.get("sap_to_pdcsap_depth_ratio", None)
        if ratio is not None and not np.isnan(float(ratio)):
            self.lbl_ratio.setText(f"{float(ratio):.2f}")
        else:
            self.lbl_ratio.setText("--")

        dil_str = (
            f"{candidate.dilution_corrected_depth_ppm:.1f}"
            if candidate.dilution_corrected_depth_ppm is not None
            else "--"
        )
        self.lbl_depths.setText(f"{candidate.depth_ppm:.1f} / {dil_str} ppm")

        p_int = candidate.parameter_intervals.get("period_days", {})
        d_int = candidate.parameter_intervals.get("depth_ppm", {})
        self.lbl_period_ci.setText(self._format_ci(p_int, ".4f"))
        self.lbl_depth_ci.setText(self._format_ci(d_int, ".1f"))

        # Display TPF stamps if present in candidate or bundle
        stamp = candidate.tpf_stamp if candidate.tpf_stamp is not None else bundle.tpf_stamp
        diff = candidate.tpf_diff if candidate.tpf_diff is not None else bundle.tpf_diff

        if stamp is not None and stamp.ndim == 2:
            self.tpf_glw.setVisible(True)
            self.img_tpf.setImage(stamp)
            rows, cols = stamp.shape
            t_col, t_row = cols / 2.0, rows / 2.0
            self.tpf_target_cross.setData([t_col], [t_row])
            self.diff_target_cross.setData([t_col], [t_row])

            if diff is not None and diff.ndim == 2:
                self.img_diff.setImage(diff)
                diff_offset_px = candidate.raw_data.get("difference_image_offset_pixels")
                if (
                    diff_offset_px is not None
                    and not np.isnan(float(diff_offset_px))
                    and float(diff_offset_px) > 0
                ):
                    pos_d = np.maximum(0.0, diff)
                    tot_d = float(np.sum(pos_d))
                    if tot_d > 1e-6:
                        y_grid, x_grid = np.mgrid[0:rows, 0:cols]
                        c_x = float(np.sum(x_grid * pos_d) / tot_d)
                        c_y = float(np.sum(y_grid * pos_d) / tot_d)
                        self.diff_centroid_marker.setData([c_x], [c_y])
                    else:
                        self.diff_centroid_marker.clear()
                else:
                    self.diff_centroid_marker.clear()
            else:
                self.img_diff.clear()
                self.diff_centroid_marker.clear()
        else:
            self.tpf_glw.setVisible(False)
            self.tpf_target_cross.clear()
            self.diff_target_cross.clear()
            self.diff_centroid_marker.clear()

    def clear(self) -> None:
        """Reset vetting card to blank state."""
        self.lbl_badge.setText("NOT ASSESSED")
        self.lbl_badge.setStyleSheet("background-color: #404040; color: white; padding: 4px 8px;")
        self.lbl_shift.setText("-- σ")
        self.lbl_crowdsap.setText("--")
        self.lbl_ratio.setText("--")
        self.lbl_depths.setText("-- / -- ppm")
        self.lbl_period_ci.setText("--")
        self.lbl_depth_ci.setText("--")
        self.lbl_exposure.setText("not recorded")
        self.tpf_glw.setVisible(False)
        self.img_tpf.clear()
        self.img_diff.clear()
        self.tpf_target_cross.clear()
        self.diff_target_cross.clear()
        self.diff_centroid_marker.clear()
