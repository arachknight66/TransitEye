"""Candidate rankings and attributes table widget."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from transiteye_desktop.bundle_loader import LoadedCandidate


class CandidateTableWidget(QWidget):
    """Table widget presenting ranked transit candidates with classification and vetting."""

    candidate_selected = pyqtSignal(int)  # candidate index

    HEADERS = [
        "Rank",
        "Candidate ID",
        "Period (d)",
        "Epoch (BJD)",
        "Depth (ppm)",
        "Dil. Depth (ppm)",
        "Duration (h)",
        "Events",
        "SNR",
        "Significance",
        "Morphology Class",
        "Calibrated Conf",
        "Source Vetting",
    ]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.table = QTableWidget(0, len(self.HEADERS))
        self.table.setHorizontalHeaderLabels(self.HEADERS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)

        layout.addWidget(self.table)
        self._candidates: list[LoadedCandidate] = []

    def set_candidates(self, candidates: list[LoadedCandidate]) -> None:
        """Populate table with detected candidates."""
        self._candidates = candidates
        self.table.setRowCount(len(candidates))

        for row, c in enumerate(candidates):
            dil_str = (
                f"{c.dilution_corrected_depth_ppm:.1f}"
                if c.dilution_corrected_depth_ppm is not None
                else "--"
            )
            items = [
                QTableWidgetItem(f"#{row + 1}"),
                QTableWidgetItem(
                    f"{c.candidate_id} (fit failed)" if "fit_failed" in c.flags else c.candidate_id
                ),
                QTableWidgetItem(f"{c.period_days:.4f}"),
                QTableWidgetItem(f"{c.epoch_days:.4f}"),
                QTableWidgetItem(f"{c.depth_ppm:.1f}"),
                QTableWidgetItem(dil_str),
                QTableWidgetItem(f"{c.duration_hours:.2f}"),
                QTableWidgetItem(str(c.observed_event_count)),
                QTableWidgetItem(f"{c.detection_snr:.1f}"),
                QTableWidgetItem(
                    f"{c.search_significance:.3f}" if c.search_significance is not None else "--"
                ),
                QTableWidgetItem(
                    "unavailable" if not c.probabilities else
                    c.predicted_class if not c.abstained else f"{c.predicted_class} (abstain)"
                ),
                QTableWidgetItem(
                    f"{c.calibrated_confidence * 100.0:.1f}%"
                    if c.calibrated_confidence is not None else "--"
                ),
                QTableWidgetItem(c.source_assessment),
            ]

            # Color-code morphology class
            class_item = items[10]
            if c.predicted_class == "transit_like" and not c.abstained:
                class_item.setBackground(QColor(220, 245, 220))
                class_item.setForeground(QColor(20, 60, 20))
            elif c.predicted_class == "eclipse_like":
                class_item.setBackground(QColor(255, 235, 205))
                class_item.setForeground(QColor(80, 40, 10))
            elif c.abstained:
                class_item.setBackground(QColor(255, 245, 200))
                class_item.setForeground(QColor(80, 60, 10))

            # Color-code vetting
            vetting_item = items[12]
            if c.source_assessment == "target_consistent":
                vetting_item.setBackground(QColor(220, 245, 220))
                vetting_item.setForeground(QColor(20, 60, 20))
            elif c.source_assessment == "blend_suspected":
                vetting_item.setBackground(QColor(255, 220, 220))
                vetting_item.setForeground(QColor(80, 20, 20))

            for col, item in enumerate(items):
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(row, col, item)

        if len(candidates) > 0:
            self.table.selectRow(0)
            self.table.horizontalScrollBar().setValue(0)

    def _on_selection_changed(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if rows:
            self.candidate_selected.emit(rows[0].row())

    def clear(self) -> None:
        """Clear candidate rows."""
        self._candidates.clear()
        self.table.setRowCount(0)
