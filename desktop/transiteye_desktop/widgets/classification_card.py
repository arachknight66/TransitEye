"""Morphology classification summary and calibrated probability card."""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from transiteye_desktop.bundle_loader import LoadedCandidate


class ClassificationCard(QFrame):
    """Visualizes candidate morphology class, abstention status, and calibrated probabilities."""

    CLASSES = ["transit_like", "eclipse_like", "stellar_variability", "other_artifact"]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet("QFrame { background-color: #242428; border-radius: 8px; padding: 8px; }")

        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        title = QLabel("Morphology Classification")
        title.setStyleSheet("font-weight: bold; font-size: 14px; color: #E0E0E0;")
        layout.addWidget(title)

        # Badge & Confidence line
        badge_layout = QHBoxLayout()
        self.lbl_badge = QLabel("NONE")
        self.lbl_badge.setStyleSheet(
            "background-color: #404040; color: white; font-weight: bold; "
            "font-size: 13px; padding: 4px 10px; border-radius: 4px;"
        )
        self.lbl_conf = QLabel("--%")
        self.lbl_conf.setStyleSheet("font-size: 14px; font-weight: bold; color: #A0C0FF;")

        badge_layout.addWidget(self.lbl_badge)
        badge_layout.addSpacing(10)
        badge_layout.addWidget(self.lbl_conf)
        badge_layout.addStretch()
        layout.addLayout(badge_layout)

        # Abstention warning banner
        self.lbl_abstain = QLabel("⚠ ABSTAINED: Ambiguous morphology / low confidence margin")
        self.lbl_abstain.setStyleSheet(
            "background-color: #5C4010; color: #FFE080; padding: 4px 8px; border-radius: 4px; font-size: 11px;"
        )
        self.lbl_abstain.setVisible(False)
        layout.addWidget(self.lbl_abstain)

        # Probability bars
        self.bars: dict[str, tuple[QLabel, QProgressBar]] = {}
        for c in self.CLASSES:
            row = QHBoxLayout()
            name_lbl = QLabel(c.replace("_", "-"))
            name_lbl.setFixedWidth(85)
            name_lbl.setStyleSheet("font-size: 11px; color: #C0C0C0;")

            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setTextVisible(True)
            bar.setFixedHeight(14)
            bar.setStyleSheet(
                "QProgressBar { border: 1px solid #444; border-radius: 3px; background: #181818; text-align: right; } "
                "QProgressBar::chunk { background-color: #4A90E2; border-radius: 2px; }"
            )

            row.addWidget(name_lbl)
            row.addWidget(bar)
            layout.addLayout(row)
            self.bars[c] = (name_lbl, bar)

    def set_candidate(self, candidate: LoadedCandidate | None) -> None:
        """Update card for selected candidate."""
        if candidate is None:
            self.clear()
            return

        cls_name = candidate.predicted_class
        conf_pct = candidate.calibrated_confidence * 100.0

        # Update Badge
        self.lbl_badge.setText(cls_name.replace("_", "-").upper())
        self.lbl_conf.setText(f"{conf_pct:.1f}%")

        if cls_name == "transit_like" and not candidate.abstained:
            self.lbl_badge.setStyleSheet(
                "background-color: #2E7D32; color: white; font-weight: bold; "
                "font-size: 13px; padding: 4px 10px; border-radius: 4px;"
            )
        elif cls_name == "eclipse_like":
            self.lbl_badge.setStyleSheet(
                "background-color: #D84315; color: white; font-weight: bold; "
                "font-size: 13px; padding: 4px 10px; border-radius: 4px;"
            )
        elif cls_name == "stellar_variability":
            self.lbl_badge.setStyleSheet(
                "background-color: #1565C0; color: white; font-weight: bold; "
                "font-size: 13px; padding: 4px 10px; border-radius: 4px;"
            )
        else:
            self.lbl_badge.setStyleSheet(
                "background-color: #555555; color: white; font-weight: bold; "
                "font-size: 13px; padding: 4px 10px; border-radius: 4px;"
            )

        # Abstention banner
        self.lbl_abstain.setVisible(candidate.abstained)

        # Probabilities
        probs = candidate.probabilities or candidate.raw_data.get("probabilities", {}) or {}
        for c, (_, bar) in self.bars.items():
            p = float(probs.get(c, 0.0))
            bar.setValue(int(round(p * 100.0)))

    def clear(self) -> None:
        """Reset card to empty state."""
        self.lbl_badge.setText("NONE")
        self.lbl_badge.setStyleSheet("background-color: #404040; color: white; padding: 4px 10px;")
        self.lbl_conf.setText("--%")
        self.lbl_abstain.setVisible(False)
        for _, bar in self.bars.values():
            bar.setValue(0)
