"""PyQtGraph items resilient to a view disappearing during Qt teardown."""

from __future__ import annotations

import pyqtgraph as pg
from PyQt6.QtCore import QRectF


class SafeInfiniteLine(pg.InfiniteLine):
    """Avoid dereferencing a detached ViewBox while a scene is being destroyed.

    PyQtGraph 0.14 can obtain a view rectangle and then lose its ViewBox before
    it asks for that view's size. Qt may invoke this callback during teardown.
    Other exceptions still propagate when the view remains attached.
    """

    def _computeBoundingRect(self) -> QRectF:
        try:
            return super()._computeBoundingRect()
        except (AttributeError, RuntimeError):
            try:
                detached = self.getViewBox() is None
            except RuntimeError:
                detached = True
            if not detached:
                raise
            return QRectF()
