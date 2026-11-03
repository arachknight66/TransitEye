"""Desktop application entrypoint for TransitEye."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QApplication

from transiteye_desktop.main_window import TransitEyeMainWindow


def apply_dark_theme(app: QApplication) -> None:
    """Apply modern dark scientific styling to Qt application."""
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(30, 30, 34))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(230, 230, 230))
    palette.setColor(QPalette.ColorRole.Base, QColor(22, 22, 26))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(36, 36, 42))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Text, QColor(230, 230, 230))
    palette.setColor(QPalette.ColorRole.Button, QColor(42, 42, 48))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(230, 230, 230))
    palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
    palette.setColor(QPalette.ColorRole.Link, QColor(66, 133, 244))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(66, 133, 244))
    palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.white)
    app.setPalette(palette)


def main() -> int:
    """Launch the TransitEye Desktop Application."""
    parser = argparse.ArgumentParser(description="TransitEye Desktop Application")
    parser.add_argument("--file", type=Path, default=None, help="Initial TESS light-curve FITS file to load")
    parser.add_argument("--bundle", type=Path, default=None, help="Initial result bundle directory to display")
    parser.add_argument("--tpf", type=Path, default=None, help="Companion Target Pixel File (TPF)")
    parser.add_argument("--demo", action="store_true", help="Analyze a local sample light curve")
    parser.add_argument("--demo-file", type=Path, help="Use this FITS file as the sample")
    args = parser.parse_args()
    if args.demo_file and not args.demo:
        parser.error("--demo-file requires --demo")
    if args.demo and (args.file or args.bundle or args.tpf):
        parser.error("--demo cannot be combined with --file, --bundle, or --tpf")

    app = QApplication(sys.argv)
    app.setApplicationName("TransitEye")
    apply_dark_theme(app)

    window = TransitEyeMainWindow()
    if args.file and args.file.exists():
        window._current_fits = args.file
        window.lbl_status.setText(f"Loaded file: {args.file.name}")
        if args.tpf and args.tpf.exists():
            window._current_tpf = args.tpf
    elif args.bundle and args.bundle.is_dir():
        window.display_bundle_from_path(args.bundle)

    window.show()
    if args.demo:
        QTimer.singleShot(0, lambda: window.start_sample(args.demo_file))
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
