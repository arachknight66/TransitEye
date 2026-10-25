"""Architectural boundary enforcement tests.

Guarantees that scientific backend code under `src/transiteye/` remains strictly
headless, with zero Qt/GUI dependencies, and that no scientific logic leaks into
or from desktop modules.
"""

from __future__ import annotations

import ast
from pathlib import Path

FORBIDDEN_GUI_MODULES = {
    "PyQt5",
    "PyQt6",
    "PySide2",
    "PySide6",
    "pyqtgraph",
    "transiteye_desktop",
    "desktop",
}


def _extract_imported_modules(py_file: Path) -> set[str]:
    """Parse a Python source file and return all root imported module names."""
    tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_pkg = alias.name.split(".")[0]
                imports.add(root_pkg)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root_pkg = node.module.split(".")[0]
                imports.add(root_pkg)
    return imports


def test_scientific_backend_has_zero_gui_imports() -> None:
    """Verify src/transiteye contains no PyQt, PySide, pyqtgraph, or desktop imports."""
    root_dir = Path(__file__).resolve().parent.parent / "src" / "transiteye"
    py_files = list(root_dir.rglob("*.py"))
    assert len(py_files) > 0, "No python files found under src/transiteye"

    violations: list[str] = []
    for file_path in py_files:
        imports = _extract_imported_modules(file_path)
        forbidden_present = imports.intersection(FORBIDDEN_GUI_MODULES)
        if forbidden_present:
            rel_path = file_path.relative_to(root_dir.parent.parent)
            violations.append(f"{rel_path}: imports forbidden modules {sorted(forbidden_present)}")

    assert not violations, "\n".join(violations)
