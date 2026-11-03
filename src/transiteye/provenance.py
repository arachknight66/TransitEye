"""Fingerprints for scientific code and locally selected model artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path


def code_fingerprint() -> str:
    """Hash the installed backend source, including uncommitted changes."""
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def model_fingerprint(root: Path) -> str:
    """Hash the model metadata, estimator and optional calibrator."""
    digest = hashlib.sha256()
    for filename in ("model_metadata.json", "estimator.joblib", "calibrator.joblib"):
        path = root / filename
        digest.update(filename.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes() if path.is_file() else b"absent")
    return digest.hexdigest()
