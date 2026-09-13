"""Small presentation helpers for the Streamlit application."""

from __future__ import annotations

from typing import Any


def humanize(name: str) -> str:
    return name.replace("_", " ").title().replace("Ls ", "LS ").replace("Bls ", "BLS ")


def number(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)
