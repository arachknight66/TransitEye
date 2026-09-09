"""Deterministic serialization helpers used by foundation records."""

from __future__ import annotations

import hashlib
import json
from typing import Any

DEFAULT_HASH_LENGTH = 20


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible data with a stable, content-based representation."""
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def content_hash(value: Any, *, length: int = DEFAULT_HASH_LENGTH) -> str:
    """Return a truncated SHA-256 digest of canonical JSON content."""
    if not 16 <= length <= 64:
        msg = "Hash length must be between 16 and 64 hexadecimal characters."
        raise ValueError(msg)
    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return digest[:length]
