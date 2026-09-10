"""Execute the frozen B011 real-cohort expansion through B031--B034."""

from __future__ import annotations

from pathlib import Path

from transiteye.expansion import run_frozen_cohort_expansion

if __name__ == "__main__":
    result = run_frozen_cohort_expansion(Path.cwd())
    print(result.manifest_id)
    print(result.dataset_version)
    print(result.dataset_path)
