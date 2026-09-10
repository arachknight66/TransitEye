"""Freeze and replay the current B031--B034 development dataset."""

from __future__ import annotations

from pathlib import Path

from transiteye.datasets.development import build_current_development_dataset

if __name__ == "__main__":
    dataset_path, split_path, dataset_version = build_current_development_dataset(Path.cwd())
    print(dataset_version)
    print(dataset_path)
    print(split_path)
