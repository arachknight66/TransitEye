"""Freeze a cleaned, source-grouped catalog-event cohort without opening FITS data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transiteye.evaluation.labeled_sequence_cohort import prepare_cohort


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".transiteye-workspace/models/labeled_candidate_sequence_v1"),
    )
    args = parser.parse_args()
    root = Path.cwd()
    manifest = prepare_cohort(root, args.output_dir)
    print(json.dumps({
        "manifest_sha256": manifest["manifest_sha256"],
        "retained_labeled_source_count": manifest["retained_labeled_source_count"],
        "retained_labeled_product_count": manifest["retained_labeled_product_count"],
        "retained_source_class_counts": manifest["retained_source_class_counts"],
        "partition_counts": manifest["partition_counts"],
        "excluded_prior_partition_group_count": manifest["excluded_prior_partition_group_count"],
        "excluded_product_conflict_count": manifest["excluded_product_conflict_count"],
        "excluded_source_conflict_count": manifest["excluded_source_conflict_count"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
