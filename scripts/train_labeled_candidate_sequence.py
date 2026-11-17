"""Train the isolated binary CNN on frozen event-matched candidate sequences."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from transiteye.evaluation.labeled_candidate_sequence_training import (
    train_labeled_candidate_sequence,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(".transiteye-workspace/models/labeled_candidate_sequence_v1/cohort_manifest.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".transiteye-workspace/models/labeled_candidate_sequence_v1"),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--maximum-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    report = train_labeled_candidate_sequence(
        args.manifest,
        args.output_dir,
        workers=args.workers,
        maximum_epochs=args.maximum_epochs,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(json.dumps({key: value for key, value in report.items() if key != "training_history"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
