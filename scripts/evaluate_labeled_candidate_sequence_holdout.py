"""Open the locked candidate-sequence holdout once for final inference-only evaluation."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from transiteye.evaluation.labeled_candidate_sequence_holdout import evaluate_locked_holdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(".transiteye-workspace/models/labeled_candidate_sequence_v1/cohort_manifest.json"),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(".transiteye-workspace/models/labeled_candidate_sequence_v1"),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    report = evaluate_locked_holdout(
        args.manifest,
        args.model_dir,
        workers=args.workers,
        device=args.device,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
