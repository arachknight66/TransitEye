"""Run and replay the frozen B046--B051 development-demo evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from transiteye.evaluation import run_robustness_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root")
    args = parser.parse_args()
    result = run_robustness_evaluation(args.root)
    print(result.evaluation_version)
    print(result.directory)


if __name__ == "__main__":
    main()
