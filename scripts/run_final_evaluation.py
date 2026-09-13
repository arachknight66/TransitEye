"""Run and replay the frozen B052--B057 evaluation package."""

from __future__ import annotations

import argparse
from pathlib import Path

from transiteye.evaluation.final_builder import run_final_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root")
    args = parser.parse_args()
    result = run_final_evaluation(args.root)
    print(result.final_evaluation_version)
    print(result.directory)


if __name__ == "__main__":
    main()
