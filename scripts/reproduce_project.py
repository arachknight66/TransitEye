"""Verify or reproduce TransitEye outputs from frozen local artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from transiteye.reproducibility.workflow import (
    demo_smoke_mode,
    full_replay_mode,
    report_mode,
    verify_mode,
)
from transiteye.serialization import canonical_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--verify", action="store_true", help="Verify frozen artifacts only")
    modes.add_argument("--report", action="store_true", help="Regenerate tables and figures")
    modes.add_argument(
        "--demo-smoke", action="store_true", help="Run a tiny offline software smoke"
    )
    modes.add_argument(
        "--full-replay",
        action="store_true",
        help="Run expensive local demo/features/model/evaluation replay",
    )
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root")
    args = parser.parse_args()
    if args.verify:
        print(canonical_json(verify_mode(args.root)))
    elif args.report:
        print(report_mode(args.root))
    elif args.demo_smoke:
        print(canonical_json(demo_smoke_mode(args.root)))
    else:
        print("Running expensive local replay; network acquisition is excluded.")
        full_replay_mode(args.root)


if __name__ == "__main__":
    main()
