"""Command-line entry point for the Qt-free TransitEye backend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transiteye import __version__
from transiteye.config import ensure_workspace, load_config
from transiteye.datasets.labels import read_labels
from transiteye.datasets.splits import read_split_manifest, validate_labels_against_split
from transiteye.domain.models import as_jsonable
from transiteye.io.fits import read_tess_light_curve
from transiteye.io.manifest import read_manifest, write_manifest
from transiteye.io.mast import AstroqueryMastClient, acquire_manifest, build_manifest
from transiteye.logging import configure_logging


def _inspect_fits(args: argparse.Namespace) -> int:
    light_curve = read_tess_light_curve(args.path, flux_preference=args.flux)
    payload = as_jsonable(light_curve)
    payload["cadence_count"] = len(light_curve.time_days)
    payload.pop("time_days")
    payload.pop("flux")
    payload.pop("flux_error")
    payload.pop("quality")
    payload.pop("cadence_number")
    payload.pop("centroid_column")
    payload.pop("centroid_row")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _inspect_manifest(args: argparse.Namespace) -> int:
    manifest = read_manifest(args.path)
    print(
        json.dumps(
            {"manifest_id": manifest.manifest_id, "entry_count": len(manifest.entries)}, indent=2
        )
    )
    return 0


def _workspace(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    ensure_workspace(config)
    print(config.workspace.root)
    return 0


def _discover_manifest(args: argparse.Namespace) -> int:
    manifest = build_manifest(AstroqueryMastClient(), args.target_id, args.sector)
    write_manifest(manifest, args.output)
    print(
        json.dumps(
            {"manifest_id": manifest.manifest_id, "entry_count": len(manifest.entries)}, indent=2
        )
    )
    return 0


def _acquire_manifest(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    ensure_workspace(config)
    manifest = read_manifest(args.manifest)
    completed = acquire_manifest(
        AstroqueryMastClient(),
        manifest,
        config.workspace.cache_directory,
        config.acquisition,
    )
    print(
        json.dumps(
            {"manifest_id": manifest.manifest_id, "completed": completed}, indent=2, default=str
        )
    )
    return 0


def _audit_cohort(args: argparse.Namespace) -> int:
    labels = read_labels(args.labels)
    splits = read_split_manifest(args.splits)
    validate_labels_against_split(labels, splits)
    by_label: dict[str, int] = {}
    by_split: dict[str, int] = {}
    assignments = {
        assignment.source_group_id: assignment.split.value for assignment in splits.assignments
    }
    for label in labels:
        by_label[label.label.value] = by_label.get(label.label.value, 0) + 1
        split = assignments[label.source_group_id]
        by_split[split] = by_split.get(split, 0) + 1
    print(
        json.dumps({"label_counts": by_label, "split_counts": by_split}, indent=2, sort_keys=True)
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the stable Phase 1 command surface."""

    parser = argparse.ArgumentParser(prog="transiteye", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--log-level", default="INFO")
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect_fits = subparsers.add_parser(
        "inspect-fits", help="validate and summarize a TESS FITS light curve"
    )
    inspect_fits.add_argument("path", type=Path)
    inspect_fits.add_argument("--flux", default="PDCSAP_FLUX")
    inspect_fits.set_defaults(handler=_inspect_fits)
    inspect_manifest = subparsers.add_parser(
        "inspect-manifest", help="validate and summarize a product manifest"
    )
    inspect_manifest.add_argument("path", type=Path)
    inspect_manifest.set_defaults(handler=_inspect_manifest)
    workspace = subparsers.add_parser(
        "workspace", help="create and print the configured external workspace"
    )
    workspace.add_argument("--config", type=Path)
    workspace.set_defaults(handler=_workspace)
    discover = subparsers.add_parser("discover-manifest", help="freeze MAST light-curve discovery")
    discover.add_argument("target_id")
    discover.add_argument("--sector", type=int)
    discover.add_argument("--output", type=Path, required=True)
    discover.set_defaults(handler=_discover_manifest)
    acquire = subparsers.add_parser(
        "acquire-manifest", help="download a frozen manifest into the workspace"
    )
    acquire.add_argument("manifest", type=Path)
    acquire.add_argument("--config", type=Path)
    acquire.set_defaults(handler=_acquire_manifest)
    audit = subparsers.add_parser(
        "audit-cohort", help="validate labels against a frozen split manifest"
    )
    audit.add_argument("--labels", type=Path, required=True)
    audit.add_argument("--splits", type=Path, required=True)
    audit.set_defaults(handler=_audit_cohort)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run a backend command without importing desktop packages."""

    parser = build_parser()
    arguments = parser.parse_args(argv)
    configure_logging(arguments.log_level)
    return arguments.handler(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
