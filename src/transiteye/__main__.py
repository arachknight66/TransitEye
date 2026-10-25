"""Command-line entry point for the Qt-free TransitEye backend."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from transiteye import __version__
from transiteye.config import ensure_workspace, load_config
from transiteye.datasets.labels import read_labels
from transiteye.datasets.splits import read_split_manifest, validate_labels_against_split
from transiteye.domain.models import as_jsonable
from transiteye.execution.runner import BatchProgress, BatchRunner
from transiteye.execution.service import analyze_light_curve
from transiteye.execution.store import JobStatus, JobStore
from transiteye.io.fits import read_tess_light_curve
from transiteye.io.manifest import read_manifest, write_manifest
from transiteye.io.mast import AstroqueryMastClient, acquire_manifest, build_manifest
from transiteye.logging import configure_logging
from transiteye.reporting.figures import generate_vetting_sheet


def _analyze_fits(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    light_curve = read_tess_light_curve(args.path, flux_preference=args.flux)

    def _progress(data: dict[str, object]) -> None:
        if args.progress_jsonl:
            print(json.dumps(data, sort_keys=True))

    res, bundle_path = analyze_light_curve(
        light_curve,
        config=config,
        max_candidates=args.max_candidates,
        snr_threshold=args.snr_threshold,
        device=args.device,
        tpf_path=args.tpf,
        output_dir=args.output_dir,
        progress_callback=_progress,
    )

    summary = {
        "analysis_id": res.analysis_id,
        "status": res.status.value,
        "candidate_count": len(res.candidates),
        "bundle_path": str(bundle_path),
        "candidates": [
            {
                "candidate_id": c.candidate_id,
                "period_days": c.period_days,
                "epoch_days": c.epoch_days,
                "duration_hours": c.duration_hours,
                "depth_ppm": c.depth_ppm,
                "detection_snr": c.detection_snr,
                "search_significance": c.search_significance,
                "classification": {
                    "predicted_class": res.classifications[c.candidate_id].predicted_class.value
                    if res.classifications.get(c.candidate_id)
                    and res.classifications[c.candidate_id].predicted_class
                    else None,
                    "abstained": res.classifications[c.candidate_id].abstained
                    if res.classifications.get(c.candidate_id)
                    else False,
                    "source_assessment": res.classifications[c.candidate_id].source_assessment.value
                    if res.classifications.get(c.candidate_id)
                    else "not_assessed",
                    "probabilities": {
                        k.value: v
                        for k, v in res.classifications[c.candidate_id].probabilities.items()
                    }
                    if res.classifications.get(c.candidate_id)
                    else {},
                }
                if c.candidate_id in res.classifications
                else None,
                "flags": list(c.flags),
            }
            for c in res.candidates
        ],
    }
    if args.progress_jsonl:
        print(json.dumps(summary, sort_keys=True))
    else:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _run_batch(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    ensure_workspace(config)

    targets: list[tuple[str, str]] = []
    source_path = args.source.expanduser().resolve()

    if source_path.is_file() and source_path.suffix == ".json":
        manifest = read_manifest(source_path)
        for entry in manifest.entries:
            if entry.local_filename:
                candidate_local = Path(".transiteye-workspace/data/pilot") / entry.local_filename
                if candidate_local.is_file():
                    uri = f"file://{candidate_local.resolve()}"
                else:
                    uri = entry.data_uri
            else:
                uri = entry.data_uri
            targets.append((entry.product_id, uri))
    elif source_path.is_dir():
        for f in sorted(source_path.rglob("*_lc.fits")):
            target_id = f.name.split("-")[2] if "-" in f.name else f.stem
            targets.append((target_id, f"file://{f}"))
    elif source_path.is_file() and source_path.name.endswith(".fits"):
        target_id = source_path.stem
        targets.append((target_id, f"file://{source_path}"))
    else:
        raise ValueError(f"Invalid batch source: {source_path}")

    db_path = config.workspace.root / "state.db"
    store = JobStore(db_path)

    job_id = args.job_id or f"batch_{int(time.time())}"
    try:
        store.get_job(job_id)
    except KeyError:
        store.create_job(
            job_id,
            targets,
            {
                "max_candidates": args.max_candidates,
                "snr_threshold": args.snr_threshold,
                "device": args.device,
            },
        )

    runner = BatchRunner(
        store=store,
        config=config,
        max_workers=args.workers,
        device=args.device,
    )

    def _progress(prog: BatchProgress) -> None:
        if args.progress_jsonl:
            print(json.dumps(asdict(prog), sort_keys=True))

    final_job = runner.run_job(job_id, progress_callback=_progress)
    summary = {
        "job_id": final_job.job_id,
        "status": final_job.status.value,
        "total_targets": final_job.total_targets,
        "completed_targets": final_job.completed_targets,
        "failed_targets": final_job.failed_targets,
    }
    if args.progress_jsonl:
        print(json.dumps(summary, sort_keys=True))
    else:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if final_job.status == JobStatus.COMPLETED else 1


def _job_status(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    db_path = config.workspace.root / "state.db"
    store = JobStore(db_path)

    job = store.get_job(args.job_id)
    summary = {
        "job_id": job.job_id,
        "status": job.status.value,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "total_targets": job.total_targets,
        "completed_targets": job.completed_targets,
        "failed_targets": job.failed_targets,
        "error_message": job.error_message,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _run_locked_eval(args: argparse.Namespace) -> int:
    from transiteye.evaluation.runner import run_locked_evaluation

    cfg = args.config or Path("configs/evaluation_locked.v1.toml")
    out = args.output_dir or Path("docs/report/generated")
    res = run_locked_evaluation(cfg, out)
    print(json.dumps(res, indent=2, sort_keys=True))
    return 0


def _export_vetting_sheet(args: argparse.Namespace) -> int:
    out_path = generate_vetting_sheet(
        bundle_path=args.bundle,
        output_file=args.output,
        candidate_index=args.candidate_index,
    )
    print(json.dumps({"output_file": str(out_path)}, indent=2, sort_keys=True))
    return 0


def _train_model(args: argparse.Namespace) -> int:
    from transiteye.classification.train import train_from_pilot_cohort

    bundle, comparison = train_from_pilot_cohort(
        pilot_dir=args.pilot_dir,
        labels_path=args.labels,
        splits_path=args.splits,
        output_dir=args.output_dir,
        include_synthetic_augmentation=not args.no_synthetic,
    )
    summary = {
        "model_id": bundle.metadata.model_id,
        "estimator_name": bundle.metadata.estimator_name,
        "training_sample_count": bundle.metadata.training_sample_count,
        "synthetic_sample_count": bundle.metadata.synthetic_sample_count,
        "cv_macro_f1": bundle.metadata.cv_macro_f1,
        "tau_conf": bundle.metadata.tau_conf,
        "tau_margin": bundle.metadata.tau_margin,
        "cv_comparison": comparison,
        "output_dir": str(args.output_dir),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


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
    analyze_fits = subparsers.add_parser(
        "analyze-fits", help="analyze a local TESS FITS light curve end to end"
    )
    analyze_fits.add_argument("path", type=Path)
    analyze_fits.add_argument("--tpf", type=Path, default=None, help="companion Target Pixel File")
    analyze_fits.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    analyze_fits.add_argument("--flux", default="PDCSAP_FLUX")
    analyze_fits.add_argument("--config", type=Path)
    analyze_fits.add_argument("--output-dir", type=Path)
    analyze_fits.add_argument("--max-candidates", type=int, default=3)
    analyze_fits.add_argument("--snr-threshold", type=float, default=6.0)
    analyze_fits.add_argument("--progress-jsonl", action="store_true")
    analyze_fits.set_defaults(handler=_analyze_fits)
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

    run_batch = subparsers.add_parser(
        "run-batch", help="run persistent batch execution over a manifest or FITS directory"
    )
    run_batch.add_argument("source", type=Path, help="path to manifest.json or directory of FITS files")
    run_batch.add_argument("--job-id", type=str, default=None, help="job ID (resumes if already exists)")
    run_batch.add_argument("--workers", type=int, default=4, help="number of worker processes")
    run_batch.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    run_batch.add_argument("--max-candidates", type=int, default=3)
    run_batch.add_argument("--snr-threshold", type=float, default=6.0)
    run_batch.add_argument("--config", type=Path)
    run_batch.add_argument("--progress-jsonl", action="store_true")
    run_batch.set_defaults(handler=_run_batch)

    job_status = subparsers.add_parser(
        "job-status", help="query persistent execution status of a batch job"
    )
    job_status.add_argument("job_id", type=str, help="job ID to inspect")
    job_status.add_argument("--config", type=Path)
    job_status.set_defaults(handler=_job_status)

    eval_parser = subparsers.add_parser(
        "run-locked-evaluation", help="run sealed single-shot evaluation across locked test set"
    )
    eval_parser.add_argument("--config", type=Path, default=None, help="evaluation config toml")
    eval_parser.add_argument("--output-dir", type=Path, default=None, help="output directory for reports")
    eval_parser.set_defaults(handler=_run_locked_eval)

    export_fig = subparsers.add_parser(
        "export-vetting-sheet",
        help="export publication-grade PNG/PDF/SVG vetting summary sheet from a result bundle",
    )
    export_fig.add_argument("bundle", type=Path, help="path to result bundle directory")
    export_fig.add_argument("output", type=Path, help="destination .png, .pdf, or .svg file")
    export_fig.add_argument("--candidate-index", type=int, default=0, help="0-based candidate index")
    export_fig.set_defaults(handler=_export_vetting_sheet)

    train_parser = subparsers.add_parser(
        "train-model",
        help="train and calibrate the morphology classifier on pilot cohort FITS files",
    )
    train_parser.add_argument(
        "--pilot-dir", type=Path, default=Path(".transiteye-workspace/data/pilot")
    )
    train_parser.add_argument(
        "--labels", type=Path, default=Path("manifests/pilot_labels.v1.json")
    )
    train_parser.add_argument(
        "--splits", type=Path, default=Path("manifests/pilot_splits.v1.json")
    )
    train_parser.add_argument(
        "--output-dir", type=Path, default=Path(".transiteye-workspace/models/morphology_v1")
    )
    train_parser.add_argument("--no-synthetic", action="store_true")
    train_parser.set_defaults(handler=_train_model)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run a backend command without importing desktop packages."""

    parser = build_parser()
    arguments = parser.parse_args(argv)
    configure_logging(arguments.log_level)
    return arguments.handler(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
