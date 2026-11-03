"""Frozen, synthetic-white-noise injection pilot (not a real-data evaluation)."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import tomllib
from importlib.metadata import version
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray
from scipy.integrate import simpson

from transiteye.domain.models import AnalysisSettings, LightCurve, Provenance, as_jsonable
from transiteye.execution.service import analyze_light_curve
from transiteye.provenance import code_fingerprint
from transiteye.reporting.bundles import verify_bundle_integrity

logger = logging.getLogger(__name__)


def injection_flux(
    times: NDArray[np.float64], period: float, epoch: float, duration_hours: float,
    depth_ppm: float, ingress_fraction: float, exposure_days: float, points: int,
) -> NDArray[np.float64]:
    """Integrate a separately implemented ideal trapezoid over each exposure."""
    if points < 3 or points % 2 == 0 or not 0 < ingress_fraction <= 1:
        raise ValueError("invalid injection integration grid or ingress")
    offsets = np.linspace(-exposure_days / 2, exposure_days / 2, points)
    phase = np.abs(((times[:, None] + offsets - epoch + period / 2) % period) - period / 2)
    half = duration_hours / 48.0
    ingress = half * ingress_fraction
    shape = np.clip((half - phase) / ingress, 0.0, 1.0)
    return np.asarray(simpson(shape, x=offsets, axis=1) / exposure_days * depth_ppm / 1e6)


def matches_truth(
    period: float, epoch: float, truth_period: float, truth_epoch: float,
    period_fraction: float, epoch_hours: float,
) -> bool:
    """Match the injected period and transit phase; harmonics do not count."""
    phase_error = abs(((epoch - truth_epoch + truth_period / 2) % truth_period) - truth_period / 2)
    return abs(period / truth_period - 1.0) <= period_fraction and phase_error * 24 <= epoch_hours


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _trial(config: dict[str, Any], trial_id: str, period: float | None, depth: float | None,
           settings: AnalysisSettings, output_dir: Path) -> dict[str, Any]:
    token = f"{config['seed']}:{trial_id}".encode()
    seed = int.from_bytes(hashlib.sha256(token).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    cadence_days = float(config["cadence_minutes"]) / 1440
    exposure_days = float(config["exposure_minutes"]) / 1440
    times = np.arange(0, float(config["baseline_days"]), cadence_days, dtype=np.float64)
    flux = 1.0 + rng.normal(0.0, float(config["noise_ppm"]) / 1e6, len(times))
    epoch = None if period is None else float(rng.uniform(0, period))
    if period is not None and epoch is not None and depth is not None:
        flux -= injection_flux(times, period, epoch, float(config["duration_hours"]), depth,
                               float(config["ingress_fraction"]), exposure_days,
                               int(config["injection_integration_points"]))
    digest = hashlib.sha256(times.tobytes() + flux.tobytes()).hexdigest()
    provenance = Provenance(
        source_uri=f"synthetic://injection-pilot/{trial_id}", source_path=None,
        checksum_sha256=digest, product_type="synthetic_light_curve", target_id=trial_id,
        sector=None, time_system="relative", time_reference=None,
    )
    curve = LightCurve(
        time_days=times, flux=flux, flux_error=np.full_like(times, float(config["noise_ppm"]) / 1e6),
        quality=np.zeros(len(times), dtype=np.int64), cadence_number=np.arange(len(times)),
        centroid_column=None, centroid_row=None, flux_kind="synthetic", flux_unit="relative",
        provenance=provenance,
    )
    row: dict[str, Any] = {
        "trial_id": trial_id, "kind": "null" if period is None else "injection", "seed": seed,
        "input_sha256": digest, "truth_period_days": period, "truth_epoch_days": epoch,
        "truth_depth_ppm": depth, "truth_duration_hours": None if period is None else config["duration_hours"],
        "status": "error", "error": None, "detected": False, "fit_failed": False,
        "physical_fit_failed": False, "fit_error": None, "matched": False, "harmonic": False,
    }
    for name in ("period_days", "depth_ppm", "duration_hours"):
        row[f"error_{name}"] = None
        for level in ("68", "95"):
            row[f"covered_{name}_{level}"] = None
    try:
        result, _ = analyze_light_curve(curve, analysis_settings=settings, output_dir=output_dir)
        row["status"] = result.status.value
        row["error"] = result.error_message
        if result.candidates:
            candidate = result.candidates[0]
            row["detected"] = True
            row["fit_failed"] = "fit_failed" in candidate.flags
            row["physical_fit_failed"] = "physical_fit_failed" in candidate.flags
            row["fit_error"] = result.candidate_errors.get(candidate.candidate_id)
            if period is not None and epoch is not None and candidate.period_days and candidate.epoch_days is not None:
                row["harmonic"] = any(
                    abs(candidate.period_days / (period * factor) - 1) <= config["match_period_fraction"]
                    for factor in (0.5, 2.0)
                )
                row["matched"] = not row["fit_failed"] and matches_truth(
                    candidate.period_days, candidate.epoch_days, period, epoch,
                    config["match_period_fraction"], config["match_epoch_hours"],
                )
                if row["matched"]:
                    truth = {"period_days": period, "depth_ppm": depth,
                             "duration_hours": float(config["duration_hours"])}
                    estimated = {"period_days": candidate.period_days, "depth_ppm": candidate.depth_ppm,
                                 "duration_hours": candidate.duration_hours}
                    intervals = result.parameter_intervals.get(candidate.candidate_id, {})
                    for name, value in truth.items():
                        if value is not None and estimated[name] is not None:
                            row[f"error_{name}"] = float(estimated[name] - value)
                            for level in ("68", "95"):
                                interval = intervals.get(name, {}).get(level)
                                if interval is not None and interval.lower is not None and interval.upper is not None:
                                    row[f"covered_{name}_{level}"] = bool(interval.lower <= value <= interval.upper)
    except Exception as exc:
        logger.exception("Synthetic pilot trial %s failed", trial_id)
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Count failures, alerts, recovery, and available conditional coverage explicitly."""
    injected = [row for row in rows if row["kind"] == "injection"]
    null = [row for row in rows if row["kind"] == "null"]
    result: dict[str, Any] = {
        "injected_trials": len(injected), "recovered": sum(row["matched"] for row in injected),
        "harmonic_candidates": sum(row["harmonic"] for row in injected),
        "null_trials": len(null), "null_alerts": sum(row["detected"] for row in null),
        "trial_errors": sum(row["status"] in {"error", "failed", "insufficient_data"} for row in rows),
        "fit_failures": sum(row["fit_failed"] for row in rows), "parameters": {},
        "physical_fit_failures": sum(row["physical_fit_failed"] for row in rows),
        "null_usable": sum(row["status"] not in {"error", "failed", "insufficient_data"} for row in null),
    }
    strata = []
    for period, depth in sorted({(row["truth_period_days"], row["truth_depth_ppm"]) for row in injected}):
        group = [row for row in injected if row["truth_period_days"] == period and row["truth_depth_ppm"] == depth]
        strata.append({"period_days": period, "depth_ppm": depth, "trials": len(group),
                       "recovered": sum(row["matched"] for row in group),
                       "detected": sum(row["detected"] for row in group),
                       "harmonic_candidates": sum(row["harmonic"] for row in group),
                       "errors": sum(row["status"] in {"error", "failed", "insufficient_data"}
                                     for row in group)})
    result["strata"] = strata
    for name in ("period_days", "depth_ppm", "duration_hours"):
        errors = [abs(row[f"error_{name}"]) for row in injected if row[f"error_{name}"] is not None]
        data: dict[str, Any] = {"median_absolute_error": float(np.median(errors)) if errors else None,
                               "error_count": len(errors)}
        for level in ("68", "95"):
            covered = [row[f"covered_{name}_{level}"] for row in injected
                       if row[f"covered_{name}_{level}"] is not None]
            data[level] = {"covered": sum(covered), "available": len(covered),
                           "matched": result["recovered"], "all_injections": len(injected)}
        result["parameters"][name] = data
    return result


def run(config_path: Path, output_dir: Path) -> dict[str, Any]:
    """Run or resume deterministic trials; retain every result and the input identities."""
    raw = config_path.read_bytes()
    config = tomllib.loads(raw.decode("utf-8"))
    if config["schema_version"] != "1":
        raise ValueError("unsupported pilot configuration")
    settings = AnalysisSettings(**config["analysis"], trapezoid_exposure_days=config["exposure_minutes"] / 1440)
    identity = {"config_sha256": hashlib.sha256(raw).hexdigest(), "backend_sha256": code_fingerprint(),
                "thread_environment": {key: os.environ.get(key) for key in
                                       ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.json"
    metadata = {**identity, "schema_version": "1", "config": config, "settings": as_jsonable(settings),
                "transiteye_version": version("transiteye"), "numpy_version": np.__version__,
                "scipy_version": version("scipy"), "batman_version": version("batman-package"),
                "python_version": platform.python_version(),
                "limitations": "Synthetic white noise and ideal trapezoids only; not real-data performance."}
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text())
        if any(existing.get(key) != value for key, value in identity.items()):
            raise ValueError("existing pilot directory has different configuration or backend code")
    else:
        _write_json(metadata_path, metadata)
    trials = [(f"inj_p{pi}_d{di}_r{rep}", float(p), float(d))
              for (pi, p), (di, d), rep in product(enumerate(config["periods_days"]),
                                                    enumerate(config["depths_ppm"]),
                                                    range(config["replicates"]))]
    trials += [(f"null_r{rep}", None, None) for rep in range(config["null_trials"])]
    rows: list[dict[str, Any]] = []
    for index, (trial_id, period, depth) in enumerate(trials, 1):
        trial_dir = output_dir / "trials" / trial_id
        trial_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = trial_dir / "trial.json"
        bundle = trial_dir / "bundle"
        row = None
        if checkpoint.exists():
            previous = json.loads(checkpoint.read_text())
            if previous["status"] == "error" or verify_bundle_integrity(bundle):
                row = previous
        if row is None:
            row = _trial(config, trial_id, period, depth, settings, bundle)
            _write_json(checkpoint, row)
        rows.append(row)
        logger.info("Synthetic pilot trial %d/%d: %s (%s)", index, len(trials), trial_id, row["status"])
    table = pa.Table.from_pylist([{key: value for key, value in row.items() if key != "seed"} for row in rows])
    table = table.append_column("seed", pa.array([row["seed"] for row in rows], type=pa.uint64()))
    pq.write_table(table, output_dir / "trials.parquet")
    summary = {**identity, **summarize(rows)}
    _write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(run(args.config, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
