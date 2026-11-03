"""Paired development-only physical fits with different exposure integrations."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import tomllib
from importlib.metadata import version
from itertools import product
from pathlib import Path
from typing import Any

import batman
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.optimize import least_squares

from transiteye.characterization.fitting import fit_batman_refinement
from transiteye.evaluation.exposure_convergence import reference_flux, simpson_flux
from transiteye.provenance import code_fingerprint


def params_from_vector(p: np.ndarray, coefficients: list[float]) -> batman.TransitParams:
    params = batman.TransitParams()
    params.t0, params.per, params.rp, params.a, params.inc = map(float, p)
    params.ecc = 0.0
    params.w = 90.0
    params.u = coefficients
    params.limb_dark = "quadratic"
    return params


def simpson_fit(time_days: np.ndarray, flux: np.ndarray, flux_err: np.ndarray,
                epoch: float, period: float, duration: float, depth: float,
                coefficients: list[float], exposure: float, points: int,
                max_nfev: int) -> dict[str, float] | None:
    """Experimental analog of the production optimizer with Simpson model evaluation."""
    initial = [epoch, period, np.sqrt(max(1e-6, depth)),
               max(2.0, period / (np.pi * max(1e-3, duration))), 88.0]
    lower = [epoch - 0.1 * period, 0.9 * period, 0.001, 1.5, 60.0]
    upper = [epoch + 0.1 * period, 1.1 * period, 0.5, 200.0, 90.0]

    def residual(vector: np.ndarray) -> np.ndarray:
        params = params_from_vector(vector, coefficients)
        return (flux - simpson_flux(params, time_days, exposure, points)) / flux_err

    result = least_squares(residual, initial, bounds=(lower, upper), loss="soft_l1",
                           max_nfev=max_nfev)
    if not result.success:
        return None
    return dict(zip(("epoch_days", "period_days", "rp_rstar", "a_rstar", "inc_deg"),
                    map(float, result.x), strict=True))


def evaluate_case(config: dict[str, Any], radius: float, impact: float,
                  exposure_minutes: float, case_index: int) -> dict[str, Any]:
    period, a = float(config["period_days"]), float(config["semi_major_axis_stellar_radii"])
    exposure = exposure_minutes / 1440.0
    inc = float(np.degrees(np.arccos(impact / a)))
    duration = period / np.pi * np.arcsin(np.sqrt(((1 + radius)**2 - impact**2) / (a*a - impact*impact)))
    truth = np.array([0.0, period, radius, a, inc])
    params = params_from_vector(truth, config["limb_darkening"])
    window = float(config["transit_window_days"])
    # Both transits use the actual cadence; no overlapping nominal integrations.
    phase = np.arange(-window, window + exposure / 2, exposure)
    times = np.concatenate((phase, phase + period))
    truth_flux = reference_flux(params, times, exposure, int(config["truth_points"]))
    check = reference_flux(params, times, exposure, int(config["truth_check_points"]))
    truth_difference = float(np.max(np.abs(truth_flux - check)) * 1e6)
    sigma = float(config["noise_ppm"]) / 1e6
    seed = int(config["seed"]) + case_index
    flux = truth_flux + np.random.default_rng(seed).normal(0, sigma, size=len(times))
    errors = np.full(len(times), sigma)
    base = {"case_index": case_index, "radius_ratio": radius, "impact_parameter": impact,
            "exposure_minutes": exposure_minutes, "noise_ppm": float(config["noise_ppm"]),
            "replicate": int(config.get("replicate", 0)), "cadences": len(times), "seed": seed,
            "truth_reference_difference_ppm": truth_difference}
    rows: list[dict[str, Any]] = []
    for method in ("production_five", "simpson_51"):
        start = time.perf_counter()
        error = None
        fit = None
        try:
            if method == "production_five":
                result = fit_batman_refinement(times, flux, errors, 0.0, period, float(duration),
                                               radius**2, tuple(config["limb_darkening"]), exposure)
                fit = None if result is None else result[0]
            else:
                fit = simpson_fit(times, flux, errors, 0.0, period, float(duration), radius**2,
                                  config["limb_darkening"], exposure, int(config["simpson_points"]),
                                  int(config["max_nfev"]))
            if fit is None:
                error = "optimizer failed or did not converge"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - start
        row = {**base, "method": method, "elapsed_seconds": elapsed, "error": error}
        for key, value in zip(("epoch_days", "period_days", "rp_rstar", "a_rstar", "inc_deg"),
                              truth, strict=True):
            row[key] = None if fit is None else fit[key]
            row[f"error_{key}"] = None if fit is None else fit[key] - float(value)
        rows.append(row)
    return {"rows": rows}


def run(config_path: Path, output_dir: Path) -> dict[str, Any]:
    raw = config_path.read_bytes()
    config = tomllib.loads(raw.decode())
    if config["schema_version"] != "1" or config["max_nfev"] != 200:
        raise ValueError("unsupported configuration or production optimizer budget")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("output directory must be empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    worker = ("import json,sys; from transiteye.evaluation.physical_fit_pilot import evaluate_case; "
              "r=evaluate_case(json.loads(sys.argv[1]),float(sys.argv[2]),float(sys.argv[3]),"
              "float(sys.argv[4]),int(sys.argv[5])); print('ROWS='+json.dumps(r),flush=True)")
    rows: list[dict[str, Any]] = []
    noise_levels = config.get("noise_levels_ppm", [config.get("noise_ppm", 100.0)])
    replicates = int(config.get("replicates", 1))
    if replicates < 1 or not noise_levels or any(float(noise) <= 0 for noise in noise_levels):
        raise ValueError("replicates and noise levels must be positive")
    cases = product(config["radius_ratios"], config["impact_parameters"],
                    config["exposure_minutes"], noise_levels, range(replicates))
    for index, (radius, impact, exposure, noise, replicate) in enumerate(cases):
        case_config = {**config, "noise_ppm": noise, "replicate": replicate}
        try:
            process = subprocess.run([sys.executable, "-c", worker, json.dumps(case_config), str(radius),
                                      str(impact), str(exposure), str(index)], capture_output=True,
                                     text=True, check=False, timeout=120)
            lines = process.stdout.splitlines()
            payload = [line[5:] for line in lines if line.startswith("ROWS=")]
            diagnostics = "\n".join([line for line in lines if not line.startswith("ROWS=")]
                                    + ([process.stderr.strip()] if process.stderr.strip() else [])) or None
            failure = f"worker exit {process.returncode} or invalid output"
            case_rows = None
            if process.returncode == 0 and len(payload) == 1:
                try:
                    candidate_rows = json.loads(payload[0])["rows"]
                    if (len(candidate_rows) == 2 and
                        {r["method"] for r in candidate_rows} == {"production_five", "simpson_51"} and
                        all(r["case_index"] == index and r["noise_ppm"] == noise and
                            r["replicate"] == replicate for r in candidate_rows)):
                        case_rows = candidate_rows
                except (KeyError, TypeError, ValueError):
                    pass
        except subprocess.TimeoutExpired:
            diagnostics = "worker exceeded 120 second timeout"
            failure = "worker timeout"
            case_rows = None
        if case_rows is None:
            case_rows = [{"case_index": index, "radius_ratio": radius, "impact_parameter": impact,
                          "exposure_minutes": exposure, "noise_ppm": noise, "replicate": replicate,
                          "method": method,
                          "cadences": None, "seed": None, "truth_reference_difference_ppm": None,
                          "elapsed_seconds": None,
                          **{key: None for name in ("epoch_days", "period_days", "rp_rstar",
                                                   "a_rstar", "inc_deg")
                             for key in (name, f"error_{name}")},
                          "error": failure}
                         for method in ("production_five", "simpson_51")]
        for row in case_rows:
            row["native_diagnostics"] = diagnostics
            row["usable"] = bool(row["error"] is None and diagnostics is None and
                                 row.get("truth_reference_difference_ppm", float("inf")) <=
                                 config["truth_tolerance_ppm"])
        rows.extend(case_rows)
    pq.write_table(pa.Table.from_pylist(rows), output_dir / "fits.parquet")
    summary = {"schema_version": "1", "config": config, "config_sha256": hashlib.sha256(raw).hexdigest(),
               "backend_sha256": code_fingerprint(), "python_version": platform.python_version(),
               "numpy_version": np.__version__, "scipy_version": version("scipy"),
               "batman_version": version("batman-package"),
               "thread_environment": {key: os.environ.get(key) for key in
                                      ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
               "rows": len(rows), "usable": sum(r["usable"] for r in rows),
               "scope": "small synthetic paired optimizer pilot; not a coverage or detection study"}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    summary = run(args.config, args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["usable"] != summary["rows"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
