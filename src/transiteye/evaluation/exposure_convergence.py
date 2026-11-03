"""Development-only comparison of finite-exposure transit-model quadrature.

The reference integrates instantaneous batman flux with trapezoidal weights; it is
not an independent transit physics model or a measurement-accuracy study.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from importlib.metadata import version
from itertools import product
from pathlib import Path
from typing import Any

import batman
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray
from scipy.integrate import simpson

from transiteye.provenance import code_fingerprint


@dataclass(frozen=True)
class ExposureCase:
    radius_ratio: float
    impact_parameter: float
    exposure_minutes: float


def _parameters(config: dict[str, Any], case: ExposureCase) -> batman.TransitParams:
    a = float(config["semi_major_axis_stellar_radii"])
    if not (0 < case.radius_ratio < 1 and 0 <= case.impact_parameter < a):
        raise ValueError("invalid radius ratio or impact parameter")
    params = batman.TransitParams()
    params.t0 = 0.0
    params.per = float(config["period_days"])
    params.rp = case.radius_ratio
    params.a = a
    params.inc = float(np.degrees(np.arccos(case.impact_parameter / a)))
    params.ecc = 0.0
    params.w = 90.0
    params.u = list(config["limb_darkening"])
    params.limb_dark = "quadratic"
    return params


def reference_flux(
    params: batman.TransitParams, times: NDArray[np.float64], exposure_days: float, points: int
) -> NDArray[np.float64]:
    """Boxcar-integrate instantaneous flux using composite trapezoidal quadrature."""
    if points < 3 or exposure_days <= 0:
        raise ValueError("reference needs at least three points and positive exposure")
    offsets = np.linspace(-exposure_days / 2, exposure_days / 2, points)
    dense_times = (times[:, None] + offsets[None, :]).reshape(-1)
    instantaneous = batman.TransitModel(params, dense_times).light_curve(params)
    samples = instantaneous.reshape(len(times), points)
    # Divide the trapezoidal integral by its width: the uniform spacing cancels.
    return (samples[:, 0] / 2 + samples[:, -1] / 2 + samples[:, 1:-1].sum(axis=1)) / (
        points - 1
    )


def simpson_flux(
    params: batman.TransitParams, times: NDArray[np.float64], exposure_days: float, points: int
) -> NDArray[np.float64]:
    """Integrate instantaneous Batman flux with composite Simpson weights."""
    if points < 3 or points % 2 == 0 or exposure_days <= 0:
        raise ValueError("Simpson integration needs an odd sample count >= 3 and positive exposure")
    offsets = np.linspace(-exposure_days / 2, exposure_days / 2, points)
    dense_times = (times[:, None] + offsets[None, :]).reshape(-1)
    samples = batman.TransitModel(params, dense_times).light_curve(params).reshape(len(times), points)
    return np.asarray(simpson(samples, x=offsets, axis=1) / exposure_days, dtype=np.float64)


def study(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return one result or explicit failure for every case and sampling factor."""
    if config["schema_version"] != "1":
        raise ValueError("unsupported exposure study configuration")
    method = config.get("integration_method", "batman")
    if method not in {"batman", "simpson"}:
        raise ValueError("unsupported integration method")
    times = np.linspace(
        float(config["time_start_days"]), float(config["time_end_days"]), int(config["time_points"])
    )
    if len(times) < 3 or times[-1] <= times[0]:
        raise ValueError("invalid time grid")
    rows: list[dict[str, Any]] = []
    cases = product(config["radius_ratios"], config["impact_parameters"], config["exposure_minutes"])
    for radius, impact, exposure in cases:
        case = ExposureCase(float(radius), float(impact), float(exposure))
        base = {"radius_ratio": case.radius_ratio, "impact_parameter": case.impact_parameter,
                "exposure_minutes": case.exposure_minutes}
        try:
            params = _parameters(config, case)
            exposure_days = case.exposure_minutes / 1440.0
            ref = reference_flux(params, times, exposure_days, int(config["reference_points"]))
            check = reference_flux(params, times, exposure_days, int(config["reference_check_points"]))
            reference_difference = float(np.max(np.abs(ref - check)) * 1e6)
            depth_ppm = float((1 - np.min(ref)) * 1e6)
            for factor in config["sample_factors"]:
                row = {**base, "sample_factor": int(factor), "reference_difference_ppm": reference_difference,
                       "depth_ppm": depth_ppm}
                try:
                    if method == "batman":
                        sampled = batman.TransitModel(
                            params, times, supersample_factor=int(factor), exp_time=exposure_days
                        ).light_curve(params)
                    else:
                        sampled = simpson_flux(params, times, exposure_days, int(factor))
                    error = float(np.max(np.abs(sampled - ref)) * 1e6)
                    if not np.isfinite(error):
                        raise ValueError("non-finite model error")
                    rows.append({**row, "max_error_ppm": error, "error": None})
                except Exception as exc:
                    rows.append({**row, "max_error_ppm": None, "error": f"{type(exc).__name__}: {exc}"})
        except Exception as exc:
            for factor in config["sample_factors"]:
                rows.append({**base, "sample_factor": int(factor), "reference_difference_ppm": None,
                             "depth_ppm": None, "max_error_ppm": None,
                             "error": f"{type(exc).__name__}: {exc}"})
    if not rows:
        raise ValueError("study grid has no cases or sampling factors")
    return rows


def run(config_path: Path, output_dir: Path) -> dict[str, Any]:
    """Run cases in isolation to capture diagnostics from the native model library."""
    raw = config_path.read_bytes()
    config = tomllib.loads(raw.decode("utf-8"))
    rows: list[dict[str, Any]] = []
    cases = product(config["radius_ratios"], config["impact_parameters"], config["exposure_minutes"])
    worker = (
        "import json,sys; from transiteye.evaluation.exposure_convergence import study; "
        "print('TRANSITEYE_ROWS='+json.dumps(study(json.loads(sys.argv[1]))),flush=True)"
    )
    for radius, impact, exposure in cases:
        case_config = {**config, "radius_ratios": [radius], "impact_parameters": [impact],
                       "exposure_minutes": [exposure]}
        process = subprocess.run(
            [sys.executable, "-c", worker, json.dumps(case_config)],
            capture_output=True, text=True, check=False,
        )
        lines = process.stdout.splitlines()
        payloads = [line.removeprefix("TRANSITEYE_ROWS=") for line in lines
                    if line.startswith("TRANSITEYE_ROWS=")]
        diagnostics = "\n".join(
            [line for line in lines if not line.startswith("TRANSITEYE_ROWS=")]
            + ([process.stderr.strip()] if process.stderr.strip() else [])
        )
        case_rows = None
        if process.returncode == 0 and len(payloads) == 1:
            try:
                parsed = json.loads(payloads[0])
                factors = [int(row["sample_factor"]) for row in parsed]
                identity_matches = all(
                    row["radius_ratio"] == radius and row["impact_parameter"] == impact
                    and row["exposure_minutes"] == exposure for row in parsed
                )
                fields = {"reference_difference_ppm", "depth_ppm", "max_error_ppm", "error"}
                complete = all(fields <= row.keys() for row in parsed)
                if sorted(factors) == sorted(config["sample_factors"]) and identity_matches and complete:
                    case_rows = parsed
            except (KeyError, TypeError, ValueError):
                pass
        if case_rows is None:
            diagnostic = f"worker exit {process.returncode}; invalid or incomplete output; {diagnostics}".strip()
            case_rows = [{"radius_ratio": radius, "impact_parameter": impact,
                          "exposure_minutes": exposure, "sample_factor": int(factor),
                          "reference_difference_ppm": None, "depth_ppm": None,
                          "max_error_ppm": None, "error": diagnostic}
                         for factor in config["sample_factors"]]
        for row in case_rows:
            row["native_diagnostics"] = diagnostics or None
            row["usable"] = row["error"] is None and not diagnostics
            row["reference_converged"] = bool(
                row["usable"] and row["reference_difference_ppm"] is not None
                and row["reference_difference_ppm"] <= config["reference_tolerance_ppm"]
            )
            row["within_tolerance"] = bool(
                row["reference_converged"] and row["max_error_ppm"] is not None
                and row["max_error_ppm"] <= config["comparison_tolerance_ppm"]
            )
        rows.extend(case_rows)
    if not rows:
        raise ValueError("study grid has no cases or sampling factors")
    output_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), output_dir / "cases.parquet")
    summary = {
        "schema_version": "1", "config_sha256": hashlib.sha256(raw).hexdigest(),
        "config": config, "backend_sha256": code_fingerprint(),
        "transiteye_version": version("transiteye"), "batman_version": version("batman-package"),
        "numpy_version": np.__version__, "python_version": platform.python_version(),
        "integration_method": config.get("integration_method", "batman"),
        "scipy_version": version("scipy"),
        "n_rows": len(rows), "n_failures": sum(row["error"] is not None for row in rows),
        "n_unusable": sum(not row["usable"] for row in rows),
        "n_outside_tolerance": sum(not row["within_tolerance"] for row in rows),
        "reference_method": "instantaneous batman flux, composite trapezoidal boxcar average",
        "limitations": "Numerical forward-model convergence only; not physical accuracy or interval coverage.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    summary = run(args.config, args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["n_outside_tolerance"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
