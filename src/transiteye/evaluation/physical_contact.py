"""Development-only independent radial check of transit-model contact behavior."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tomllib
import warnings
from importlib.metadata import version
from itertools import product
from pathlib import Path
from typing import Any

import batman
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.integrate import IntegrationWarning, quad

from transiteye.provenance import code_fingerprint


def radial_flux(z: float, radius: float, u1: float, u2: float) -> float:
    """Integrate blocked stellar annuli; z and radius are in stellar-radius units."""
    if not (np.isfinite(z) and np.isfinite(radius) and z >= 0 and 0 < radius < 1):
        raise ValueError("invalid geometry")
    normalization = 1.0 - u1 / 3.0 - u2 / 6.0
    if normalization <= 0:
        raise ValueError("invalid limb darkening normalization")

    def integrand(r: float) -> float:
        if z == 0.0:
            fraction = float(r <= radius)
        elif r == 0.0:
            fraction = float(z <= radius)
        else:
            x = (r * r + z * z - radius * radius) / (2.0 * r * z)
            fraction = np.arccos(np.clip(x, -1.0, 1.0)) / np.pi
        mu = np.sqrt(max(0.0, 1.0 - r * r))
        intensity = 1.0 - u1 * (1.0 - mu) - u2 * (1.0 - mu) ** 2
        return float(2.0 * r * intensity * fraction / normalization)

    cuts = sorted({0.0, 1.0, *(x for x in (abs(z - radius), z + radius) if 0 < x < 1)})
    with warnings.catch_warnings():
        warnings.simplefilter("error", IntegrationWarning)
        blocked = sum(quad(integrand, left, right, epsabs=1e-12, epsrel=1e-11, limit=300)[0]
                      for left, right in zip(cuts, cuts[1:], strict=False))
    return float(1.0 - blocked)


def evaluate_case(config: dict[str, Any], radius: float, offset: float) -> dict[str, Any]:
    """Compare at the actual projected separation computed by Batman."""
    requested_z = 1.0 - radius + offset
    a = float(config["semi_major_axis_stellar_radii"])
    params = batman.TransitParams()
    params.t0 = 0.0
    params.per = float(config["period_days"])
    params.rp = radius
    params.a = a
    params.inc = float(np.degrees(np.arccos(requested_z / a)))
    params.ecc = 0.0
    params.w = 90.0
    params.u = config["limb_darkening"]
    params.limb_dark = "quadratic"
    model = batman.TransitModel(params, np.array([0.0]))
    z = float(model.ds[0])
    flux = float(model.light_curve(params)[0])
    independent = radial_flux(z, radius, *params.u)
    return {"radius_ratio": radius, "offset": offset, "requested_z": requested_z,
            "actual_z": z, "batman_flux": flux, "radial_flux": independent,
            "absolute_error_ppm": abs(flux - independent) * 1e6}


def run(config_path: Path, output_dir: Path) -> dict[str, Any]:
    """Isolate each native call so diagnostics are recorded alongside every case."""
    raw = config_path.read_bytes()
    config = tomllib.loads(raw.decode())
    if config["schema_version"] != "1":
        raise ValueError("unsupported diagnostic configuration")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("output directory must be empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    worker = ("import json,sys; from transiteye.evaluation.physical_contact import evaluate_case; "
              "c=json.loads(sys.argv[1]); print('ROW='+json.dumps(evaluate_case(c,float(sys.argv[2]),"
              "float(sys.argv[3]))),flush=True)")
    for radius, offset in product(config["radius_ratios"], config["contact_offsets"]):
        process = subprocess.run([sys.executable, "-c", worker, json.dumps(config), str(radius),
                                  str(offset)], capture_output=True, text=True, check=False)
        lines = process.stdout.splitlines()
        payload = [line[4:] for line in lines if line.startswith("ROW=")]
        diagnostics = "\n".join([line for line in lines if not line.startswith("ROW=")]
                                + ([process.stderr.strip()] if process.stderr.strip() else [])) or None
        if process.returncode == 0 and len(payload) == 1:
            row = json.loads(payload[0])
            row.update(error=None, native_diagnostics=diagnostics)
        else:
            row = {"radius_ratio": radius, "offset": offset, "requested_z": None, "actual_z": None,
                   "batman_flux": None, "radial_flux": None, "absolute_error_ppm": None,
                   "error": f"worker exit {process.returncode}", "native_diagnostics": diagnostics}
        row["usable"] = bool(row["error"] is None and diagnostics is None and
                             np.isfinite(row["absolute_error_ppm"]))
        row["within_tolerance"] = bool(row["usable"] and
                                       row["absolute_error_ppm"] <= config["absolute_tolerance_ppm"])
        rows.append(row)
    pq.write_table(pa.Table.from_pylist(rows), output_dir / "cases.parquet")
    summary = {"config_sha256": hashlib.sha256(raw).hexdigest(), "backend_sha256": code_fingerprint(),
               "config": config, "python_version": platform.python_version(),
               "numpy_version": np.__version__, "scipy_version": version("scipy"),
               "batman_version": version("batman-package"),
               "thread_environment": {key: os.environ.get(key) for key in
                                      ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
               "cases": len(rows), "unusable": sum(not r["usable"] for r in rows),
               "within_tolerance": sum(r["within_tolerance"] for r in rows),
               "max_usable_error_ppm": max((r["absolute_error_ppm"] for r in rows if r["usable"]),
                                           default=None),
               "scope": "instantaneous circular quadratic model at contact; not exposure or fit validation"}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    summary = run(args.config, args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["within_tolerance"] != summary["cases"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
