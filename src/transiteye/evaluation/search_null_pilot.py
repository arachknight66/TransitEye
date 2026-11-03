"""Experimental IID-only primary-BLS permutation test and synthetic null calibration.

This does not validate the production search, which also detrends, screens and may
search iteratively. It must not be used as a TESS significance without a validated
noise model and the complete search procedure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import platform
import tomllib
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray

from transiteye.detection.bls import run_bls
from transiteye.evaluation.injection_pilot import _write_json
from transiteye.provenance import code_fingerprint

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PermutationResult:
    """Monte Carlo p-value for a predeclared primary-BLS maximum statistic."""

    observed_max_power: float
    exceedances: int
    permutations: int
    p_value: float
    seed: int


def iid_primary_bls_permutation(
    time_days: NDArray[np.float64], flux: NDArray[np.float64], search: dict[str, Any],
    permutations: int, seed: int, *, assume_iid: bool = False,
) -> PermutationResult:
    """Repeat the full primary BLS grid for each permutation under an IID null.

    Callers must explicitly acknowledge exchangeability. This helper does not
    establish that assumption and is not wired into production result bundles.
    """
    if not assume_iid:
        raise ValueError("IID/exchangeability must be explicitly assumed for this experimental test")
    t = np.asarray(time_days, dtype=np.float64)
    y = np.asarray(flux, dtype=np.float64)
    if (t.ndim != 1 or y.ndim != 1 or len(t) != len(y) or len(t) < 50
            or not np.all(np.isfinite(t)) or not np.all(np.isfinite(y))
            or not np.all(np.diff(t) > 0) or permutations < 1):
        raise ValueError("finite, ordered equal-length arrays and positive permutations are required")

    def statistic(values: NDArray[np.float64]) -> float:
        result = run_bls(t, values, flux_err=None, **search)
        value = float(np.max(result.powers))
        if not np.isfinite(value):
            raise ValueError("non-finite maximum BLS power")
        return value

    observed = statistic(y)
    tolerance = 32 * np.finfo(np.float64).eps * max(1.0, abs(observed))
    rng = np.random.default_rng(seed)
    exceedances = sum(statistic(rng.permutation(y)) >= observed - tolerance for _ in range(permutations))
    return PermutationResult(observed, int(exceedances), permutations,
                             float((exceedances + 1) / (permutations + 1)), seed)


def _seed(base: int, kind: str, index: int, role: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{base}:{kind}:{index}:{role}".encode()).digest()[:8], "big")


def _trial(config: dict[str, Any], kind: str, index: int) -> dict[str, Any]:
    noise_seed, permutation_seed = (_seed(config["seed"], kind, index, role) for role in ("noise", "permutation"))
    rng = np.random.default_rng(noise_seed)
    times = np.arange(config["points"], dtype=np.float64) * config["cadence_minutes"] / 1440.0
    innovations = rng.normal(0.0, config["noise_ppm"] / 1e6, len(times))
    if kind == "correlated":
        rho = float(config["ar1_rho"])
        if not -1 < rho < 1:
            raise ValueError("AR(1) correlation must be strictly between -1 and 1")
        noise = np.empty_like(innovations)
        noise[0] = innovations[0]
        for index2 in range(1, len(noise)):
            noise[index2] = rho * noise[index2 - 1] + math.sqrt(1 - rho * rho) * innovations[index2]
    else:
        noise = innovations
    flux = 1.0 + noise
    row: dict[str, Any] = {"kind": kind, "trial_index": index, "noise_seed": noise_seed,
        "permutation_seed": permutation_seed, "input_sha256": hashlib.sha256(times.tobytes() + flux.tobytes()).hexdigest(),
        "status": "error", "error": None, "observed_max_power": None,
        "exceedances": None, "p_value": None}
    try:
        result = iid_primary_bls_permutation(times, flux, config["search"], config["permutations"],
                                             permutation_seed, assume_iid=True)
        row.update(status="complete", observed_max_power=result.observed_max_power,
                   exceedances=result.exceedances, p_value=result.p_value)
    except Exception as exc:
        logger.exception("Synthetic null trial %s-%d failed", kind, index)
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def summarize(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    """Report trial failures and a predeclared white-noise calibration diagnostic."""
    output: dict[str, Any] = {}
    for kind in ("white", "correlated"):
        group = [row for row in rows if row["kind"] == kind]
        values = np.sort([row["p_value"] for row in group if row["p_value"] is not None])
        output[kind] = {"trials": len(group), "valid": len(values),
                        "errors": sum(row["status"] != "complete" for row in group),
                        "alerts_at_alpha": int(np.count_nonzero(values <= config["alpha"]))}
        if kind == "white":
            gap = float(np.max(np.arange(1, len(values) + 1) / len(values) - values)) if len(values) else None
            bound = math.sqrt(math.log(2 / (1 - config["dkw_confidence"])) / (2 * len(values))) if len(values) else None
            output[kind]["max_empirical_excess"] = gap
            output[kind]["dkw_bound"] = bound
            output[kind]["pilot_passed"] = bool(
                len(group) == config["white_trials"] and len(values) == len(group)
                and output[kind]["alerts_at_alpha"] <= config["max_white_alerts"]
                and gap is not None and bound is not None and gap <= bound)
    return output


def run(config_path: Path, output_dir: Path) -> dict[str, Any]:
    """Run or resume the frozen synthetic calibration, preserving every trial."""
    raw = config_path.read_bytes()
    config = tomllib.loads(raw.decode("utf-8"))
    if config["schema_version"] != "1" or config["search"]["device"] != "cpu":
        raise ValueError("unsupported null-pilot configuration")
    identity = {"config_sha256": hashlib.sha256(raw).hexdigest(), "backend_sha256": code_fingerprint(),
                "thread_environment": {key: os.environ.get(key) for key in
                                       ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text())
        if any(previous.get(key) != value for key, value in identity.items()):
            raise ValueError("existing null-pilot directory has different code, config or thread settings")
    else:
        _write_json(metadata_path, {**identity, "schema_version": "1", "config": config,
                    "python_version": platform.python_version(), "transiteye_version": version("transiteye"),
                    "numpy_version": np.__version__, "astropy_version": version("astropy"),
                    "limitations": "Primary BLS only; IID null; not production or correlated-noise significance."})
    rows = []
    for kind, count in (("white", config["white_trials"]), ("correlated", config["correlated_trials"])):
        for index in range(count):
            checkpoint = output_dir / "trials" / f"{kind}-{index:04d}.json"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            if checkpoint.exists():
                row = json.loads(checkpoint.read_text())
            else:
                row = _trial(config, kind, index)
                _write_json(checkpoint, row)
            rows.append(row)
            if (index + 1) % 10 == 0 or index + 1 == count:
                logger.info("Null pilot %s %d/%d", kind, index + 1, count)
    table = pa.Table.from_pylist([{key: value for key, value in row.items()
                                  if key not in {"noise_seed", "permutation_seed"}} for row in rows])
    for name in ("noise_seed", "permutation_seed"):
        table = table.append_column(name, pa.array([row[name] for row in rows], type=pa.uint64()))
    pq.write_table(table, output_dir / "trials.parquet")
    summary = {**identity, "alpha": config["alpha"], "permutations": config["permutations"],
               **summarize(rows, config)}
    _write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    summary = run(args.config, args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["white"]["pilot_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
