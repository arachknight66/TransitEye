"""Desktop loader for versioned scientific result bundles.

Reads result.json, arrays.npz, and candidates.parquet from disk into UI-ready models.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


@dataclass(frozen=True, slots=True)
class LoadedCandidate:
    """Desktop representation of a transit candidate."""

    candidate_id: str
    period_days: float
    epoch_days: float
    duration_hours: float
    depth_ppm: float
    dilution_corrected_depth_ppm: float | None
    observed_event_count: int
    phase_coverage: float
    aliases_days: list[float]
    detection_snr: float
    search_significance: float
    predicted_class: str
    calibrated_confidence: float
    probabilities: dict[str, float]
    abstained: bool
    source_assessment: str
    flags: list[str]
    parameter_intervals: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)
    trapezoid_model_curve: np.ndarray | None = None
    batman_model_curve: np.ndarray | None = None
    tpf_stamp: np.ndarray | None = None
    tpf_diff: np.ndarray | None = None
    raw_data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LoadedBundle:
    """Complete in-memory model of an analyzed target for desktop visualization."""

    bundle_path: Path
    target_id: str
    analysis_id: str
    status: str
    times: np.ndarray
    raw_flux: np.ndarray
    detrended_flux: np.ndarray
    trend_flux: np.ndarray
    flux_err: np.ndarray
    quality_mask: np.ndarray
    bls_periods: np.ndarray
    bls_powers: np.ndarray
    ls_periods: np.ndarray
    ls_powers: np.ndarray
    candidates: list[LoadedCandidate]
    tpf_stamp: np.ndarray | None
    tpf_diff: np.ndarray | None
    metadata: dict[str, Any]


def load_result_bundle(bundle_dir: Path | str) -> LoadedBundle:
    """Load and parse a saved scientific result bundle directory."""
    bundle_path = Path(bundle_dir).expanduser().resolve()
    if not bundle_path.is_dir():
        raise FileNotFoundError(f"Result bundle directory not found: {bundle_path}")

    # 1. Load result.json
    json_path = bundle_path / "result.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Missing result.json in bundle: {bundle_path}")

    with open(json_path, encoding="utf-8") as f:
        meta = json.load(f)

    target_id = str(
        meta.get("target_id")
        or (meta.get("provenance") or {}).get("target_id")
        or "unknown"
    )

    # 2. Load arrays.npz
    npz_path = bundle_path / "arrays.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing arrays.npz in bundle: {bundle_path}")

    with np.load(npz_path) as npz:
        times = npz["time_days"] if "time_days" in npz else npz["time"]
        raw_flux = (
            npz["flux_raw"]
            if "flux_raw" in npz
            else (npz["raw_flux"] if "raw_flux" in npz else npz["flux_detrended"])
        )
        detrended_flux = (
            npz["flux_detrended"] if "flux_detrended" in npz else npz["flux"]
        )
        trend_flux = (
            npz["trend_flux"]
            if "trend_flux" in npz
            else (npz["trend"] if "trend" in npz else np.ones_like(times))
        )
        flux_err = (
            npz["flux_error"]
            if "flux_error" in npz
            else (npz["flux_err"] if "flux_err" in npz else np.zeros_like(times))
        )
        if "exclusion_flags" in npz:
            raw_excl = np.asarray(npz["exclusion_flags"])
            if len(raw_excl) == len(times):
                quality_mask = raw_excl == 0
            elif "raw_cadence_mask" in npz and len(npz["raw_cadence_mask"]) == len(raw_excl):
                cad_mask = np.asarray(npz["raw_cadence_mask"], dtype=bool)
                quality_mask = raw_excl[cad_mask] == 0
            else:
                quality_mask = np.ones(len(times), dtype=bool)
        elif "quality_mask" in npz:
            qm = np.asarray(npz["quality_mask"], dtype=bool)
            if len(qm) == len(times):
                quality_mask = qm
            else:
                quality_mask = np.ones(len(times), dtype=bool)
        else:
            quality_mask = np.ones(len(times), dtype=bool)

        bls_periods = npz["bls_periods"] if "bls_periods" in npz else np.array([])
        bls_powers = npz["bls_powers"] if "bls_powers" in npz else np.array([])
        ls_periods = npz["ls_periods"] if "ls_periods" in npz else np.array([])
        ls_powers = npz["ls_powers"] if "ls_powers" in npz else np.array([])
        tpf_stamp = npz["tpf_stamp"] if "tpf_stamp" in npz else None
        tpf_diff = npz["tpf_diff"] if "tpf_diff" in npz else None

        cand_models: dict[int, dict[str, np.ndarray]] = {}
        for key in npz.files:
            if key.startswith("cand_"):
                parts = key.split("_", 2)
                if len(parts) == 3 and parts[1].isdigit():
                    idx = int(parts[1])
                    cand_models.setdefault(idx, {})[parts[2]] = npz[key]

    # 3. Load candidates.parquet if present
    candidates: list[LoadedCandidate] = []
    parquet_path = bundle_path / "candidates.parquet"
    meta_classifications = meta.get("classifications") or {}
    meta_intervals = meta.get("parameter_intervals") or {}

    if parquet_path.exists():
        table = pq.read_table(parquet_path)
        dicts = table.to_pylist()
        for idx, row in enumerate(dicts):
            cand_id = str(row.get("candidate_id", f"cand_{idx:02d}"))
            depth_val = row.get("measured_depth_ppm")
            if depth_val is None:
                depth_val = row.get("depth_ppm", 0.0)
            dil_depth_raw = row.get("dilution_corrected_depth_ppm")
            dil_depth = (
                float(dil_depth_raw)
                if dil_depth_raw is not None and not np.isnan(float(dil_depth_raw))
                else None
            )

            # Reconstruct probabilities from parquet columns or result.json classifications
            cls_meta = meta_classifications.get(cand_id) or {}
            probs_from_meta = cls_meta.get("probabilities") or {}
            probs: dict[str, float] = {}
            for m_cls in (
                "transit_like",
                "eclipse_like",
                "stellar_variability",
                "other_artifact",
            ):
                col_key = f"prob_{m_cls}"
                if col_key in row and row[col_key] is not None:
                    probs[m_cls] = float(row[col_key])
                elif m_cls in probs_from_meta:
                    probs[m_cls] = float(probs_from_meta[m_cls])

            if not probs and isinstance(row.get("probabilities"), dict):
                probs = {str(k): float(v) for k, v in row["probabilities"].items()}

            conf_raw = row.get("calibrated_confidence")
            if conf_raw is not None and not np.isnan(float(conf_raw)):
                cal_conf = float(conf_raw)
            elif probs:
                cal_conf = float(max(probs.values()))
            else:
                cal_conf = 0.0

            raw_flags = row.get("flags")
            if isinstance(raw_flags, str):
                flags_list = [f for f in raw_flags.split(",") if f]
            elif isinstance(raw_flags, list | tuple):
                flags_list = [str(f) for f in raw_flags]
            else:
                flags_list = []

            raw_aliases = row.get("aliases_days")
            if isinstance(raw_aliases, list | tuple):
                aliases_list = [float(a) for a in raw_aliases]
            else:
                aliases_list = []

            sig_raw = row.get("search_significance")
            sig_val = (
                float(sig_raw)
                if sig_raw is not None and not np.isnan(float(sig_raw))
                else 0.0
            )

            c_models = cand_models.get(idx, {})
            cand_intervals = meta_intervals.get(cand_id) or {}

            row_enriched = dict(row)
            row_enriched["probabilities"] = probs
            row_enriched["parameter_intervals"] = cand_intervals

            candidates.append(
                LoadedCandidate(
                    candidate_id=cand_id,
                    period_days=float(row.get("period_days", 0.0)),
                    epoch_days=float(row.get("epoch_days", 0.0)),
                    duration_hours=float(row.get("duration_hours", 0.0)),
                    depth_ppm=float(depth_val),
                    dilution_corrected_depth_ppm=dil_depth,
                    observed_event_count=int(row.get("observed_event_count") or 0),
                    phase_coverage=float(row.get("phase_coverage") or 0.0),
                    aliases_days=aliases_list,
                    detection_snr=float(row.get("detection_snr") or 0.0),
                    search_significance=sig_val,
                    predicted_class=str(row.get("predicted_class", "none")),
                    calibrated_confidence=cal_conf,
                    probabilities=probs,
                    abstained=bool(row.get("abstained", False)),
                    source_assessment=str(row.get("source_assessment", "not_assessed")),
                    flags=flags_list,
                    parameter_intervals=cand_intervals,
                    trapezoid_model_curve=c_models.get("trapezoid_model"),
                    batman_model_curve=c_models.get("batman_model"),
                    tpf_stamp=c_models.get("tpf_stamp", tpf_stamp),
                    tpf_diff=c_models.get("tpf_diff", tpf_diff),
                    raw_data=row_enriched,
                )
            )

    return LoadedBundle(
        bundle_path=bundle_path,
        target_id=target_id,
        analysis_id=str(meta.get("analysis_id", "unknown")),
        status=str(meta.get("status", "unknown")),
        times=times,
        raw_flux=raw_flux,
        detrended_flux=detrended_flux,
        trend_flux=trend_flux,
        flux_err=flux_err,
        quality_mask=quality_mask,
        bls_periods=bls_periods,
        bls_powers=bls_powers,
        ls_periods=ls_periods,
        ls_powers=ls_powers,
        candidates=candidates,
        tpf_stamp=tpf_stamp,
        tpf_diff=tpf_diff,
        metadata=meta,
    )
