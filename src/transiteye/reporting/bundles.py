"""Structured result bundle writer and loader (result.json, arrays.npz, candidates.parquet)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from transiteye.characterization.fitting import RefinedCandidate
from transiteye.detection.peaks import DetectionResult
from transiteye.domain.models import AnalysisResult, MorphologyClass, as_jsonable
from transiteye.preprocessing.detrend import DualLightCurve
from transiteye.vetting.source import VettingEvidence

_BUNDLE_COMPONENTS = ("result.json", "arrays.npz", "candidates.parquet")


def _file_checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_bundle_integrity(bundle_dir: Path) -> bool:
    """Check a v3 bundle's component hashes; older bundles are not reusable as cache entries."""
    try:
        manifest = json.loads((bundle_dir / "integrity.json").read_text(encoding="utf-8"))
        checksums = manifest["sha256"]
        return all(
            isinstance(checksums[name], str)
            and len(checksums[name]) == 64
            and checksums[name] == _file_checksum(bundle_dir / name)
            for name in _BUNDLE_COMPONENTS
        )
    except (OSError, ValueError, TypeError, KeyError):
        return False


def write_result_bundle(
    analysis_result: AnalysisResult,
    dual_light_curve: DualLightCurve | None,
    detection_result: DetectionResult | None,
    refined_candidates: tuple[RefinedCandidate, ...],
    output_dir: Path,
    vetting_map: Mapping[str, VettingEvidence] | None = None,
    features_map: Mapping[str, Mapping[str, float]] | None = None,
) -> Path:
    """Serialize complete analysis outputs into a self-contained bundle directory."""
    target_dir = output_dir.expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    # 1. Write result.json
    result_json_path = target_dir / "result.json"
    payload = as_jsonable(analysis_result)
    payload["target_id"] = analysis_result.provenance.target_id or "unknown"
    result_json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # 2. Write arrays.npz
    if dual_light_curve is not None:
        arrays_dict: dict[str, Any] = {
            "time_days": dual_light_curve.prepared.time_days,
            "flux_raw": dual_light_curve.prepared.flux_normalized,
            "flux_detrended": dual_light_curve.detrended_flux,
            "trend_flux": dual_light_curve.trend_flux,
            "exclusion_flags": dual_light_curve.prepared.exclusion_flags,
            "bls_periods": detection_result.primary_bls.periods
            if detection_result is not None
            else np.array([], dtype=np.float64),
            "bls_powers": detection_result.primary_bls.powers
            if detection_result is not None
            else np.array([], dtype=np.float64),
            "ls_periods": detection_result.variability.periods
            if detection_result is not None
            else np.array([], dtype=np.float64),
            "ls_powers": detection_result.variability.powers
            if detection_result is not None
            else np.array([], dtype=np.float64),
        }

        if dual_light_curve.prepared.flux_error_normalized is not None:
            arrays_dict["flux_error"] = dual_light_curve.prepared.flux_error_normalized
    else:
        empty_f64 = np.array([], dtype=np.float64)
        arrays_dict = {
            "time_days": empty_f64,
            "flux_raw": empty_f64,
            "flux_detrended": empty_f64,
            "trend_flux": empty_f64,
            "exclusion_flags": np.array([], dtype=np.int32),
            "bls_periods": empty_f64,
            "bls_powers": empty_f64,
            "ls_periods": empty_f64,
            "ls_powers": empty_f64,
        }

    for idx, ref in enumerate(refined_candidates):
        arrays_dict[f"cand_{idx:02d}_trapezoid_model"] = ref.trapezoid_model_curve
        if ref.batman_model_curve is not None:
            arrays_dict[f"cand_{idx:02d}_batman_model"] = ref.batman_model_curve

        if vetting_map and ref.candidate.candidate_id in vetting_map:
            ev = vetting_map[ref.candidate.candidate_id]
            tpf_stamp = ev.details.get("tpf_stamp")
            tpf_diff = ev.details.get("tpf_diff")
            if isinstance(tpf_stamp, np.ndarray):
                arrays_dict[f"cand_{idx:02d}_tpf_stamp"] = tpf_stamp
                if "tpf_stamp" not in arrays_dict:
                    arrays_dict["tpf_stamp"] = tpf_stamp
            if isinstance(tpf_diff, np.ndarray):
                arrays_dict[f"cand_{idx:02d}_tpf_diff"] = tpf_diff
                if "tpf_diff" not in arrays_dict:
                    arrays_dict["tpf_diff"] = tpf_diff

    np.savez_compressed(target_dir / "arrays.npz", **arrays_dict)

    # 3. Write candidates.parquet
    def _cls_prob(cand_id: str, m_cls: MorphologyClass) -> float | None:
        cls_obj = analysis_result.classifications.get(cand_id)
        if cls_obj is None or not cls_obj.probabilities:
            return None
        return float(cls_obj.probabilities.get(m_cls, 0.0))

    def _cls_conf(cand_id: str) -> float | None:
        cls_obj = analysis_result.classifications.get(cand_id)
        if cls_obj is None or not cls_obj.probabilities:
            return None
        return float(max(cls_obj.probabilities.values()))

    def _vet_val(cand_id: str, attr: str, feat_fallback: str | None = None) -> float:
        if vetting_map and cand_id in vetting_map:
            val = getattr(vetting_map[cand_id], attr, None)
            if val is not None:
                return float(val)
            return float("nan")
        if feat_fallback and features_map and cand_id in features_map:
            f_val = features_map[cand_id].get(feat_fallback)
            if f_val is not None:
                return float(f_val)
        return float("nan")

    result_candidates = {candidate.candidate_id: candidate for candidate in analysis_result.candidates}
    columns: dict[str, Any] = {
        "candidate_id": [c.candidate.candidate_id for c in refined_candidates],
        "period_days": [float(c.period_days) for c in refined_candidates],
        "epoch_days": [float(c.epoch_days) for c in refined_candidates],
        "duration_hours": [float(c.duration_hours) for c in refined_candidates],
        "measured_depth_ppm": [float(c.measured_depth_ppm) for c in refined_candidates],
        "dilution_corrected_depth_ppm": [
            float(c.dilution_corrected_depth_ppm)
            if c.dilution_corrected_depth_ppm is not None
            else float("nan")
            for c in refined_candidates
        ],
        "observed_event_count": [
            int(c.candidate.observed_event_count or 0) for c in refined_candidates
        ],
        "detection_snr": [
            result_candidates[c.candidate.candidate_id].detection_snr for c in refined_candidates
        ],
        "search_significance": [
            result_candidates[c.candidate.candidate_id].search_significance
            for c in refined_candidates
        ],
        "ingress_fraction": [float(c.ingress_fraction) for c in refined_candidates],
        "chi2_reduced": [float(c.chi2_reduced) for c in refined_candidates],
        "bic": [float(c.bic) for c in refined_candidates],
        "batman_delta_bic": [
            float(c.batman_delta_bic) if c.batman_delta_bic is not None else float("nan")
            for c in refined_candidates
        ],
        "phase_coverage": [float(c.phase_coverage) for c in refined_candidates],
        "aliases_days": [
            [float(a) for a in c.candidate.aliases_days] for c in refined_candidates
        ],
        "flags": [list(c.candidate.flags) for c in refined_candidates],
        "predicted_class": [
            analysis_result.classifications[c.candidate.candidate_id].predicted_class.value
            if c.candidate.candidate_id in analysis_result.classifications
            and analysis_result.classifications[c.candidate.candidate_id].predicted_class
            is not None
            else "none"
            for c in refined_candidates
        ],
        "calibrated_confidence": [
            _cls_conf(c.candidate.candidate_id) for c in refined_candidates
        ],
        "prob_transit_like": [
            _cls_prob(c.candidate.candidate_id, MorphologyClass.TRANSIT_LIKE)
            for c in refined_candidates
        ],
        "prob_eclipse_like": [
            _cls_prob(c.candidate.candidate_id, MorphologyClass.ECLIPSE_LIKE)
            for c in refined_candidates
        ],
        "prob_stellar_variability": [
            _cls_prob(c.candidate.candidate_id, MorphologyClass.STELLAR_VARIABILITY)
            for c in refined_candidates
        ],
        "prob_other_artifact": [
            _cls_prob(c.candidate.candidate_id, MorphologyClass.OTHER_ARTIFACT)
            for c in refined_candidates
        ],
        "abstained": [
            bool(analysis_result.classifications[c.candidate.candidate_id].abstained)
            if c.candidate.candidate_id in analysis_result.classifications
            else False
            for c in refined_candidates
        ],
        "source_assessment": [
            analysis_result.classifications[c.candidate.candidate_id].source_assessment.value
            if c.candidate.candidate_id in analysis_result.classifications
            else "not_assessed"
            for c in refined_candidates
        ],
        "centroid_shift_sigma": [
            _vet_val(c.candidate.candidate_id, "centroid_shift_sigma", "centroid_shift_sigma")
            for c in refined_candidates
        ],
        "crowdsap_value": [
            _vet_val(c.candidate.candidate_id, "crowdsap", "crowdsap_value")
            for c in refined_candidates
        ],
        "sap_to_pdcsap_depth_ratio": [
            _vet_val(
                c.candidate.candidate_id,
                "sap_to_pdcsap_depth_ratio",
                "sap_to_pdcsap_depth_ratio",
            )
            for c in refined_candidates
        ],
        "difference_image_offset_pixels": [
            _vet_val(c.candidate.candidate_id, "difference_image_offset_pixels")
            for c in refined_candidates
        ],
        "difference_image_offset_arcsec": [
            _vet_val(c.candidate.candidate_id, "difference_image_offset_arcsec")
            for c in refined_candidates
        ],
        "centroid_shift_arcsec": [
            _vet_val(c.candidate.candidate_id, "centroid_shift_arcsec")
            for c in refined_candidates
        ],
        "aperture_depth_growth_ratio": [
            _vet_val(c.candidate.candidate_id, "aperture_depth_growth_ratio")
            for c in refined_candidates
        ],
    }

    schema = pa.schema(
        [
            ("candidate_id", pa.string()),
            ("period_days", pa.float64()),
            ("epoch_days", pa.float64()),
            ("duration_hours", pa.float64()),
            ("measured_depth_ppm", pa.float64()),
            ("dilution_corrected_depth_ppm", pa.float64()),
            ("observed_event_count", pa.int64()),
            ("detection_snr", pa.float64()),
            ("search_significance", pa.float64()),
            ("ingress_fraction", pa.float64()),
            ("chi2_reduced", pa.float64()),
            ("bic", pa.float64()),
            ("batman_delta_bic", pa.float64()),
            ("phase_coverage", pa.float64()),
            ("aliases_days", pa.list_(pa.float64())),
            ("flags", pa.list_(pa.string())),
            ("predicted_class", pa.string()),
            ("calibrated_confidence", pa.float64()),
            ("prob_transit_like", pa.float64()),
            ("prob_eclipse_like", pa.float64()),
            ("prob_stellar_variability", pa.float64()),
            ("prob_other_artifact", pa.float64()),
            ("abstained", pa.bool_()),
            ("source_assessment", pa.string()),
            ("centroid_shift_sigma", pa.float64()),
            ("crowdsap_value", pa.float64()),
            ("sap_to_pdcsap_depth_ratio", pa.float64()),
            ("difference_image_offset_pixels", pa.float64()),
            ("aperture_depth_growth_ratio", pa.float64()),
        ]
    )

    table = pa.Table.from_pydict(columns, schema=schema)
    refined_ids = {ref.candidate.candidate_id for ref in refined_candidates}
    failed_rows: list[dict[str, Any]] = []
    for candidate in analysis_result.candidates:
        if candidate.candidate_id in refined_ids:
            continue
        row: dict[str, Any] = dict.fromkeys(schema.names)
        row.update({
            "candidate_id": candidate.candidate_id,
            "period_days": candidate.period_days,
            "epoch_days": candidate.epoch_days,
            "duration_hours": candidate.duration_hours,
            "measured_depth_ppm": candidate.depth_ppm,
            "observed_event_count": candidate.observed_event_count,
            "detection_snr": candidate.detection_snr,
            "aliases_days": list(candidate.aliases_days),
            "flags": list(candidate.flags),
            "predicted_class": "none",
            "abstained": True,
            "source_assessment": "not_assessed",
        })
        failed_rows.append(row)
    if failed_rows:
        table = pa.concat_tables([table, pa.Table.from_pylist(failed_rows, schema=schema)])
    pq.write_table(table, target_dir / "candidates.parquet")

    if analysis_result.schema_version == "3":
        manifest = {"sha256": {name: _file_checksum(target_dir / name) for name in _BUNDLE_COMPONENTS}}
        (target_dir / "integrity.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    return target_dir


def read_result_bundle(bundle_dir: Path) -> dict[str, Any]:
    """Load and validate result bundle components from directory."""
    path = bundle_dir.expanduser().resolve()
    result_json_path = path / "result.json"
    arrays_npz_path = path / "arrays.npz"
    parquet_path = path / "candidates.parquet"

    if not result_json_path.is_file():
        raise FileNotFoundError(f"Missing result.json in {path}")
    if not arrays_npz_path.is_file():
        raise FileNotFoundError(f"Missing arrays.npz in {path}")
    if not parquet_path.is_file():
        raise FileNotFoundError(f"Missing candidates.parquet in {path}")

    result_doc = json.loads(result_json_path.read_text(encoding="utf-8"))
    arrays = dict(np.load(arrays_npz_path))
    candidates_table = pq.read_table(parquet_path)

    return {
        "result": result_doc,
        "arrays": arrays,
        "candidates": candidates_table,
    }
