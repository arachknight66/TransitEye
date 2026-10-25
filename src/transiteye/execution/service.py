"""Scientific analysis service API shared between CLI and background/desktop runners."""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from transiteye.characterization.fitting import RefinedCandidate, refine_candidate
from transiteye.characterization.uncertainty import (
    compute_empirical_significance,
    compute_parameter_intervals,
    compute_red_noise_snr,
)
from transiteye.classification.models import ModelBundle
from transiteye.classification.train import get_default_morphology_model
from transiteye.config import AppConfig, default_config, ensure_workspace
from transiteye.detection.peaks import find_candidates
from transiteye.domain.models import (
    AnalysisResult,
    AnalysisStatus,
    Candidate,
    Classification,
    Interval,
    LightCurve,
    MissingReason,
)
from transiteye.features.extractors import extract_features
from transiteye.io.tpf import TargetPixelData, read_target_pixel_fits
from transiteye.preprocessing.detrend import two_pass_detrend
from transiteye.preprocessing.representation import prepare_light_curve
from transiteye.reporting.bundles import write_result_bundle
from transiteye.vetting.source import VettingEvidence, vet_source

logger = logging.getLogger(__name__)


def analyze_light_curve(
    light_curve: LightCurve,
    config: AppConfig | None = None,
    max_candidates: int = 3,
    snr_threshold: float = 6.0,
    screen_snr_threshold: float = 5.0,
    device: str = "auto",
    model_bundle: ModelBundle | None = None,
    tpf_path: Path | None = None,
    tpf_data: TargetPixelData | None = None,
    output_dir: Path | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[AnalysisResult, Path]:
    """Execute end-to-end scientific analysis on a delivered light curve.

    Produces typed AnalysisResult contract and writes structured result bundle.
    """
    cfg = config or default_config()
    ensure_workspace(cfg)

    t0 = time.time()
    target_id = light_curve.provenance.target_id or "unknown"
    analysis_id = f"analysis_{target_id}_{int(t0)}"

    # Load TPF if path is provided and object not pre-loaded
    if tpf_path is not None and tpf_data is None:
        try:
            tpf_data = read_target_pixel_fits(tpf_path)
        except Exception as e:
            logger.warning("Failed to load companion TPF %s: %s", tpf_path, e)

    def _progress(stage: str, percent: int) -> None:
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": stage,
                    "percent": percent,
                    "fraction": round(percent / 100.0, 3),
                    "target_id": target_id,
                    "analysis_id": analysis_id,
                    "timestamp": time.time(),
                }
            )

    _progress("preparing_light_curve", 10)

    # 1. Quality filtering and segmentation
    try:
        prepared = prepare_light_curve(light_curve)
    except Exception as exc:
        res = AnalysisResult(
            analysis_id=analysis_id,
            status=AnalysisStatus.INSUFFICIENT_DATA,
            provenance=light_curve.provenance,
            candidates=(),
            error_message=str(exc),
            configuration_version=cfg.version,
            schema_version="2",
        )
        dest_dir = output_dir or (cfg.workspace.results_directory / analysis_id)
        write_result_bundle(res, None, None, (), dest_dir)
        _progress("complete", 100)
        return res, dest_dir

    _progress("detrending_pass_1", 25)

    # 2. Pass 1 Detrending
    dual_p1 = two_pass_detrend(prepared, baseline_window_days=0.75, device=device)

    _progress("searching_candidates", 45)

    # 3. Blind BLS and Lomb-Scargle search
    detection = find_candidates(
        dual_p1,
        max_candidates=max_candidates,
        snr_threshold=snr_threshold,
        device=device,
    )

    if not detection.candidates:
        res = AnalysisResult(
            analysis_id=analysis_id,
            status=AnalysisStatus.NO_SIGNIFICANT_SIGNAL,
            provenance=light_curve.provenance,
            candidates=(),
            configuration_version=cfg.version,
            schema_version="2",
        )
        dest_dir = output_dir or (cfg.workspace.results_directory / analysis_id)
        write_result_bundle(res, dual_p1, detection, (), dest_dir)
        _progress("complete", 100)
        return res, dest_dir

    _progress("detrending_pass_2", 60)

    # 4. Pass 2 Detrending (masking candidate events)
    candidate_tuples = tuple(
        (c.epoch_days or 0.0, c.period_days or 1.0, c.duration_hours or 2.0)
        for c in detection.candidates
    )
    dual_final = two_pass_detrend(
        prepared,
        candidate_events=candidate_tuples,
        device=device,
    )

    _progress("characterizing_and_uncertainty", 75)

    # 5. Candidate refinement, Red-noise SNR, Block-bootstrap, and Empirical Significance
    crowdsap_str = light_curve.metadata.get("crowdsap")
    crowdsap = float(crowdsap_str) if crowdsap_str and crowdsap_str != "None" else None

    refined_list: list[RefinedCandidate] = []
    updated_candidates: list[Candidate] = []
    intervals_map: dict[str, dict[str, dict[str, Interval]]] = {}
    classifications_map: dict[str, Classification] = {}
    vetting_map: dict[str, VettingEvidence] = {}
    features_map: dict[str, dict[str, float]] = {}
    missing_reasons: dict[str, MissingReason] = {}

    active_model = model_bundle or get_default_morphology_model(
        cfg.workspace.root / "models"
    )

    for cand in detection.candidates:
        pass1_snr = float(cand.detection_snr or 0.0)
        is_fast_screened = pass1_snr < screen_snr_threshold

        # When fast-screened, skip batman fit by using a high batman threshold
        refined = refine_candidate(
            cand,
            dual_final.prepared.time_days,
            dual_final.detrended_flux,
            dual_final.prepared.flux_error_normalized,
            crowdsap=crowdsap,
            batman_snr_threshold=1e9 if is_fast_screened else 7.0,
        )
        refined_list.append(refined)

        # Red-noise corrected SNR
        red_snr = compute_red_noise_snr(
            dual_final.detrended_flux,
            refined.trapezoid_model_curve,
            dual_final.prepared.time_days,
            depth_ppm=refined.measured_depth_ppm,
            duration_hours=refined.duration_hours,
            observed_event_count=cand.observed_event_count or 1,
        )

        seed = int(hashlib.sha256(f"{target_id}_{cand.candidate_id}".encode()).hexdigest()[:8], 16)

        if is_fast_screened:
            emp_sig = None
            missing_reasons[f"{cand.candidate_id}.search_significance"] = (
                MissingReason.NOT_COMPUTED
            )
            missing_reasons[f"{cand.candidate_id}.parameter_intervals"] = (
                MissingReason.NOT_COMPUTED
            )
            intervals_map[cand.candidate_id] = {
                param_name: {
                    "68": Interval(
                        lower=None,
                        upper=None,
                        coverage=0.68,
                        missing_reason=MissingReason.NOT_COMPUTED,
                    ),
                    "95": Interval(
                        lower=None,
                        upper=None,
                        coverage=0.95,
                        missing_reason=MissingReason.NOT_COMPUTED,
                    ),
                }
                for param_name in ("period_days", "epoch_days", "duration_hours", "depth_ppm")
            }
        else:
            # Seeded empirical search significance
            emp_sig, _ = compute_empirical_significance(
                dual_final.prepared.time_days,
                dual_final.detrended_flux,
                refined.trapezoid_model_curve,
                candidate_period=refined.period_days,
                candidate_duration_hours=refined.duration_hours,
                observed_snr=red_snr,
                n_surrogates=200,
                random_seed=seed,
                device=device,
            )

            # Block-bootstrap intervals
            intervals = compute_parameter_intervals(
                dual_final.prepared.time_days,
                dual_final.detrended_flux,
                dual_final.prepared.flux_error_normalized,
                refined=refined,
                n_boot=200,
                random_seed=seed,
            )
            intervals_map[cand.candidate_id] = intervals

        # Re-pack Candidate with refined parameters, red_snr, and empirical significance
        updated_cand = Candidate(
            candidate_id=cand.candidate_id,
            period_days=refined.period_days,
            epoch_days=refined.epoch_days,
            duration_hours=refined.duration_hours,
            depth_ppm=refined.measured_depth_ppm,
            observed_event_count=cand.observed_event_count,
            detection_snr=red_snr,
            search_significance=emp_sig,
            aliases_days=cand.aliases_days,
            flags=cand.flags,
        )
        updated_candidates.append(updated_cand)

        # Extract features, vet source, and classify morphology
        features = extract_features(
            dual_final, refined, detection.variability, light_curve
        )
        features_map[cand.candidate_id] = features
        evidence = vet_source(dual_final, refined, tpf_data=tpf_data)
        vetting_map[cand.candidate_id] = evidence
        classification = active_model.predict(
            features, source_assessment=evidence.assessment
        )
        classifications_map[cand.candidate_id] = classification

    _progress("writing_result_bundle", 90)

    analysis_res = AnalysisResult(
        analysis_id=analysis_id,
        status=AnalysisStatus.COMPLETE,
        provenance=light_curve.provenance,
        candidates=tuple(updated_candidates),
        classifications=classifications_map,
        parameter_intervals=intervals_map,
        missing_reasons=missing_reasons,
        configuration_version=cfg.version,
        schema_version="2",
    )

    dest_dir = output_dir or (cfg.workspace.results_directory / analysis_id)
    write_result_bundle(
        analysis_res,
        dual_final,
        detection,
        tuple(refined_list),
        dest_dir,
        vetting_map=vetting_map,
        features_map=features_map,
    )

    _progress("complete", 100)
    return analysis_res, dest_dir
