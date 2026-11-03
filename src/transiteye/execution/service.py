"""Scientific analysis service API shared between CLI and background/desktop runners."""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from transiteye import __version__
from transiteye.characterization.fitting import RefinedCandidate, refine_candidate
from transiteye.characterization.uncertainty import (
    compute_parameter_intervals,
    compute_red_noise_snr,
)
from transiteye.classification.models import ModelBundle
from transiteye.config import AppConfig, default_config, ensure_workspace
from transiteye.detection.peaks import find_candidates
from transiteye.domain.models import (
    AnalysisResult,
    AnalysisSettings,
    AnalysisStatus,
    Candidate,
    Classification,
    Interval,
    LightCurve,
    MissingReason,
    SourceAssessment,
)
from transiteye.execution.compute import resolve_device
from transiteye.features.extractors import extract_features, stellar_temperature
from transiteye.io.tpf import TargetPixelData, read_target_pixel_fits
from transiteye.preprocessing.detrend import two_pass_detrend
from transiteye.preprocessing.representation import prepare_light_curve
from transiteye.provenance import code_fingerprint
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
    analysis_settings: AnalysisSettings | None = None,
    execution_device: str | None = None,
) -> tuple[AnalysisResult, Path]:
    """Execute end-to-end scientific analysis on a delivered light curve.

    Produces typed AnalysisResult contract and writes structured result bundle.
    When supplied, analysis_settings takes precedence over the legacy analysis arguments.
    """
    cfg = config or default_config()
    ensure_workspace(cfg)

    settings = analysis_settings or AnalysisSettings(
        max_candidates=max_candidates,
        snr_threshold=snr_threshold,
        screen_snr_threshold=screen_snr_threshold,
        device=device,
    )
    max_candidates = settings.max_candidates
    snr_threshold = settings.snr_threshold
    screen_snr_threshold = settings.screen_snr_threshold
    device = resolve_device(execution_device or settings.device)
    result_provenance = {
        "analysis_settings": settings,
        "code_version": __version__,
        "code_sha256": code_fingerprint(),
        "resolved_device": device,
        "model_id": model_bundle.metadata.model_id if model_bundle is not None else None,
        "model_sha256": model_bundle.artifact_sha256 if model_bundle is not None else None,
    }

    target_id = light_curve.provenance.target_id or "unknown"
    analysis_id = f"analysis_{target_id}_{uuid.uuid4().hex}"

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
        prepared = prepare_light_curve(
            light_curve, quality_mask=settings.quality_mask,
            gap_threshold_hours=settings.gap_threshold_hours,
            reject_positive_outliers_sigma=settings.reject_positive_outliers_sigma,
        )
    except Exception as exc:
        res = AnalysisResult(
            analysis_id=analysis_id,
            status=AnalysisStatus.INSUFFICIENT_DATA,
            provenance=light_curve.provenance,
            candidates=(),
            error_message=str(exc),
            configuration_version=cfg.version,
            schema_version="3",
            **result_provenance,
        )
        dest_dir = output_dir or (cfg.workspace.results_directory / analysis_id)
        write_result_bundle(res, None, None, (), dest_dir)
        _progress("complete", 100)
        return res, dest_dir

    if len(prepared.time_days) < 200 or prepared.time_days[-1] - prepared.time_days[0] < 2.0:
        res = AnalysisResult(
            analysis_id=analysis_id,
            status=AnalysisStatus.INSUFFICIENT_DATA,
            provenance=light_curve.provenance,
            candidates=(),
            error_message="At least 200 valid cadences spanning two days are required for the search.",
            configuration_version=cfg.version,
            schema_version="3",
            **result_provenance,
        )
        dest_dir = output_dir or (cfg.workspace.results_directory / analysis_id)
        write_result_bundle(res, None, None, (), dest_dir)
        _progress("complete", 100)
        return res, dest_dir

    _progress("detrending_pass_1", 25)

    # 2. Pass 1 Detrending
    dual_p1 = two_pass_detrend(
        prepared, baseline_window_days=settings.baseline_window_days, device=device
    )

    _progress("searching_candidates", 45)

    # 3. Blind BLS and Lomb-Scargle search
    detection = find_candidates(
        dual_p1,
        max_candidates=max_candidates,
        snr_threshold=snr_threshold,
        device=device,
        min_period=settings.bls_min_period_days,
        max_period=settings.bls_max_period_days,
        min_duration_hours=settings.bls_min_duration_hours,
        max_duration_hours=settings.bls_max_duration_hours,
        num_durations=settings.bls_num_durations,
        frequency_factor=settings.bls_frequency_factor,
    )

    if not detection.candidates:
        res = AnalysisResult(
            analysis_id=analysis_id,
            status=AnalysisStatus.NO_SIGNIFICANT_SIGNAL,
            provenance=light_curve.provenance,
            candidates=(),
            configuration_version=cfg.version,
            schema_version="3",
            **result_provenance,
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
        baseline_window_days=settings.baseline_window_days,
        candidate_events=candidate_tuples,
        device=device,
    )

    _progress("characterizing_and_uncertainty", 75)

    # 5. Candidate refinement, Red-noise SNR, Block-bootstrap, and Empirical Significance
    crowdsap_str = light_curve.metadata.get("crowdsap")
    crowdsap = float(crowdsap_str) if crowdsap_str and crowdsap_str != "None" else None
    teff = stellar_temperature(light_curve)

    refined_list: list[RefinedCandidate] = []
    updated_candidates: list[Candidate] = []
    intervals_map: dict[str, dict[str, dict[str, Interval]]] = {}
    classifications_map: dict[str, Classification] = {}
    vetting_map: dict[str, VettingEvidence] = {}
    features_map: dict[str, dict[str, float]] = {}
    missing_reasons: dict[str, MissingReason] = {}
    candidate_errors: dict[str, str] = {}
    candidate_seeds: dict[str, int] = {}
    failed_candidates: list[Candidate] = []

    active_model = model_bundle

    for cand in detection.candidates:
        pass1_snr = float(cand.detection_snr or 0.0)
        is_fast_screened = pass1_snr < screen_snr_threshold

        # When fast-screened, skip batman fit by using a high batman threshold
        try:
            refined = refine_candidate(
                cand,
                dual_final.prepared.time_days,
                dual_final.detrended_flux,
                dual_final.prepared.flux_error_normalized,
                crowdsap=crowdsap,
                batman_snr_threshold=1e9 if is_fast_screened else 7.0,
                teff=teff,
                exposure_days=settings.trapezoid_exposure_days,
            )
        except Exception as exc:
            logger.exception("Candidate fit failed for %s", cand.candidate_id)
            candidate_errors[cand.candidate_id] = f"{type(exc).__name__}: {exc}"
            failed_candidates.append(replace(cand, flags=(*cand.flags, "fit_failed", "preliminary_detection")))
            for field_name in ("fit", "parameter_intervals", "classification", "source_assessment"):
                missing_reasons[f"{cand.candidate_id}.{field_name}"] = MissingReason.FIT_FAILED
            missing_reasons[f"{cand.candidate_id}.search_significance"] = MissingReason.NOT_COMPUTED
            classifications_map[cand.candidate_id] = Classification(
                probabilities={}, predicted_class=None, abstained=True,
                source_assessment=SourceAssessment.NOT_ASSESSED, model_version=None,
            )
            continue
        if refined.batman_model_curve is not None and "physical_quadrature_unvalidated" not in cand.flags:
            cand = replace(cand, flags=(*cand.flags, "physical_quadrature_unvalidated"))
            refined = replace(refined, candidate=cand)
        elif (
            refined.batman_model_curve is None
            and not is_fast_screened
            and pass1_snr >= 7.0
            and refined.ingress_fraction < 0.8
        ):
            if "physical_fit_failed" not in cand.flags:
                cand = replace(cand, flags=(*cand.flags, "physical_fit_failed"))
                refined = replace(refined, candidate=cand)
            missing_reasons[f"{cand.candidate_id}.physical_fit"] = MissingReason.FIT_FAILED
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
        candidate_seeds[cand.candidate_id] = seed

        # The former fixed-phase surrogate test did not repeat the blind search and
        # cannot quantify search-wide significance. Do not report it as such.
        emp_sig = None
        missing_reasons[f"{cand.candidate_id}.search_significance"] = MissingReason.NOT_COMPUTED
        if is_fast_screened:
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
            # Block-bootstrap intervals
            intervals = compute_parameter_intervals(
                dual_final.prepared.time_days,
                dual_final.detrended_flux,
                dual_final.prepared.flux_error_normalized,
                refined=refined,
                n_boot=settings.bootstrap_replicates,
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
        if evidence.centroid_shift_sigma is None:
            missing_reasons[f"{cand.candidate_id}.centroid_shift"] = MissingReason.SOURCE_DATA_MISSING
        if evidence.crowdsap is None:
            missing_reasons[f"{cand.candidate_id}.crowdsap"] = MissingReason.SOURCE_DATA_MISSING
        if evidence.sap_to_pdcsap_depth_ratio is None:
            missing_reasons[f"{cand.candidate_id}.sap_to_pdcsap_depth_ratio"] = MissingReason.SOURCE_DATA_MISSING
        if tpf_data is not None and evidence.difference_image_offset_pixels is None:
            missing_reasons[f"{cand.candidate_id}.difference_image_offset"] = MissingReason.INSUFFICIENT_DATA
        if active_model is None:
            missing_reasons[f"{cand.candidate_id}.classification"] = MissingReason.NOT_COMPUTED
            classification = Classification(
                probabilities={}, predicted_class=None, abstained=True,
                source_assessment=evidence.assessment, model_version=None,
            )
        else:
            classification = active_model.predict(features, source_assessment=evidence.assessment)
        classifications_map[cand.candidate_id] = classification

    _progress("writing_result_bundle", 90)

    analysis_res = AnalysisResult(
        analysis_id=analysis_id,
        status=AnalysisStatus.COMPLETE,
        provenance=light_curve.provenance,
        candidates=tuple(updated_candidates + failed_candidates),
        classifications=classifications_map,
        parameter_intervals=intervals_map,
        missing_reasons=missing_reasons,
        candidate_errors=candidate_errors,
        candidate_seeds=candidate_seeds,
        configuration_version=cfg.version,
        schema_version="3",
        **result_provenance,
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
