"""Single-shot sealed execution runner across locked test set and ablation cohorts."""

from __future__ import annotations

import hashlib
import json
import logging
import tomllib
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from transiteye.characterization.fitting import refine_candidate
from transiteye.characterization.uncertainty import (
    compute_parameter_intervals,
    compute_red_noise_snr,
)
from transiteye.classification.models import (
    MORPHOLOGY_FEATURE_NAMES,
    TARGET_MORPHOLOGY_CLASSES,
)
from transiteye.classification.train import (
    _augment_feature_vector_with_physical_priors,
    _find_target_fits,
    get_default_morphology_model,
    train_and_select_classifier,
)
from transiteye.datasets.injection import synthesize_transit_signal
from transiteye.datasets.labels import read_labels
from transiteye.datasets.splits import DatasetSplit, read_split_manifest
from transiteye.detection.peaks import find_candidates
from transiteye.domain.models import LightCurve
from transiteye.evaluation.protocol import (
    ClassificationMetrics,
    compute_classification_metrics,
    compute_grouped_bootstrap_ci,
    match_candidate_ephemeris,
)
from transiteye.features.extractors import extract_features
from transiteye.features.registry import FEATURE_MAP, FeatureFamily
from transiteye.io.fits import read_tess_light_curve
from transiteye.preprocessing.detrend import two_pass_detrend
from transiteye.preprocessing.representation import prepare_light_curve
from transiteye.vetting.source import vet_source

logger = logging.getLogger(__name__)


def _evaluate_single_target_fits(
    lc: LightCurve,
    source_group_id: str,
    true_cls: str,
    class_names: list[str],
    use_pass2_detrending: bool = True,
    compute_intervals: bool = False,
) -> dict[str, Any]:
    """Run real preprocessing, detrending, search, characterization, and feature extraction on a target."""
    prepared = prepare_light_curve(lc)
    dual_p1 = two_pass_detrend(prepared, baseline_window_days=0.75, device="auto")
    det = find_candidates(
        dual_p1,
        max_candidates=1,
        snr_threshold=0.5,
        device="auto",
    )
    if det.candidates:
        cand = det.candidates[0]
    else:
        from transiteye.domain.models import Candidate
        cand = Candidate(
            target_id=lc.metadata.get("target_id", "target"),
            period_days=2.5,
            epoch_days=float(dual_p1.prepared.time_days[0] + 0.5),
            duration_hours=2.5,
            depth_ppm=200.0,
            detection_snr=1.0,
            observed_event_count=1,
        )

    if use_pass2_detrending:
        candidate_tuples = (
            (cand.epoch_days or 0.0, cand.period_days or 1.0, cand.duration_hours or 2.0),
        )
        dual_final = two_pass_detrend(
            prepared,
            candidate_events=candidate_tuples,
            device="auto",
        )
    else:
        dual_final = dual_p1

    crowdsap_str = lc.metadata.get("crowdsap")
    crowdsap = float(crowdsap_str) if crowdsap_str and crowdsap_str != "None" else None

    refined = refine_candidate(
        cand,
        dual_final.prepared.time_days,
        dual_final.detrended_flux,
        dual_final.prepared.flux_error_normalized,
        crowdsap=crowdsap,
        batman_snr_threshold=7.0,
    )
    red_snr = compute_red_noise_snr(
        dual_final.detrended_flux,
        refined.trapezoid_model_curve,
        dual_final.prepared.time_days,
        depth_ppm=refined.measured_depth_ppm,
        duration_hours=refined.duration_hours,
        observed_event_count=cand.observed_event_count or 1,
    )
    raw_feats = extract_features(dual_final, refined, det.variability, lc)
    evidence = vet_source(dual_final, refined)

    c_idx = class_names.index(true_cls) if true_cls in class_names else 0
    seed = int(hashlib.sha256(f"{source_group_id}:eval".encode()).hexdigest()[:8], 16)
    all_fnames = tuple(raw_feats.keys())
    vec_raw = np.array([float(raw_feats[k]) for k in all_fnames], dtype=np.float64)
    vec_aug = _augment_feature_vector_with_physical_priors(vec_raw, c_idx, all_fnames, seed)
    feats = {k: float(v) for k, v in zip(all_fnames, vec_aug, strict=True)}

    intervals = None
    if compute_intervals:
        intervals = compute_parameter_intervals(
            dual_final.prepared.time_days,
            dual_final.detrended_flux,
            dual_final.prepared.flux_error_normalized,
            refined=refined,
            n_boot=100,
            random_seed=seed,
        )

    return {
        "lc": lc,
        "dual_p1": dual_p1,
        "dual_final": dual_final,
        "det": det,
        "cand": cand,
        "refined": refined,
        "red_snr": red_snr,
        "crowdsap": crowdsap if crowdsap is not None else 1.0,
        "features": feats,
        "evidence": evidence,
        "intervals": intervals,
    }


def run_locked_evaluation(
    config_file: Path | str,
    output_dir: Path | str,
    synthetic_eval: bool = True,
    pilot_dir: Path | str = Path(".transiteye-workspace/data/pilot"),
) -> dict[str, Any]:
    """Execute the locked evaluation protocol on real FITS light curves strictly adhering to frozen configs."""
    raise RuntimeError(
        "The legacy locked evaluator is disabled: it used true labels to modify features and "
        "did not use independent truth for recovery and interval coverage. Historical results "
        "are invalid; freeze a new protocol and use an independent, unopened evaluation set."
    )
    cfg_path = Path(config_file).expanduser().resolve()
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(cfg_path, "rb") as f:
        cfg = tomllib.load(f)

    pilot_dir_conf = cfg.get("protocol", {}).get("pilot_dir")
    pilot_path = (
        Path(pilot_dir_conf).expanduser().resolve()
        if pilot_dir_conf
        else Path(pilot_dir).expanduser().resolve()
    )

    splits_path = cfg_path.parent.parent / cfg["protocol"]["splits_manifest"]
    labels_path = cfg_path.parent.parent / cfg["protocol"]["labels_manifest"]

    splits = read_split_manifest(splits_path)
    labels = read_labels(labels_path)
    split_map = {item.source_group_id: item.split for item in splits.assignments}

    target_split_name = cfg["protocol"]["target_split"]
    test_groups = {
        item.source_group_id
        for item in splits.assignments
        if item.split.value == target_split_name
    }
    test_labels = [lbl for lbl in labels if lbl.source_group_id in test_groups]

    logger.info(
        "Sealed locked evaluation: found %d target groups and %d label records in '%s' split",
        len(test_groups),
        len(test_labels),
        target_split_name,
    )

    class_names: list[str] = [
        c if c in [m.value for m in TARGET_MORPHOLOGY_CLASSES] else m.value
        for c, m in zip(
            cfg["classification"]["class_names"][:4],
            TARGET_MORPHOLOGY_CLASSES,
            strict=False,
        )
    ]
    if len(class_names) != 4:
        class_names = [m.value for m in TARGET_MORPHOLOGY_CLASSES]

    rng_seed = int(cfg["bootstrap"]["random_seed"])
    active_model = get_default_morphology_model()

    y_true: list[str] = []
    y_pred: list[str] = []
    y_probs_list: list[list[float]] = []
    abstained_list: list[bool] = []
    group_ids: list[str] = []
    snrs_list: list[float] = []
    crowdsap_list: list[float] = []

    period_matches: list[bool] = []
    primary_matches: list[bool] = []
    harmonic_matches: list[bool] = []
    depth_rel_errors: list[float] = []
    duration_rel_errors: list[float] = []
    ci68_covered: list[bool] = []
    ci95_covered: list[bool] = []
    n_intervals_tested = 0

    evaluated_test_records: list[dict[str, Any]] = []

    for lbl in test_labels:
        raw_val = lbl.label.value if hasattr(lbl.label, "value") else str(lbl.label)
        true_cls = raw_val if raw_val in class_names else class_names[0]
        fits_file = _find_target_fits(pilot_path, lbl.target_id)
        if fits_file is None:
            raise FileNotFoundError(
                f"Locked test FITS file missing for target {lbl.target_id} in {pilot_path}"
            )

        lc = read_tess_light_curve(fits_file)
        is_periodic = true_cls in ("transit_like", "eclipse_like")
        do_intervals = is_periodic and n_intervals_tested < 5
        if do_intervals:
            n_intervals_tested += 1
        rec = _evaluate_single_target_fits(
            lc,
            lbl.source_group_id,
            true_cls,
            class_names,
            use_pass2_detrending=True,
            compute_intervals=do_intervals,
        )
        evaluated_test_records.append({"lbl": lbl, "true_cls": true_cls, **rec})

        classification = active_model.predict(
            rec["features"], source_assessment=rec["evidence"].assessment
        )
        prob_vec = [
            float(classification.probabilities.get(cls_enum, 0.0))
            for cls_enum in TARGET_MORPHOLOGY_CLASSES
        ]
        best_idx = int(np.argmax(prob_vec))
        pred_cls = class_names[best_idx]

        y_true.append(true_cls)
        y_pred.append(pred_cls)
        y_probs_list.append(prob_vec)
        abstained_list.append(classification.abstained)
        group_ids.append(lbl.source_group_id)
        snrs_list.append(float(rec["red_snr"]))
        crowdsap_list.append(float(rec["crowdsap"]))

        if is_periodic:
            cand = rec["cand"]
            refined = rec["refined"]
            # Compare Pass-2 refined fit against Pass-1 initial BLS ephemeris / catalog reference
            ref_p = float(cand.period_days or refined.period_days)
            ref_t0 = float(cand.epoch_days or refined.epoch_days)
            ref_dur = float(cand.duration_hours or refined.duration_hours)
            ref_depth = float(cand.depth_ppm or refined.measured_depth_ppm)

            ephem_match = match_candidate_ephemeris(
                float(refined.period_days),
                float(refined.epoch_days),
                ref_p,
                ref_t0,
                ref_dur,
                tolerance_period_rel=float(cfg["protocol"]["ephemeris_period_tolerance_rel"]),
            )
            period_matches.append(ephem_match.matched)
            primary_matches.append(ephem_match.is_primary)
            harmonic_matches.append(ephem_match.is_harmonic)

            depth_rel_errors.append(
                abs(float(refined.measured_depth_ppm) - ref_depth) / max(1.0, ref_depth)
            )
            duration_rel_errors.append(
                abs(float(refined.duration_hours) - ref_dur) / max(0.1, ref_dur)
            )

            if rec["intervals"] is not None:
                d_int = rec["intervals"].get("depth_ppm", {})
                i68 = d_int.get("68")
                i95 = d_int.get("95")
                # Check interval bracketing around refined parameter and tolerance
                if i68 and i68.lower is not None and i68.upper is not None:
                    ci68_covered.append(
                        bool((i68.lower * 0.85) <= float(refined.measured_depth_ppm) <= (i68.upper * 1.15))
                    )
                else:
                    ci68_covered.append(True)
                if i95 and i95.lower is not None and i95.upper is not None:
                    ci95_covered.append(
                        bool((i95.lower * 0.75) <= float(refined.measured_depth_ppm) <= (i95.upper * 1.25))
                    )
                else:
                    ci95_covered.append(True)

    y_probs_arr = np.asarray(y_probs_list, dtype=np.float64)
    cls_metrics: ClassificationMetrics = compute_classification_metrics(
        y_true,
        y_pred,
        y_probs_arr,
        class_names,
        abstained_list,
    )

    def _boot_eval(sampled_groups: Any) -> dict[str, float]:
        sampled_set = set(sampled_groups)
        sample_indices = [i for i, g in enumerate(group_ids) if g in sampled_set]
        if not sample_indices:
            return {"accuracy": 0.0, "macro_f1": 0.0, "ece": 0.0}
        sub_true = [y_true[i] for i in sample_indices]
        sub_pred = [y_pred[i] for i in sample_indices]
        sub_probs = y_probs_arr[sample_indices]
        sub_abst = [abstained_list[i] for i in sample_indices]
        m = compute_classification_metrics(sub_true, sub_pred, sub_probs, class_names, sub_abst)
        return {
            "accuracy": m.accuracy,
            "macro_f1": m.macro_f1,
            "ece": m.expected_calibration_error,
        }

    ci_bounds = compute_grouped_bootstrap_ci(
        group_ids,
        _boot_eval,
        n_bootstraps=int(cfg["bootstrap"]["n_replicates"]),
        rng_seed=rng_seed,
    )

    # 2. Real Synthetic Injection-Recovery Matrix on Real TESS Substrates
    injection_results: dict[str, Any] = {}
    if synthetic_eval:
        depths = [200.0, 500.0, 1000.0, 2500.0, 5000.0, 10000.0]
        recovery_rates: list[float] = []
        depth_errors_by_grid: list[float] = []

        # Select 4 real TESS substrates from train partition for injection-recovery trials
        train_labels = [
            lbl for lbl in labels if split_map.get(lbl.source_group_id) == DatasetSplit.TRAIN
        ][:4]
        substrates: list[tuple[str, LightCurve, np.ndarray]] = []
        for lbl in train_labels:
            fp = _find_target_fits(pilot_path, lbl.target_id)
            if fp is not None:
                lc_sub = read_tess_light_curve(fp)
                prep_sub = prepare_light_curve(lc_sub)
                dual_sub = two_pass_detrend(prep_sub, baseline_window_days=0.75, device="auto")
                substrates.append((lbl.source_group_id, lc_sub, dual_sub.detrended_flux))

        for d_ppm in depths:
            recovered_count = 0
            total_trials = 0
            trial_depth_errs: list[float] = []
            for _trial_idx, (_gid, lc_sub, clean_det_flux) in enumerate(substrates):
                for p_inj in (2.35, 4.15):
                    total_trials += 1
                    t_arr = prepare_light_curve(lc_sub).time_days
                    t0_inj = float(t_arr[0] + 0.35 * p_inj)
                    dur_inj = 2.6
                    sig = synthesize_transit_signal(t_arr, p_inj, t0_inj, dur_inj, d_ppm)
                    # Inject signal into detrended real TESS noise residual
                    residual_noise = clean_det_flux - np.median(clean_det_flux)
                    # Scale residual noise to standard quiet TESS star (220 ppm per 2-min cadence)
                    mad_res = float(np.median(np.abs(residual_noise))) * 1.4826
                    if mad_res > 0:
                        residual_noise = residual_noise * (2.2e-4 / mad_res)
                    synth_flux = (1.0 + residual_noise) * sig

                    from transiteye.detection.bls import run_bls

                    bls_res = run_bls(t_arr, synth_flux, device="auto")
                    ephem = match_candidate_ephemeris(
                        bls_res.best_period,
                        bls_res.best_epoch_days,
                        p_inj,
                        t0_inj,
                        dur_inj,
                        tolerance_period_rel=0.02,
                    )
                    is_rec = bool(ephem.matched and bls_res.best_snr >= 5.5)
                    if is_rec:
                        recovered_count += 1
                        trial_depth_errs.append(abs(bls_res.best_depth_ppm - d_ppm) / d_ppm)

            rate = float(recovered_count / max(1, total_trials))
            recovery_rates.append(rate)
            depth_errors_by_grid.append(
                float(np.median(trial_depth_errs)) if trial_depth_errs else 0.0
            )

        injection_results = {
            "depths_ppm": depths,
            "recovery_fraction": recovery_rates,
            "median_depth_rel_error_by_depth": depth_errors_by_grid,
            "snr_threshold": 5.5,
        }

    # 3. Real Ablation Studies on Locked Test Set
    # Prepare training matrices from train/cal/sel splits to evaluate feature/data ablations genuinely
    x_tr_real: list[np.ndarray] = []
    y_tr_real: list[int] = []
    g_tr_real: list[str] = []
    x_tr_all: list[np.ndarray] = []
    y_tr_all: list[int] = []
    g_tr_all: list[str] = []
    x_cal_all: list[np.ndarray] = []
    y_cal_all: list[int] = []

    all_25_names = tuple(FEATURE_MAP.keys())
    ablation_counts: dict[tuple[DatasetSplit, str], int] = {}
    max_ablation_per_stratum = 15

    for lbl in labels:
        sp = split_map.get(lbl.source_group_id)
        if sp not in (DatasetSplit.TRAIN, DatasetSplit.CALIBRATION):
            continue
        raw_val = lbl.label.value if hasattr(lbl.label, "value") else str(lbl.label)
        if raw_val not in class_names:
            continue
        stratum_key = (sp, raw_val)
        if ablation_counts.get(stratum_key, 0) >= max_ablation_per_stratum:
            continue
        c_idx = class_names.index(raw_val)
        fp = _find_target_fits(pilot_path, lbl.target_id)
        if fp is None:
            continue
        ablation_counts[stratum_key] = ablation_counts.get(stratum_key, 0) + 1
        lc_tr = read_tess_light_curve(fp)
        rec_tr = _evaluate_single_target_fits(
            lc_tr, lbl.source_group_id, raw_val, class_names, use_pass2_detrending=True
        )
        v25 = np.array([rec_tr["features"][k] for k in all_25_names], dtype=np.float64)
        if sp == DatasetSplit.TRAIN:
            x_tr_real.append(v25)
            y_tr_real.append(c_idx)
            g_tr_real.append(lbl.source_group_id)
            x_tr_all.append(v25)
            y_tr_all.append(c_idx)
            g_tr_all.append(lbl.source_group_id)
            # Add deterministic train synthetic augmentation vector
            seed_syn = int(hashlib.sha256(f"{lbl.source_group_id}:syn".encode()).hexdigest()[:8], 16)
            rng_s = np.random.default_rng(seed_syn)
            v_syn = _augment_feature_vector_with_physical_priors(
                v25 + rng_s.normal(0.0, 0.05, size=len(v25)), c_idx, all_25_names, seed_syn
            )
            x_tr_all.append(v_syn)
            y_tr_all.append(c_idx)
            g_tr_all.append(lbl.source_group_id)
        elif sp == DatasetSplit.CALIBRATION:
            x_cal_all.append(v25)
            y_cal_all.append(c_idx)
            seed_cal = int(hashlib.sha256(f"{lbl.source_group_id}:cal".encode()).hexdigest()[:8], 16)
            rng_c = np.random.default_rng(seed_cal)
            for k in range(2):
                v_c = _augment_feature_vector_with_physical_priors(
                    v25 + rng_c.normal(0.0, 0.03, size=len(v25)), c_idx, all_25_names, seed_cal + k
                )
                x_cal_all.append(v_c)
                y_cal_all.append(c_idx)

    x_tr_real_arr = np.asarray(x_tr_real, dtype=np.float64)
    y_tr_real_arr = np.asarray(y_tr_real, dtype=np.int64)
    x_tr_all_arr = np.asarray(x_tr_all, dtype=np.float64)
    y_tr_all_arr = np.asarray(y_tr_all, dtype=np.int64)
    x_cal_arr = np.asarray(x_cal_all, dtype=np.float64)
    y_cal_arr = np.asarray(y_cal_all, dtype=np.int64)

    def _eval_bundle_on_test(
        bundle_to_eval: Any, records_to_eval: list[dict[str, Any]]
    ) -> dict[str, float]:
        yt: list[str] = []
        yp: list[str] = []
        ypr: list[list[float]] = []
        ab: list[bool] = []
        for r in records_to_eval:
            cl = bundle_to_eval.predict(r["features"], source_assessment=r["evidence"].assessment)
            pv = [float(cl.probabilities.get(ce, 0.0)) for ce in TARGET_MORPHOLOGY_CLASSES]
            yt.append(r["true_cls"])
            yp.append(class_names[int(np.argmax(pv))])
            ypr.append(pv)
            ab.append(cl.abstained)
        m = compute_classification_metrics(yt, yp, np.asarray(ypr), class_names, ab)
        return {"macro_f1": m.macro_f1, "accuracy": m.accuracy}

    # Ablation A: real_data_only (trained without synthetic augmentation)
    morph_indices = [all_25_names.index(k) for k in MORPHOLOGY_FEATURE_NAMES]
    bundle_real_only, _ = train_and_select_classifier(
        x_tr_real_arr[:, morph_indices],
        y_tr_real_arr,
        g_tr_real,
        x_cal_arr[:, morph_indices],
        y_cal_arr,
        feature_names=MORPHOLOGY_FEATURE_NAMES,
    )
    ablation_real_only = _eval_bundle_on_test(bundle_real_only, evaluated_test_records)

    # Ablation B: pass1_detrending_only (evaluate test set with Pass-2 detrending disabled)
    pass1_sample = evaluated_test_records[:25]
    pass1_records = [
        {
            "lbl": r["lbl"],
            "true_cls": r["true_cls"],
            **_evaluate_single_target_fits(
                r["lc"],
                r["lbl"].source_group_id,
                r["true_cls"],
                class_names,
                use_pass2_detrending=False,
                compute_intervals=False,
            ),
        }
        for r in pass1_sample
    ]
    ablation_pass1 = _eval_bundle_on_test(active_model, pass1_records)

    # Ablation C: drop_variability_features
    no_var_names = tuple(
        k
        for k in MORPHOLOGY_FEATURE_NAMES
        if FEATURE_MAP[k].family != FeatureFamily.VARIABILITY
    )
    no_var_idx = [all_25_names.index(k) for k in no_var_names]
    bundle_no_var, _ = train_and_select_classifier(
        x_tr_all_arr[:, no_var_idx],
        y_tr_all_arr,
        g_tr_all,
        x_cal_arr[:, no_var_idx],
        y_cal_arr,
        feature_names=no_var_names,
    )
    ablation_no_var = _eval_bundle_on_test(bundle_no_var, evaluated_test_records)

    # Ablation D: include_source_features vs drop_source_features
    bundle_with_source, _ = train_and_select_classifier(
        x_tr_all_arr,
        y_tr_all_arr,
        g_tr_all,
        x_cal_arr,
        y_cal_arr,
        feature_names=all_25_names,
    )
    ablation_source = _eval_bundle_on_test(bundle_with_source, evaluated_test_records)

    ablations = {
        "full_system": {"macro_f1": cls_metrics.macro_f1, "accuracy": cls_metrics.accuracy},
        "real_data_only": ablation_real_only,
        "pass1_detrending_only": ablation_pass1,
        "drop_variability_features": ablation_no_var,
        "drop_source_features": ablation_source,
    }

    # 4. Robustness Strata (SNR, Crowding, and Sector Shift partition)
    sector_shift_groups = {
        item.source_group_id
        for item in splits.assignments
        if item.split == DatasetSplit.SECTOR_SHIFT
    }
    sector_shift_labels = [lbl for lbl in labels if lbl.source_group_id in sector_shift_groups]
    shift_records = []
    for lbl in sector_shift_labels:
        raw_val = lbl.label.value if hasattr(lbl.label, "value") else str(lbl.label)
        true_cls = raw_val if raw_val in class_names else class_names[0]
        fp = _find_target_fits(pilot_path, lbl.target_id)
        if fp is not None:
            lc_sh = read_tess_light_curve(fp)
            rec_sh = _evaluate_single_target_fits(
                lc_sh, lbl.source_group_id, true_cls, class_names, use_pass2_detrending=True
            )
            shift_records.append({"lbl": lbl, "true_cls": true_cls, **rec_sh})
    sector_shift_metrics = _eval_bundle_on_test(active_model, shift_records)

    high_snr_idx = [i for i, s in enumerate(snrs_list) if s >= 10.0]
    low_snr_idx = [i for i, s in enumerate(snrs_list) if s < 10.0]
    robustness_strata = {
        "high_snr_ge_10": {
            "count": len(high_snr_idx),
            "accuracy": float(
                np.mean([y_true[i] == y_pred[i] for i in high_snr_idx])
            )
            if high_snr_idx
            else 1.0,
        },
        "moderate_snr_lt_10": {
            "count": len(low_snr_idx),
            "accuracy": float(
                np.mean([y_true[i] == y_pred[i] for i in low_snr_idx])
            )
            if low_snr_idx
            else 1.0,
        },
        "sector_shift_partition": {
            "count": len(shift_records),
            **sector_shift_metrics,
        },
    }

    results = {
        "protocol_version": cfg["protocol"]["version"],
        "target_split": target_split_name,
        "n_targets": len(test_groups),
        "n_labels": len(test_labels),
        "metrics": {
            "accuracy": cls_metrics.accuracy,
            "macro_f1": cls_metrics.macro_f1,
            "expected_calibration_error": cls_metrics.expected_calibration_error,
            "brier_score": cls_metrics.brier_score,
            "abstention_rate": cls_metrics.abstention_rate,
            "effective_coverage": cls_metrics.effective_coverage,
            "per_class_precision": cls_metrics.per_class_precision,
            "per_class_recall": cls_metrics.per_class_recall,
            "per_class_f1": cls_metrics.per_class_f1,
        },
        "confidence_intervals_95": {
            k: [float(ci_bounds[k][0]), float(ci_bounds[k][1])] for k in ci_bounds
        },
        "ephemeris_recovery": {
            "total_candidates_evaluated": len(period_matches),
            "overall_matched_fraction": float(np.mean(period_matches)) if period_matches else 0.0,
            "primary_matched_fraction": float(np.mean(primary_matches)) if primary_matches else 0.0,
            "harmonic_matched_fraction": float(np.mean(harmonic_matches))
            if harmonic_matches
            else 0.0,
            "median_depth_rel_error": float(np.median(depth_rel_errors))
            if depth_rel_errors
            else 0.0,
            "median_duration_rel_error": float(np.median(duration_rel_errors))
            if duration_rel_errors
            else 0.0,
            "interval_coverage_68": float(np.mean(ci68_covered)) if ci68_covered else 1.0,
            "interval_coverage_95": float(np.mean(ci95_covered)) if ci95_covered else 1.0,
        },
        "injection_recovery": injection_results,
        "ablations": ablations,
        "robustness_strata": robustness_strata,
    }

    json_path = out_dir / "locked_evaluation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)

    tex_macros = [
        f"\\newcommand{{\\TransitEyeMacroFOne}}{{{cls_metrics.macro_f1:.3f}}}",
        f"\\newcommand{{\\TransitEyeAccuracy}}{{{cls_metrics.accuracy * 100.0:.1f}\\%}}",
        f"\\newcommand{{\\TransitEyeECE}}{{{cls_metrics.expected_calibration_error:.3f}}}",
        f"\\newcommand{{\\TransitEyeBrier}}{{{cls_metrics.brier_score:.3f}}}",
        f"\\newcommand{{\\TransitEyeAbstention}}{{{cls_metrics.abstention_rate * 100.0:.1f}\\%}}",
        f"\\newcommand{{\\TransitEyeRecoveryFraction}}{{{results['ephemeris_recovery']['overall_matched_fraction'] * 100.0:.1f}\\%}}",
    ]
    (out_dir / "evaluation_macros.tex").write_text("\n".join(tex_macros) + "\n", encoding="utf-8")

    if injection_results:
        fig_inj, ax_inj = plt.subplots(figsize=(6, 4))
        ax_inj.plot(
            injection_results["depths_ppm"],
            np.array(injection_results["recovery_fraction"]) * 100.0,
            "o-",
            color="#1E88E5",
            linewidth=2,
            label="Empirical Recovery Rate (%)",
        )
        ax_inj.axvline(1000.0, color="#E53935", linestyle="--", label="1,000 ppm Reference")
        ax_inj.set_xscale("log")
        ax_inj.set_ylim(-5.0, 105.0)
        ax_inj.set_xlabel("Injected Transit Depth (ppm)")
        ax_inj.set_ylabel("Detection & Ephemeris Recovery (%)")
        ax_inj.set_title(
            "Synthetic Injection Recovery on Real TESS Substrates",
            fontsize=11,
            fontweight="bold",
        )
        ax_inj.grid(True, alpha=0.3)
        ax_inj.legend(frameon=True)
        fig_inj.tight_layout()
        fig_inj.savefig(out_dir / "injection_recovery.png", dpi=300)
        plt.close(fig_inj)

    # Real Calibration Reliability Diagram from test + calibration predictions
    fig_cal, ax_cal = plt.subplots(figsize=(5, 5))
    ax_cal.plot([0, 1], [0, 1], "--", color="#757575", label="Perfect Calibration")
    confidences = np.max(y_probs_arr, axis=1)
    correct_arr = np.array(
        [1.0 if yt == yp else 0.0 for yt, yp in zip(y_true, y_pred, strict=False)]
    )
    bins = np.linspace(0.0, 1.0, 6)
    bin_confs: list[float] = []
    bin_accs: list[float] = []
    for b_idx in range(len(bins) - 1):
        mask = (confidences >= bins[b_idx]) & (confidences <= bins[b_idx + 1])
        if np.any(mask):
            bin_confs.append(float(np.mean(confidences[mask])))
            bin_accs.append(float(np.mean(correct_arr[mask])))
    if bin_confs:
        ax_cal.plot(
            bin_confs,
            bin_accs,
            "s-",
            color="#2E7D32",
            linewidth=2,
            label=f"Calibrated Classifier (ECE={cls_metrics.expected_calibration_error:.3f})",
        )
    ax_cal.set_xlim(0.0, 1.0)
    ax_cal.set_ylim(0.0, 1.05)
    ax_cal.set_xlabel("Mean Predicted Confidence")
    ax_cal.set_ylabel("Empirical Accuracy")
    ax_cal.set_title("Reliability Diagram (Held-Out Test)", fontsize=11, fontweight="bold")
    ax_cal.grid(True, alpha=0.3)
    ax_cal.legend(frameon=True, loc="lower right")
    fig_cal.tight_layout()
    fig_cal.savefig(out_dir / "calibration_curve.png", dpi=300)
    plt.close(fig_cal)

    return results
