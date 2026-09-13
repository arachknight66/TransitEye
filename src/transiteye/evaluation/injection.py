"""B048 staged injection-recovery accounting for the frozen demo grid."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd


def injection_recovery_table(
    events: pd.DataFrame,
    candidates: pd.DataFrame,
    matches: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    development_tics: tuple[str, ...],
) -> pd.DataFrame:
    """Separate BLS recovery from fixed-model correctness among recovered events."""
    selected_events = events.loc[
        events["object_id"].astype(str).isin(development_tics)
        & (events["synthetic_event_class"] != "no_injection_control")
    ].copy()
    candidate_info = candidates[["candidate_id", "candidate_rank", "observation_group_id"]].merge(
        predictions[["candidate_id", "score", "prediction"]], on="candidate_id"
    )
    matched = matches.loc[matches["matched"].astype(bool)].merge(
        candidate_info, on="candidate_id", validate="many_to_one"
    )
    rows: list[dict[str, object]] = []
    for event in selected_events.sort_values("synthetic_event_id", kind="stable").itertuples():
        recovered = matched.loc[
            matched["synthetic_event_id"] == event.synthetic_event_id
        ].sort_values(["candidate_rank", "candidate_id"], kind="stable")
        best = recovered.iloc[0] if not recovered.empty else None
        expected = 1 if event.demo_gold_class == "positive" else 0
        rows.append(
            {
                "synthetic_event_id": event.synthetic_event_id,
                "variant_id": event.variant_id,
                "object_id": str(event.object_id),
                "variant_name": event.variant_name,
                "synthetic_event_class": event.synthetic_event_class,
                "difficulty": event.difficulty,
                "injected_period_days": event.injected_period_days,
                "injected_epoch": event.injected_epoch,
                "injected_duration_days": event.injected_duration_days,
                "injected_depth_or_amplitude": event.injected_depth_or_amplitude,
                "bls_recovered": best is not None,
                "recovery_match_type": None if best is None else str(best["match_type"]),
                "recovered_candidate_id": None if best is None else str(best["candidate_id"]),
                "recovered_candidate_rank": None if best is None else int(best["candidate_rank"]),
                "expected_class": expected,
                "classifier_score": None if best is None else float(best["score"]),
                "classifier_prediction": None if best is None else int(best["prediction"]),
                "classifier_correct_conditional": (
                    None if best is None else bool(int(best["prediction"]) == expected)
                ),
            }
        )
    return pd.DataFrame(rows)


def add_observability_diagnostics(
    table: pd.DataFrame, timestamps_by_variant: dict[str, np.ndarray]
) -> pd.DataFrame:
    """Add cadence alignment and observable-window counts without changing injections."""
    output = table.copy()
    cadence_offsets: list[float | None] = []
    phase_offsets: list[float | None] = []
    window_counts: list[int | None] = []
    cadence_days: list[float | None] = []
    for row in output.itertuples():
        times = np.asarray(timestamps_by_variant.get(str(row.variant_id), []), dtype=float)
        times = np.sort(times[np.isfinite(times)])
        if times.size < 2:
            cadence_days.append(None)
            cadence_offsets.append(None)
            phase_offsets.append(None)
            window_counts.append(None)
            continue
        deltas = np.diff(times)
        cadence = float(np.median(deltas[deltas > 0]))
        period = cast(float, row.injected_period_days)
        epoch = cast(float, row.injected_epoch)
        duration = cast(float, row.injected_duration_days)
        cadence_days.append(cadence)
        phase_residual = ((times - epoch + period / 2.0) % period) - period / 2.0
        cadence_offsets.append(float(np.min(np.abs(phase_residual)) / cadence))
        phase_offsets.append(float(((epoch - times[0]) % period) / period))
        first = int(np.ceil((times[0] - epoch) / period))
        last = int(np.floor((times[-1] - epoch) / period))
        count = 0
        for event_number in range(first, last + 1):
            center = epoch + event_number * period
            if np.any(np.abs(times - center) <= duration / 2.0):
                count += 1
        window_counts.append(count)
    output["median_cadence_days"] = cadence_days
    output["epoch_nearest_cadence_offset"] = cadence_offsets
    output["phase_offset_fraction"] = phase_offsets
    output["observable_event_windows"] = window_counts
    return output


def summarize_injection_recovery(table: pd.DataFrame) -> dict[str, object]:
    """Summarize each predeclared variant without combining pipeline stages."""
    summaries: dict[str, object] = {}
    for name, group in table.groupby("variant_name", sort=True):
        recovered = group.loc[group["bls_recovered"]]
        summaries[str(name)] = {
            "events": int(len(group)),
            "bls_recovered": int(group["bls_recovered"].sum()),
            "bls_recovery_rate": float(group["bls_recovered"].mean()),
            "classifier_evaluable_recovered": int(len(recovered)),
            "classifier_correct_recovered": int(
                sum(value is True for value in recovered["classifier_correct_conditional"])
            ),
            "classifier_accuracy_conditional_on_recovery": (
                float(recovered["classifier_correct_conditional"].mean())
                if len(recovered)
                else None
            ),
        }
    return summaries
