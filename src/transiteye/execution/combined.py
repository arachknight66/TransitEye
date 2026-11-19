"""One bounded preliminary search and exposure-aware fit across an observation set.

The BLS SNR is a screening statistic, not empirical search significance or a
calibrated probability. This module makes no classification claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from transiteye.characterization.joint_fit import JointFitObservation, fit_joint_trapezoid
from transiteye.detection.bls import BlsSearchBudgetExceeded, run_bls
from transiteye.io.exposure import resolve_exposure
from transiteye.io.observation_set import ObservationSet
from transiteye.preprocessing.combined import (
    IncompleteObservationSetError,
    prepare_observation_set,
)

CombinedStatus = Literal[
    "insufficient_data", "preparation_failed", "incomplete_errors",
    "budget_exceeded", "no_significant_signal", "exposure_missing",
    "joint_fit_failed", "fitted",
]


@dataclass(frozen=True, slots=True)
class CombinedSearchSettings:
    """Explicit BLS bounds and a heuristic SNR screen; no significance test."""

    min_period_days: float
    max_period_days: float
    max_trial_periods: int
    min_bls_snr: float = 6.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.min_period_days) or self.min_period_days <= 0:
            raise ValueError("min_period_days must be finite and positive")
        if not np.isfinite(self.max_period_days) or self.max_period_days <= self.min_period_days:
            raise ValueError("max_period_days must be finite and exceed min_period_days")
        if (
            isinstance(self.max_trial_periods, bool)
            or not isinstance(self.max_trial_periods, (int, np.integer))
            or self.max_trial_periods < 1
        ):
            raise ValueError("max_trial_periods must be a positive integer")
        if not np.isfinite(self.min_bls_snr) or self.min_bls_snr <= 0:
            raise ValueError("min_bls_snr must be finite and positive")


@dataclass(frozen=True, slots=True)
class PreliminarySignal:
    period_days: float
    epoch_bjd_tdb: float
    duration_days: float
    detection_snr: float


@dataclass(frozen=True, slots=True)
class CombinedObservationResult:
    input_checksum: str
    status: str
    depth: float | None = None
    exposure_days: float | None = None
    exposure_status: str | None = None


@dataclass(frozen=True, slots=True)
class CombinedAnalysisResult:
    status: CombinedStatus
    detail: str
    observations: tuple[CombinedObservationResult, ...]
    preliminary: PreliminarySignal | None = None
    fitted_period_days: float | None = None
    fitted_epoch_bjd_tdb: float | None = None
    fitted_duration_days: float | None = None


def analyze_observation_set(
    observation_set: ObservationSet,
    settings: CombinedSearchSettings,
) -> CombinedAnalysisResult:
    """Search once on independently detrended flux and fit one shared event.

    Every input product has a result row. Empty products, unresolved overlap,
    and mixed measured-error availability prevent a joint scientific result.
    """
    rows = tuple(
        CombinedObservationResult(checksum, "not_processed")
        for checksum in observation_set.input_checksums
    )

    def outcome(
        status: CombinedStatus, detail: str, *,
        observations: tuple[CombinedObservationResult, ...] = rows,
        preliminary: PreliminarySignal | None = None,
    ) -> CombinedAnalysisResult:
        return CombinedAnalysisResult(status, detail, observations, preliminary)

    try:
        prepared = prepare_observation_set(observation_set)
    except IncompleteObservationSetError as error:
        return outcome("preparation_failed", str(error))
    if len(prepared.time_days) < 50 or len(prepared.observations) < 2 or any(
        record.dual is None for record in prepared.observations
    ):
        mapped = tuple(
            CombinedObservationResult(checksum, record.status)
            for checksum, record in zip(
                observation_set.input_checksums, prepared.observations, strict=True
            )
        )
        return outcome("insufficient_data", "All products need usable rows and at least 50 total cadences.", observations=mapped)

    error = prepared.flux_error_detrended
    if error is not None and (not np.all(np.isfinite(error)) or np.any(error <= 0)):
        return outcome("incomplete_errors", "Measured flux errors must be available and valid for every selected cadence.")

    try:
        bls = run_bls(
            prepared.time_days, prepared.detrended_flux, error,
            min_period=settings.min_period_days,
            max_period=settings.max_period_days,
            device="cpu",
            max_trial_periods=settings.max_trial_periods,
        )
    except BlsSearchBudgetExceeded as failure:
        return outcome("budget_exceeded", str(failure))
    except ValueError as failure:
        return outcome("insufficient_data", f"BLS search could not run: {failure}")

    preliminary = PreliminarySignal(
        bls.best_period,
        prepared.time_origin_bjd_tdb + bls.best_epoch_days,
        bls.best_duration_hours / 24.0,
        bls.best_snr,
    )
    if not np.isfinite(bls.best_snr) or bls.best_snr < settings.min_bls_snr:
        return outcome(
            "no_significant_signal",
            "No BLS peak passed the configured SNR screen; this is not an empirical significance test.",
            preliminary=preliminary,
        )

    exposure_rows = []
    invalid_exposures = []
    for checksum, record in zip(
        observation_set.input_checksums, prepared.observations, strict=True
    ):
        try:
            exposure = resolve_exposure(record.observation.source)
        except ValueError as failure:
            invalid_exposures.append(f"{checksum}: {failure}")
            exposure_rows.append(CombinedObservationResult(checksum, "not_fitted", exposure_status="invalid"))
        else:
            exposure_rows.append(CombinedObservationResult(
                checksum, "not_fitted", exposure_days=exposure.days, exposure_status=exposure.status,
            ))
    if any(row.exposure_days is None for row in exposure_rows):
        detail = "At least one product lacks usable measured exposure."
        if invalid_exposures:
            detail += " Invalid exposure metadata: " + " ".join(invalid_exposures)
        return outcome("exposure_missing", detail, observations=tuple(exposure_rows), preliminary=preliminary)

    joint_inputs = []
    for index, row in enumerate(exposure_rows):
        assert row.exposure_days is not None
        selected = prepared.observation_index == index
        joint_inputs.append(JointFitObservation(
            row.input_checksum,
            prepared.time_days[selected],
            prepared.detrended_flux[selected],
            row.exposure_days,
            error[selected] if error is not None else None,
        ))
    try:
        fitted = fit_joint_trapezoid(
            tuple(joint_inputs), bls.best_epoch_days, bls.best_period,
            bls.best_duration_hours / 24.0,
        )
    except (ValueError, RuntimeError) as failure:
        return outcome("joint_fit_failed", str(failure), observations=tuple(exposure_rows),
                       preliminary=preliminary)

    fit_rows = tuple(
        CombinedObservationResult(
            row.input_checksum, fit.status, fit.depth if fit.status == "fitted" else None,
            row.exposure_days, row.exposure_status,
        )
        for row, fit in zip(exposure_rows, fitted.observations, strict=True)
    )
    return CombinedAnalysisResult(
        "fitted", "Shared timing and per-observation depths fitted.", fit_rows, preliminary,
        fitted.period_days, prepared.time_origin_bjd_tdb + fitted.epoch_days,
        fitted.duration_days,
    )
