"""Exposure-aware joint trapezoid fit for independent observations of one source.

Inputs must use one common relative-day coordinate and contain disjoint
exposure intervals. The caller must retain the external BJD origin needed to
interpret epochs. This module cannot verify time standards or origins and does
not infer them, rescale flux, or estimate uncertainties.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from transiteye.characterization.fitting import evaluate_trapezoid


@dataclass(frozen=True, slots=True)
class JointFitObservation:
    """One normalized light curve, with its measured exposure in days."""

    observation_id: str
    time_days: NDArray[np.float64]
    flux: NDArray[np.float64]
    exposure_days: float
    flux_err: NDArray[np.float64] | None = None


@dataclass(frozen=True, slots=True)
class ObservationFit:
    """Depth is unavailable when the event cannot be measured in this curve."""

    observation_id: str
    status: Literal["fitted", "no_event", "insufficient_data"]
    depth: float | None
    baseline: float | None


@dataclass(frozen=True, slots=True)
class JointFitResult:
    """Shared event shape and independent per-observation photometry."""

    period_days: float
    epoch_days: float
    duration_days: float
    ingress_fraction: float
    observations: tuple[ObservationFit, ...]


def _validate(observations: tuple[JointFitObservation, ...]) -> None:
    if len(observations) < 2:
        raise ValueError("joint fit requires at least two observations")
    ids: set[str] = set()
    intervals: list[tuple[float, float]] = []
    for obs in observations:
        if not isinstance(obs.observation_id, str) or not obs.observation_id.strip():
            raise ValueError("observation_id must be a nonempty stable string")
        if obs.observation_id in ids:
            raise ValueError(f"duplicate observation_id: {obs.observation_id}")
        ids.add(obs.observation_id)
        time = np.asarray(obs.time_days)
        flux = np.asarray(obs.flux)
        if time.ndim != 1 or flux.ndim != 1 or len(time) == 0 or time.shape != flux.shape:
            raise ValueError(f"{obs.observation_id}: time and flux must be nonempty 1D arrays of equal length")
        if not np.all(np.isfinite(time)) or not np.all(np.isfinite(flux)):
            raise ValueError(f"{obs.observation_id}: time and flux must be finite")
        if not 0.95 <= float(np.median(flux)) <= 1.05:
            raise ValueError(f"{obs.observation_id}: flux must be normalized near unity")
        if not np.all(np.diff(time) > 0):
            raise ValueError(f"{obs.observation_id}: time must be strictly increasing")
        if not np.isfinite(obs.exposure_days) or obs.exposure_days <= 0:
            raise ValueError(f"{obs.observation_id}: exposure_days must be finite and positive")
        if obs.exposure_days >= float(time[-1] - time[0]) and len(time) > 1:
            raise ValueError(f"{obs.observation_id}: exposure exceeds observation span")
        if len(time) > 1 and np.any(np.diff(time) < obs.exposure_days):
            raise ValueError(f"{obs.observation_id}: exposure intervals overlap")
        if obs.flux_err is not None:
            error = np.asarray(obs.flux_err)
            if error.shape != flux.shape or not np.all(np.isfinite(error)) or np.any(error <= 0):
                raise ValueError(f"{obs.observation_id}: flux_err must match flux and be finite and positive")
        intervals.extend((float(t - obs.exposure_days / 2), float(t + obs.exposure_days / 2)) for t in time)
    if len({obs.flux_err is None for obs in observations}) != 1:
        raise ValueError("joint fit requires flux_err for all observations or none")
    intervals.sort()
    if any(next_start < end for (_, end), (next_start, _) in zip(intervals, intervals[1:], strict=False)):
        raise ValueError("exposure intervals overlap across observations")


def _coverage(obs: JointFitObservation, epoch: float, period: float, duration: float) -> str:
    dip = 1.0 - evaluate_trapezoid(
        np.asarray(obs.time_days), epoch, period, duration, 1.0, 0.2,
        exposure_days=obs.exposure_days,
    )
    event = dip > 1e-10
    if not np.any(event):
        return "no_event"
    if np.count_nonzero(event) < 3 or np.count_nonzero(~event) < 3:
        return "insufficient_data"
    return "fitted"


def fit_joint_trapezoid(
    observations: tuple[JointFitObservation, ...],
    init_epoch_days: float,
    init_period_days: float,
    init_duration_days: float,
) -> JointFitResult:
    """Fit one event shape with independent baseline and depth per observation.

    `flux_err=None` for every observation uses unweighted flux residuals; it
    does not invent measured uncertainties. Mixing measured and unmeasured
    errors is rejected because it gives arbitrary relative weights. The initial
    ephemeris selects observations with enough event
    and out-of-event samples. A failed optimizer raises RuntimeError.
    """
    _validate(observations)
    if not all(np.isfinite(x) for x in (init_epoch_days, init_period_days, init_duration_days)):
        raise ValueError("initial ephemeris must be finite")
    if init_period_days <= 0 or not 0 < init_duration_days < init_period_days:
        raise ValueError("initial period must be positive and duration inside one period")
    if any(obs.exposure_days >= init_period_days for obs in observations):
        raise ValueError("exposure_days must be shorter than the initial period")
    statuses = tuple(
        _coverage(obs, init_epoch_days, init_period_days, init_duration_days)
        for obs in observations
    )
    eligible = tuple(obs for obs, status in zip(observations, statuses, strict=True) if status == "fitted")
    if not eligible:
        raise ValueError("joint fit has no observation with measurable event coverage")
    event_cycles: set[int] = set()
    for obs in eligible:
        cycles = np.rint((np.asarray(obs.time_days) - init_epoch_days) / init_period_days)
        near_event = np.abs(
            np.asarray(obs.time_days) - (init_epoch_days + cycles * init_period_days)
        ) <= 0.5 * (init_duration_days + obs.exposure_days)
        event_cycles.update(int(cycle) for cycle in cycles[near_event])
    if len(event_cycles) < 2:
        raise ValueError("joint period fit requires coverage of at least two event cycles")

    # Keep the search local to the supplied ephemeris. Each observation gets
    # independent nuisance parameters; there is no cross-observation stitching.
    lower = [init_epoch_days - 0.2 * init_period_days, 0.8 * init_period_days,
             max(1e-5, 0.2 * init_duration_days), 0.01]
    upper = [init_epoch_days + 0.2 * init_period_days, 1.2 * init_period_days,
             min(0.8 * init_period_days, 3.0 * init_duration_days), 0.99]
    if lower[2] >= upper[2]:
        raise ValueError("initial duration leaves no feasible optimization interval")
    x0 = [init_epoch_days, init_period_days, init_duration_days, 0.2]
    for obs in eligible:
        baseline = float(np.median(obs.flux))
        x0.extend((baseline, min(0.999, max(1e-5, baseline - float(np.min(obs.flux))))))
        lower.extend((0.9, 0.0))
        upper.extend((1.1, 1.0))

    def residual(p: NDArray[np.float64]) -> NDArray[np.float64]:
        parts = []
        for index, obs in enumerate(eligible):
            model = evaluate_trapezoid(
                np.asarray(obs.time_days), p[0], p[1], p[2], p[5 + 2 * index], p[3],
                baseline=p[4 + 2 * index], exposure_days=obs.exposure_days,
            )
            difference = np.asarray(obs.flux) - model
            if obs.flux_err is not None:
                difference = difference / np.asarray(obs.flux_err)
            parts.append(difference)
        return np.concatenate(parts)

    result = least_squares(residual, x0, bounds=(lower, upper), max_nfev=500)
    if not result.success or not np.all(np.isfinite(result.x)):
        raise RuntimeError(f"joint trapezoid optimizer did not converge: {result.message}")
    fitted = iter((float(result.x[5 + 2 * i]), float(result.x[4 + 2 * i])) for i in range(len(eligible)))
    per_observation = []
    for obs, initial_status in zip(observations, statuses, strict=True):
        if initial_status == "fitted":
            depth, baseline = next(fitted)
            final_status = _coverage(obs, float(result.x[0]), float(result.x[1]), float(result.x[2]))
            if final_status != "fitted":
                per_observation.append(ObservationFit(obs.observation_id, final_status, None, None))
            else:
                per_observation.append(ObservationFit(obs.observation_id, "fitted", depth, baseline))
        else:
            per_observation.append(ObservationFit(obs.observation_id, initial_status, None, None))
    return JointFitResult(
        period_days=float(result.x[1]), epoch_days=float(result.x[0]),
        duration_days=float(result.x[2]), ingress_fraction=float(result.x[3]),
        observations=tuple(per_observation),
    )
