"""Independent numerical checks for a shared-shape, per-observation fit."""

import numpy as np
import pytest

import transiteye.characterization.joint_fit as joint_fit
from transiteye.characterization.fitting import evaluate_trapezoid
from transiteye.characterization.joint_fit import JointFitObservation, fit_joint_trapezoid


def _observation(
    name: str, start: float, stop: float, depth: float, exposure: float,
    *, with_error: bool = False,
) -> JointFitObservation:
    time = np.arange(start, stop, 0.012)
    flux = evaluate_trapezoid(
        time, 0.5, 2.0, 0.12, depth, 0.3, baseline=1.002,
        exposure_days=exposure,
    )
    error = np.full_like(time, 0.001) if with_error else None
    return JointFitObservation(name, time, flux, exposure, error)


def test_joint_fit_recovers_separate_depths_and_shared_shape() -> None:
    observations = (
        _observation("first", 0.0, 1.5, 0.012, 0.008, with_error=True),
        _observation("second", 4.0, 5.5, 0.025, 0.010, with_error=True),
    )
    result = fit_joint_trapezoid(observations, 0.505, 1.998, 0.115)

    # Analytical fixture: each input is the exact exposure-integrated
    # trapezoid with P=2 d, T0=0.5 d, duration=0.12 d and depths above.
    assert result.period_days == pytest.approx(2.0, abs=0.003)
    assert result.epoch_days == pytest.approx(0.5, abs=0.003)
    assert result.duration_days == pytest.approx(0.12, abs=0.005)
    assert result.ingress_fraction == pytest.approx(0.3, abs=0.08)
    assert [entry.status for entry in result.observations] == ["fitted", "fitted"]
    assert result.observations[0].depth == pytest.approx(0.012, abs=0.001)
    assert result.observations[1].depth == pytest.approx(0.025, abs=0.001)
    assert result.observations[0].baseline == pytest.approx(1.002, abs=0.001)


def test_no_event_and_insufficient_observations_have_no_depth() -> None:
    observations = (
        _observation("event", 0.0, 1.5, 0.012, 0.008),
        _observation("none", 3.0, 3.3, 0.025, 0.008),
        _observation("another_event", 4.0, 5.5, 0.025, 0.008),
        JointFitObservation(
            "sparse", np.array([6.49, 6.50]), np.array([0.99, 0.99]), 0.005,
        ),
    )
    result = fit_joint_trapezoid(observations, 0.5, 2.0, 0.12)
    assert [entry.status for entry in result.observations] == [
        "fitted", "no_event", "fitted", "insufficient_data",
    ]
    assert result.observations[1].depth is None
    assert result.observations[3].depth is None


@pytest.mark.parametrize("change, match", [
    ({"time_days": np.array([0.0, 0.0, 0.1])}, "strictly increasing"),
    ({"flux": np.array([1.0, np.nan, 1.0])}, "finite"),
    ({"flux_err": np.array([0.1, 0.0, 0.1])}, "flux_err"),
    ({"exposure_days": 0.0}, "exposure_days"),
])
def test_invalid_observation_is_rejected(change: dict, match: str) -> None:
    fields = {
        "observation_id": "bad", "time_days": np.array([0.0, 0.1, 0.2]),
        "flux": np.ones(3), "exposure_days": 0.01,
    }
    bad = JointFitObservation(**(fields | change))
    other = _observation("other", 4.0, 5.5, 0.01, 0.008)
    with pytest.raises(ValueError, match=match):
        fit_joint_trapezoid((bad, other), 0.5, 2.0, 0.12)


def test_overlapping_rows_across_observations_are_rejected() -> None:
    first = JointFitObservation("a", np.array([0.1, 0.2]), np.ones(2), 0.02)
    second = JointFitObservation("b", np.array([0.11, 0.21]), np.ones(2), 0.02)
    with pytest.raises(ValueError, match="overlap"):
        fit_joint_trapezoid((first, second), 0.5, 2.0, 0.12)


def test_single_event_cycle_cannot_determine_period() -> None:
    first = _observation("event", 0.0, 1.5, 0.01, 0.008)
    second = _observation("none", 3.0, 3.3, 0.01, 0.008)
    with pytest.raises(ValueError, match="two event cycles"):
        fit_joint_trapezoid((first, second), 0.5, 2.0, 0.12)


def test_optimizer_failure_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    class Failed:
        success = False
        message = "iteration limit reached"
        x = np.array([])

    monkeypatch.setattr(joint_fit, "least_squares", lambda *args, **kwargs: Failed())
    observations = (
        _observation("first", 0.0, 1.5, 0.012, 0.008),
        _observation("second", 4.0, 5.5, 0.025, 0.008),
    )
    with pytest.raises(RuntimeError, match="optimizer did not converge"):
        fit_joint_trapezoid(observations, 0.5, 2.0, 0.12)


def test_mixed_error_availability_is_rejected() -> None:
    observations = (
        _observation("measured", 0.0, 1.5, 0.012, 0.008, with_error=True),
        _observation("unknown", 4.0, 5.5, 0.025, 0.008),
    )
    with pytest.raises(ValueError, match="flux_err for all observations or none"):
        fit_joint_trapezoid(observations, 0.5, 2.0, 0.12)
