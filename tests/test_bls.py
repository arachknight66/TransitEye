"""Focused checks for the optional BLS trial-period budget."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.timeseries import BoxLeastSquares

from transiteye.detection.bls import BlsSearchBudgetExceeded, run_bls


def _flat_curve(baseline_days: float = 20.0) -> tuple[np.ndarray, np.ndarray]:
    times = np.linspace(0.0, baseline_days, 100)
    return times, np.ones_like(times)


def test_large_grid_budget_rejects_before_autoperiod(monkeypatch: pytest.MonkeyPatch) -> None:
    times, flux = _flat_curve(1800.0)

    def unexpected_autoperiod(*args: object, **kwargs: object) -> None:
        pytest.fail("oversized grid reached autoperiod")

    monkeypatch.setattr(BoxLeastSquares, "autoperiod", unexpected_autoperiod)
    with pytest.raises(BlsSearchBudgetExceeded, match="max_trial_periods=1000"):
        run_bls(times, flux, max_trial_periods=1000)


def test_budget_uses_exact_filtered_grid_count() -> None:
    times, flux = _flat_curve()
    settings = {"min_period": 1.0, "max_period": 2.0}
    # The unrestricted result is Astropy's own grid after the requested-bound filter.
    unrestricted = run_bls(times, flux, **settings)
    at_limit = run_bls(times, flux, max_trial_periods=len(unrestricted.periods), **settings)

    np.testing.assert_array_equal(at_limit.periods, unrestricted.periods)
    with pytest.raises(BlsSearchBudgetExceeded, match="max_trial_periods="):
        run_bls(times, flux, max_trial_periods=len(unrestricted.periods) - 1, **settings)


@pytest.mark.parametrize("budget", [0, -1, 1.5, True])
def test_budget_must_be_a_positive_integer(budget: object) -> None:
    times, flux = _flat_curve()
    with pytest.raises(ValueError, match="max_trial_periods must be a positive integer"):
        run_bls(times, flux, max_trial_periods=budget)  # type: ignore[arg-type]
