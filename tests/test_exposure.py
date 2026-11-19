"""Per-product exposure handling must not confuse cadence and integration time."""

import numpy as np
import pytest

from transiteye.domain.models import LightCurve, Provenance
from transiteye.io.exposure import resolve_exposure


def _curve(headers: dict[str, str]) -> LightCurve:
    return LightCurve(
        time_days=np.array([1.0, 2.0]), flux=np.array([100.0, 100.0]),
        flux_error=None, quality=None, cadence_number=None,
        centroid_column=None, centroid_row=None, flux_kind="SAP_FLUX", flux_unit="e-/s",
        provenance=Provenance(
            source_uri="mast:test", source_path=None, checksum_sha256="a" * 64,
            product_type="kepler_light_curve", target_id="42", sector=None,
            time_system="TDB", time_reference="2454833", metadata={"mission": "Kepler", **headers},
        ),
        metadata={"time_unit": "d"},
    )


def test_frame_integration_sum_is_exposure_not_timedel() -> None:
    result = resolve_exposure(_curve({
        "INT_TIME": "6.01980289", "NUM_FRM": "270", "TIMEDEL": "0.0003472222222222222",
        "EXPOSURE": "82.7273",
    }))
    assert result.days == pytest.approx(6.01980289 * 270 / 86400)
    assert result.source == "INT_TIME*NUM_FRM"
    assert result.status == "frame_sum_boxcar_approximation"


def test_missing_exposure_is_reported_and_override_is_explicit() -> None:
    curve = _curve({"TIMEDEL": "0.0204", "EXPOSURE": "82.7"})
    missing = resolve_exposure(curve)
    assert missing.days is None
    assert missing.status == "missing"
    override = resolve_exposure(curve, override_seconds=120.0)
    assert override.days == pytest.approx(120.0 / 86400)
    assert override.status == "user_supplied"


@pytest.mark.parametrize("headers", [
    {"INT_TIME": "0", "NUM_FRM": "60"},
    {"INT_TIME": "1.98", "NUM_FRM": "-1"},
    {"INT_TIME": "1.98", "NUM_FRM": "1.5"},
    {"INT_TIME": "nan", "NUM_FRM": "60"},
])
def test_invalid_exposure_keywords_fail_explicitly(headers: dict[str, str]) -> None:
    with pytest.raises(ValueError, match="INT_TIME|NUM_FRM"):
        resolve_exposure(_curve(headers))


def test_invalid_override_is_rejected() -> None:
    curve = _curve({"INT_TIME": "1.98", "NUM_FRM": "60"})
    with pytest.raises(ValueError, match="override_seconds"):
        resolve_exposure(curve, override_seconds=-1)
