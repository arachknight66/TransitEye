"""Tests for controlled synthetic augmentation, partition isolation, and lineage."""

from __future__ import annotations

import numpy as np
import pytest

from transiteye.datasets.injection import (
    compute_deterministic_seed,
    inject_synthetic_signal,
    synthesize_eclipsing_binary_signal,
    synthesize_transit_signal,
    synthesize_variability_signal,
)
from transiteye.datasets.splits import DatasetSplit
from transiteye.domain.models import LightCurve, MorphologyClass, Provenance


@pytest.fixture
def substrate_light_curve() -> LightCurve:
    """Fixture providing a clean quiet substrate light curve."""
    time = np.linspace(0.0, 27.0, 2000)
    flux = np.ones_like(time)
    provenance = Provenance(
        source_uri="file:///test_substrate.fits",
        source_path="/path/test.fits",
        checksum_sha256="abcdef123456",
        product_type="tess_light_curve",
        target_id="12345678",
        sector=1,
        time_system="TDB",
        time_reference="BJD",
    )
    return LightCurve(
        time_days=time,
        flux=flux,
        flux_error=np.full_like(time, 0.0005),
        quality=np.zeros(len(time), dtype=np.int64),
        cadence_number=np.arange(len(time), dtype=np.int64),
        centroid_column=np.full_like(time, 500.0),
        centroid_row=np.full_like(time, 500.0),
        flux_kind="PDCSAP_FLUX",
        flux_unit="e-/s",
        provenance=provenance,
    )


def test_partition_isolation_strictly_enforces_train_only(substrate_light_curve: LightCurve) -> None:
    """Verify injection raises ValueError on non-train partitions."""
    for split in (
        DatasetSplit.CALIBRATION,
        DatasetSplit.SELECTION,
        DatasetSplit.TEST,
        DatasetSplit.SECTOR_SHIFT,
    ):
        with pytest.raises(ValueError, match="strictly prohibited on non-train partitions"):
            inject_synthetic_signal(
                substrate=substrate_light_curve,
                parent_source_group_id="tic-12345678",
                split=split,
                variant_idx=1,
                injection_kind="transit",
            )


def test_deterministic_seed_reproducibility() -> None:
    """Verify seed generation is deterministic and unique across variants."""
    seed1 = compute_deterministic_seed("tic-123", 1, "transit")
    seed2 = compute_deterministic_seed("tic-123", 1, "transit")
    seed3 = compute_deterministic_seed("tic-123", 2, "transit")
    seed4 = compute_deterministic_seed("tic-456", 1, "transit")

    assert seed1 == seed2
    assert seed1 != seed3
    assert seed1 != seed4


def test_inject_synthetic_transit(substrate_light_curve: LightCurve) -> None:
    """Verify transit injection creates expected signal and lineage."""
    injected = inject_synthetic_signal(
        substrate=substrate_light_curve,
        parent_source_group_id="tic-12345678",
        split=DatasetSplit.TRAIN,
        variant_idx=0,
        injection_kind="transit",
        period_days=3.0,
        epoch_days=1.0,
        duration_hours=2.5,
        depth_ppm=5000.0,
    )

    assert injected.injection.is_synthetic is True
    assert injected.injection.parent_source_group_id == "tic-12345678"
    assert injected.injection.target_class == MorphologyClass.TRANSIT_LIKE
    assert injected.injection.parameters["depth_ppm"] == 5000.0

    # Injected flux has a minimum dip of approximately depth_ppm (amplified slightly at center by limb darkening)
    min_flux = float(np.min(injected.light_curve.flux))
    assert min_flux < 1.0
    assert (1.0 - min_flux) == pytest.approx(0.005, rel=0.25)


def test_inject_synthetic_grazing_eb(substrate_light_curve: LightCurve) -> None:
    """Verify grazing EB injection creates primary and secondary eclipses."""
    injected = inject_synthetic_signal(
        substrate=substrate_light_curve,
        parent_source_group_id="tic-12345678",
        split=DatasetSplit.TRAIN,
        variant_idx=1,
        injection_kind="grazing_eb",
        period_days=4.0,
        epoch_days=1.0,
        duration_hours=4.0,
        depth_ppm=20000.0,
    )

    assert injected.injection.target_class == MorphologyClass.ECLIPSE_LIKE
    assert "secondary_depth_ratio" in injected.injection.parameters


def test_synthesize_signal_functions() -> None:
    """Verify low-level synthesizer functions produce valid signals."""
    t = np.linspace(0.0, 10.0, 1000)

    # Transit
    tr = synthesize_transit_signal(t, period_days=2.5, epoch_days=0.5, duration_hours=2.0, depth_ppm=1000.0)
    assert np.all(tr <= 1.000001)
    assert np.min(tr) < 0.9995

    # EB
    eb = synthesize_eclipsing_binary_signal(t, period_days=3.0, epoch_days=0.5, duration_hours=3.0, primary_depth_ppm=10000.0)
    assert np.all(eb <= 1.000001)
    assert np.min(eb) < 0.995

    # Variability
    var = synthesize_variability_signal(t, period_days=2.0, amplitude_ppm=5000.0)
    assert np.min(var) < 1.0 < np.max(var)
