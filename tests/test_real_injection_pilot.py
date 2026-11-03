"""Source isolation and provenance for the TRAIN-only injection pilot."""

import json
from pathlib import Path

import numpy as np
import pytest

from transiteye.datasets.splits import DatasetSplit
from transiteye.domain.models import LightCurve, Provenance
from transiteye.evaluation.real_injection_pilot import (
    _assumed_settings,
    make_injected_curve,
    run,
    select_sources,
)


def test_source_selection_only_uses_train_groups_and_latest_download(tmp_path: Path) -> None:
    root = tmp_path / "data"
    root.mkdir()
    for name in ("old.fits", "new.fits", "test.fits"):
        (root / name).write_bytes(b"placeholder")
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"assignments": [
        {"source_group_id": "tic-1", "split": "test"},
        {"source_group_id": "tic-2", "split": "train"},
    ]}))
    products = tmp_path / "products.json"
    products.write_text(json.dumps({"entries": [
        {"target_id": "1", "local_filename": "test.fits", "sector": 100, "product_id": "test"},
        {"target_id": "2", "local_filename": "old.fits", "sector": 1, "product_id": "old"},
        {"target_id": "2", "local_filename": "new.fits", "sector": 2, "product_id": "new"},
    ]}))
    config = tmp_path / "pilot.toml"
    config.write_text(f'''split_manifest = "{splits}"
product_manifest = "{products}"
data_directory = "{root}"
source_count = 1
source_selection = "fixed training selection"
''')
    selected = select_sources(config)["sources"]
    assert len(selected) == 1
    assert selected[0]["source_group_id"] == "tic-2"
    assert selected[0]["product"]["local_filename"] == "new.fits"


def test_injection_preserves_raw_input_masks_units_and_parent_provenance() -> None:
    times = np.linspace(100.0, 105.0, 1000)
    quality = np.arange(len(times), dtype=np.int64) % 2
    parent = LightCurve(
        time_days=times, flux=np.full(len(times), 1000.0), flux_error=np.ones(len(times)),
        quality=quality, cadence_number=np.arange(len(times)), centroid_column=None, centroid_row=None,
        flux_kind="PDCSAP", flux_unit="electron/s",
        provenance=Provenance(source_uri="file:///parent.fits", source_path="/parent.fits",
                              checksum_sha256="a" * 64, product_type="tess_light_curve",
                              target_id="2", sector=1, time_system="TDB", time_reference="2457000"),
    )
    config = {"duration_hours": 2.5, "ingress_fraction": 0.2, "exposure_minutes_assumed": 2.0,
              "injection_integration_points": 101}
    trial, epoch = make_injected_curve(parent, "tic-2", DatasetSplit.TRAIN, "trial-1", 123, config, 2.0, 2000.0)
    assert epoch is not None
    assert np.any(trial.flux < parent.flux)
    np.testing.assert_array_equal(parent.flux, np.full(len(times), 1000.0))
    np.testing.assert_array_equal(trial.quality, quality)
    np.testing.assert_array_equal(trial.time_days, times)
    np.testing.assert_array_equal(trial.flux_error, parent.flux_error)
    assert trial.flux_unit == parent.flux_unit
    assert trial.provenance.time_system == "TDB"
    assert trial.provenance.metadata["parent_input_sha256"] == "a" * 64
    assert trial.provenance.source_path is None
    assert trial.provenance.metadata["parent_source_path"] == "/parent.fits"
    assert trial.provenance.metadata["parent_source_group_id"] == "tic-2"
    assert trial.provenance.metadata["parent_split"] == "train"
    control, _ = make_injected_curve(parent, "tic-2", DatasetSplit.TRAIN, "control", 123, config, None, None)
    np.testing.assert_array_equal(control.flux, parent.flux)
    with pytest.raises(ValueError, match="restricted to TRAIN"):
        make_injected_curve(parent, "tic-2", DatasetSplit.TEST, "trial-2", 123, config, 2.0, 2000.0)


def test_bad_source_checksum_stops_and_records_preflight_failure(tmp_path: Path) -> None:
    root = tmp_path / "data"
    root.mkdir()
    (root / "source.fits").write_bytes(b"not the manifest data")
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"assignments": [{"source_group_id": "tic-2", "split": "train"}]}))
    products = tmp_path / "products.json"
    products.write_text(json.dumps({"entries": [{
        "target_id": "2", "local_filename": "source.fits", "sector": 1,
        "product_id": "source", "expected_sha256": "0" * 64,
    }]}))
    config = tmp_path / "pilot.toml"
    config.write_text(f'''schema_version = "1"
split_manifest = "{splits}"
product_manifest = "{products}"
data_directory = "{root}"
source_count = 1
source_selection = "fixed training selection"
exposure_minutes_assumed = 2.0
[analysis]
max_candidates = 1
snr_threshold = 6.0
screen_snr_threshold = 5.0
device = "cpu"
''')
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="failed preflight"):
        run(config, output)
    audit = json.loads((output / "input_audit.json").read_text())
    assert audit["failures"] == 1
    assert "checksum mismatch" in audit["sources"][0]["error"]


def test_assumed_exposure_is_not_reported_as_user_supplied() -> None:
    config = {"exposure_minutes_assumed": 2.0, "analysis": {
        "max_candidates": 1, "snr_threshold": 6, "screen_snr_threshold": 5, "device": "cpu",
    }}
    settings = _assumed_settings(config)
    assert settings.exposure_status == "fixed_assumption_unverified"
    assert settings.trapezoid_exposure_days == 2 / 1440
    config["exposure_minutes_assumed"] = 10.0
    with pytest.raises(ValueError, match="unverified two-minute fallback"):
        _assumed_settings(config)
