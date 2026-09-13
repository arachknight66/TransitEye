"""Offline B035--B039 feature-contract tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from transiteye.config import FeatureSettings, load_config
from transiteye.features.bls_features import extract_bls_features
from transiteye.features.builder import (
    BuiltFeatureMatrix,
    FeatureIdentityInputs,
    build_feature_matrix,
    freeze_feature_matrix,
    make_feature_version,
)
from transiteye.features.frequency_domain import (
    extract_fft_features,
    extract_lomb_scargle_features,
    fft_spectrum,
    lomb_scargle_spectrum,
)
from transiteye.features.registry import (
    FEATURE_REGISTRY,
    model_feature_names,
    validate_registry,
)
from transiteye.features.schemas import ColumnRole, FeatureDefinition
from transiteye.features.time_domain import extract_time_domain_features
from transiteye.features.validation import validate_feature_matrix

ROOT = Path(__file__).resolve().parents[1]
DEMO_VERSION = "dataset-demo-a74b6aad7c21faf15b0d"
SCIENTIFIC_VERSION = "dataset-4b84e8acaa6f6b334c2f"


@pytest.fixture
def settings() -> FeatureSettings:
    value = load_config(ROOT / "configs/features/mvp.yaml").features
    assert value is not None
    return value


@pytest.fixture(scope="module")
def real_feature_builds() -> tuple[BuiltFeatureMatrix, BuiltFeatureMatrix]:
    """Offline integration fixture over the accepted frozen datasets."""
    return (
        build_feature_matrix(ROOT, DEMO_VERSION),
        build_feature_matrix(ROOT, SCIENTIFIC_VERSION),
    )


def _cadences(*, irregular: bool = False, constant: bool = False) -> pd.DataFrame:
    time = np.arange(0.013, 12.013, 0.02)
    if irregular:
        time = np.delete(time, np.arange(25, len(time), 37))
    flux = np.ones_like(time) if constant else 1 + 0.002 * np.sin(2 * np.pi * time / 2.5)
    phase = (time - 0.317 + 1.25) % 2.5 - 1.25
    flux[np.abs(phase) <= 0.065] -= 0.01
    return pd.DataFrame(
        {
            "time": time,
            "detrended_flux": flux,
            "valid": True,
            "segment_id": np.where(time < 6, 0, 1),
        }
    )


def _candidate(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "candidate_id": "cand-test",
        "candidate_period": 2.5,
        "candidate_duration": 0.13,
        "candidate_epoch": 0.317,
        "candidate_depth": 0.01,
        "candidate_rank": 1,
        "bls_power": 0.8,
        "relation_to_stronger": None,
        "detection_harmonic_ratio": None,
    }
    value.update(updates)
    return value


def _periodogram() -> pd.DataFrame:
    period = np.linspace(0.5, 6, 500)
    power = 0.1 + np.exp(-(((period - 2.5) / 0.04) ** 2)) * 0.7
    return pd.DataFrame({"period": period, "power": power})


def _registry_frame() -> pd.DataFrame:
    ordered = [
        entry.name
        for entry in FEATURE_REGISTRY
        if entry.role in {ColumnRole.IDENTITY, ColumnRole.PROVENANCE, ColumnRole.FEATURE}
        and entry.group != "label_provenance"
    ]
    row: dict[str, object] = {column: 1.0 for column in ordered}
    row.update(
        {
            "candidate_id": "cand-a",
            "object_id": "tic-1",
            "observation_group_id": "og-a",
            "dataset_version": "dataset-a",
            "feature_version": "features-a",
            "preprocessing_config_hash": "a" * 20,
            "bls_config_hash": "b" * 20,
            "bls_candidate_period": 2.0,
            "bls_candidate_duration": 0.1,
            "bls_duty_cycle": 0.05,
        }
    )
    return pd.DataFrame([row], columns=ordered)


def test_registry_is_ordered_and_truth_blind() -> None:
    validate_registry()
    names = model_feature_names()
    assert names == model_feature_names()
    assert len(names) == len(set(names))
    assert "catalog_period" not in names
    assert "injected_period_days" not in names
    assert "gold_candidate_label" not in names


@pytest.mark.parametrize(
    "forbidden",
    ["catalog_period", "injected_depth", "gold_label", "match_type", "dataset_role", "split"],
)
def test_adversarial_forbidden_feature_registration(forbidden: str) -> None:
    registry = FEATURE_REGISTRY + (
        FeatureDefinition(forbidden, ColumnRole.FEATURE, "adversarial", "must fail"),
    )
    with pytest.raises(ValueError, match="cannot be model inputs"):
        validate_registry(registry)


def test_time_domain_features_are_deterministic_and_candidate_centered(
    settings: FeatureSettings,
) -> None:
    cadences = _cadences()
    first = extract_time_domain_features(_candidate(), cadences, settings)
    second = extract_time_domain_features(_candidate(), cadences, settings)
    assert first == second
    assert first["td_observed_depth"] == pytest.approx(0.01, abs=0.002)
    assert first["td_in_transit_count"] > 0
    assert first["td_observed_transit_windows"] >= 4
    assert all(not np.isinf(value) for value in first.values())


def test_non_grid_aligned_epoch_and_gap_remain_supported(settings: FeatureSettings) -> None:
    cadences = _cadences(irregular=True)
    cadences.loc[(cadences["time"] > 4) & (cadences["time"] < 4.8), "valid"] = False
    result = extract_time_domain_features(_candidate(candidate_epoch=0.3237), cadences, settings)
    assert result["td_in_transit_count"] > 0
    assert np.isfinite(result["td_observed_depth"])


def test_too_few_transits_leave_odd_even_missing(settings: FeatureSettings) -> None:
    cadences = _cadences().loc[lambda frame: frame["time"] < 1.0]
    result = extract_time_domain_features(_candidate(), cadences, settings)
    assert np.isnan(result["td_odd_even_depth_difference"])
    assert np.isnan(result["td_odd_even_depth_normalized_difference"])


def test_constant_curve_has_missing_undefined_moments_not_infinity(
    settings: FeatureSettings,
) -> None:
    cadences = _cadences(constant=True)
    cadences["detrended_flux"] = 1.0
    result = extract_time_domain_features(_candidate(), cadences, settings)
    assert np.isnan(result["td_global_skewness"])
    assert np.isnan(result["td_global_excess_kurtosis"])
    assert not any(np.isinf(value) for value in result.values())


def test_bls_features_use_detector_metadata(settings: FeatureSettings) -> None:
    group = pd.DataFrame(
        [
            _candidate(),
            _candidate(candidate_id="cand-2", candidate_rank=2, bls_power=0.4),
        ]
    )
    candidate = _candidate(relation_to_stronger="harmonic", detection_harmonic_ratio=0.5)
    result = extract_bls_features(candidate, _cadences(), _periodogram(), group, settings)
    assert result["bls_duty_cycle"] == pytest.approx(0.13 / 2.5)
    assert result["bls_period_baseline_ratio"] > 0
    assert result["bls_candidate_rank"] == 1
    assert result["bls_has_harmonic_relation"] == 1
    assert result["bls_detector_harmonic_ratio"] == 0.5


def test_lomb_scargle_handles_irregular_time_and_finds_dominant_signal(
    settings: FeatureSettings,
) -> None:
    frequency, power = lomb_scargle_spectrum(_cadences(irregular=True), settings)
    result = extract_lomb_scargle_features(frequency, power, 2.5, settings)
    assert result["ls_dominant_frequency_per_day"] > 0
    assert result["ls_power_at_bls_frequency"] >= 0
    assert 0 <= result["ls_spectral_entropy"] <= 1
    np.testing.assert_array_equal(
        frequency, lomb_scargle_spectrum(_cadences(irregular=True), settings)[0]
    )


def test_lomb_scargle_constant_signal_yields_missing_features(settings: FeatureSettings) -> None:
    cadences = _cadences(constant=True)
    cadences["detrended_flux"] = 1.0
    frequency, power = lomb_scargle_spectrum(cadences, settings)
    result = extract_lomb_scargle_features(frequency, power, 2.5, settings)
    assert frequency.size == 0
    assert all(np.isnan(value) for value in result.values())


def test_fft_regular_and_short_gap_are_deterministic(settings: FeatureSettings) -> None:
    cadences = _cadences()
    cadences = cadences.drop(index=[20]).reset_index(drop=True)
    first_frequency, first_power = fft_spectrum(cadences, settings)
    second_frequency, second_power = fft_spectrum(cadences, settings)
    np.testing.assert_array_equal(first_frequency, second_frequency)
    np.testing.assert_array_equal(first_power, second_power)
    result = extract_fft_features(first_frequency, first_power, 2.5, settings)
    assert result["fft_usable"] == 1
    assert result["fft_dominant_frequency_per_day"] > 0
    assert 0 <= result["fft_spectral_entropy"] <= 1


def test_fft_never_bridges_large_gap_and_unusable_is_missing(settings: FeatureSettings) -> None:
    cadences = _cadences().loc[lambda frame: (frame["time"] < 1) | (frame["time"] > 11)]
    frequency, power = fft_spectrum(cadences, settings)
    result = extract_fft_features(frequency, power, 2.5, settings)
    assert result["fft_usable"] == 0
    assert np.isnan(result["fft_dominant_power"])


def test_feature_version_is_deterministic_portable_and_input_sensitive() -> None:
    identity = FeatureIdentityInputs(
        dataset_version="dataset-a",
        feature_config_hash="a" * 20,
        registry_hash="b" * 20,
        preprocessing_config_hashes=("c" * 20,),
        bls_config_hashes=("d" * 20,),
        generator_version="v1",
    )
    assert make_feature_version(identity) == make_feature_version(identity)
    relocated = Path("/elsewhere") / make_feature_version(identity)
    assert relocated.name == make_feature_version(identity)
    changed = FeatureIdentityInputs(**{**identity.__dict__, "dataset_version": "dataset-b"})
    assert make_feature_version(changed) != make_feature_version(identity)


def test_validation_rejects_infinity_and_preserves_registry_order() -> None:
    valid = _registry_frame()
    assert validate_feature_matrix(valid)["candidate_rows"] == 1
    invalid = valid.copy()
    invalid.loc[0, model_feature_names()[0]] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        validate_feature_matrix(invalid)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("bls_candidate_period", 0.0, "periods"),
        ("bls_candidate_duration", 3.0, "shorter"),
        ("bls_duty_cycle", 0.0, "duty"),
        ("bls_estimated_transit_count", -1.0, "counts"),
        ("ls_dominant_frequency_per_day", -1.0, "positive"),
        ("fft_dominant_frequency_per_day", -1.0, "positive"),
        ("fft_usable", 0.5, "binary"),
        ("ls_spectral_entropy", 1.1, "range"),
    ],
)
def test_validation_rejects_invalid_scientific_feature_values(
    column: str, value: float, message: str
) -> None:
    frame = _registry_frame()
    frame.loc[0, column] = value
    with pytest.raises(ValueError, match=message):
        validate_feature_matrix(frame)


def test_validation_rejects_duplicate_ids_and_schema_drift() -> None:
    frame = _registry_frame()
    with pytest.raises(ValueError, match="duplicate"):
        validate_feature_matrix(pd.concat([frame, frame], ignore_index=True))
    with pytest.raises(ValueError, match="columns/order"):
        validate_feature_matrix(frame[frame.columns[::-1]])


def test_demo_and_scientific_matrices_use_one_schema_and_preserve_all_rows(
    real_feature_builds: tuple[BuiltFeatureMatrix, BuiltFeatureMatrix],
) -> None:
    demo, scientific = real_feature_builds
    assert len(demo.features) == 375
    assert len(scientific.features) == 75
    assert demo.features.columns.tolist() == scientific.features.columns.tolist()
    assert demo.features["candidate_id"].is_unique
    assert scientific.features["candidate_id"].is_unique
    assert not np.isinf(demo.features[list(model_feature_names())].to_numpy()).any()
    assert not np.isinf(scientific.features[list(model_feature_names())].to_numpy()).any()
    assert demo.validation["all_null_feature_count"] == 0
    assert scientific.validation["all_null_feature_count"] == 0


def test_real_feature_build_and_freeze_replay_are_deterministic(
    tmp_path: Path,
    real_feature_builds: tuple[BuiltFeatureMatrix, BuiltFeatureMatrix],
) -> None:
    _, scientific = real_feature_builds
    replay = build_feature_matrix(ROOT, SCIENTIFIC_VERSION)
    assert replay.feature_version == scientific.feature_version
    pd.testing.assert_frame_equal(replay.features, scientific.features)
    first = freeze_feature_matrix(scientific, tmp_path)
    second = freeze_feature_matrix(replay, tmp_path)
    assert first == second
    assert (first / "feature_registry.json").is_file()
    assert (first / "feature_validation.json").is_file()
