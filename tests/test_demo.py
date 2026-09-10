"""Offline tests for the controlled real-substrate demo path."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import transiteye.demo.builder as demo_builder
from transiteye.config import DemoSettings, DemoVariantSettings
from transiteye.datasets.schemas import assert_model_input_boundary
from transiteye.demo.builder import (
    _demo_identity,
    build_demo_candidate_tables,
    build_demo_event_recovery,
    evaluate_demo_acceptance,
)
from transiteye.demo.injection import inject_variant
from transiteye.demo.validation import DemoValidationError, validate_demo_dataset
from transiteye.demo.variants import (
    build_variant_manifest,
    freeze_variant_manifest,
    load_variant_manifest,
)
from transiteye.identifiers import make_candidate_id
from transiteye.preprocessing.pipeline import LightCurveRecord


def _variant(name: str = "planet_easy", event_class: str = "planet_like") -> DemoVariantSettings:
    if event_class == "no_injection_control":
        return DemoVariantSettings(name=name, event_class=event_class, difficulty="control")
    return DemoVariantSettings(
        name=name,
        event_class=event_class,
        difficulty="easy",
        period_days=2.5,
        duration_days=0.12,
        depth_or_amplitude=0.02,
        secondary_depth=0.01 if event_class == "eclipsing_binary_like" else None,
    )


def _settings() -> DemoSettings:
    return DemoSettings(
        policy_version="demo-v1",
        generator_version="generator-v1",
        base_expansion_manifest_id="expansion-test",
        seed_component="demo",
        split_policy="all_development",
        variants=(
            _variant(),
            _variant("eb", "eclipsing_binary_like"),
            _variant("sin", "sinusoidal_variability_like"),
            _variant("control", "no_injection_control"),
        ),
    )


def test_demo_configuration_rejects_incoherent_or_duplicate_variants() -> None:
    with pytest.raises(ValueError, match="No-injection"):
        DemoVariantSettings(
            name="bad-control",
            event_class="no_injection_control",
            difficulty="control",
            period_days=2.0,
        )
    with pytest.raises(ValueError, match="require period"):
        DemoVariantSettings(name="incomplete", event_class="planet_like", difficulty="easy")
    with pytest.raises(ValueError, match="Secondary depth"):
        DemoVariantSettings(
            name="bad-secondary",
            event_class="planet_like",
            difficulty="easy",
            period_days=2.0,
            duration_days=0.1,
            depth_or_amplitude=0.01,
            secondary_depth=0.005,
        )
    with pytest.raises(ValueError, match="names must be unique"):
        DemoSettings(
            policy_version="demo-v1",
            generator_version="generator-v1",
            base_expansion_manifest_id="expansion-test",
            seed_component="demo",
            split_policy="all_development",
            variants=(_variant(), _variant()),
        )


def _base() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "object_id": "tic-1",
                "source_observation_id": "obs-1",
                "source_product_id": "mast:one",
                "source_raw_checksum": "1" * 64,
                "source_preprocessing_hash": "2" * 20,
                "source_processed_checksum": "3" * 64,
                "time_min": 100.0,
            },
            {
                "object_id": "tic-2",
                "source_observation_id": "obs-2",
                "source_product_id": "mast:two",
                "source_raw_checksum": "4" * 64,
                "source_preprocessing_hash": "2" * 20,
                "source_processed_checksum": "5" * 64,
                "time_min": 200.0,
            },
        ]
    )


def _source() -> LightCurveRecord:
    time = np.arange(100.0, 110.0, 0.02)
    return LightCurveRecord(
        pd.DataFrame(
            {
                "cadence_row": np.arange(len(time)),
                "time": time,
                "pdcsap_flux": np.full(len(time), 1000.0),
                "pdcsap_flux_err": np.full(len(time), 1.0),
                "sap_flux": np.full(len(time), 900.0),
                "sap_flux_err": np.full(len(time), 1.0),
                "quality": np.zeros(len(time), dtype=int),
            }
        ),
        {"source_raw_sha256": "1" * 64},
    )


def test_variant_manifest_is_deterministic_portable_and_truthful(tmp_path: Path) -> None:
    first = build_variant_manifest(_base(), settings=_settings(), master_seed=42)
    second = build_variant_manifest(_base(), settings=_settings(), master_seed=42)
    assert first.manifest_id == second.manifest_id
    assert len(first.variants) == 8
    assert first.variants["dataset_role"].eq("demo").all()
    assert first.variants["truth_source"].eq("synthetic_injection").all()
    assert (
        not first.variants.astype(str)
        .apply(lambda column: column.str.contains(str(tmp_path)).any())
        .any()
    )
    frozen = freeze_variant_manifest(first, root=tmp_path / "one")
    loaded = load_variant_manifest(frozen)
    assert loaded.variant_manifest_hash == first.variant_manifest_hash
    moved = freeze_variant_manifest(second, root=tmp_path / "elsewhere")
    assert load_variant_manifest(moved).manifest_id == first.manifest_id
    changed = build_variant_manifest(_base(), settings=_settings(), master_seed=43)
    assert changed.manifest_id != first.manifest_id
    injected = first.synthetic_events.loc[first.synthetic_events["injected_epoch"].notna()]
    assert not injected["injected_epoch"].isin(np.arange(100.0, 210.0, 0.02)).any()


@pytest.mark.parametrize(
    "event_class",
    ["planet_like", "eclipsing_binary_like", "sinusoidal_variability_like"],
)
def test_injections_are_deterministic_and_do_not_mutate_base(event_class: str) -> None:
    source = _source()
    before = source.data.copy(deep=True)
    variant = _variant(event_class, event_class)
    first = inject_variant(source, variant=variant, epoch=100.317)
    second = inject_variant(source, variant=variant, epoch=100.317)
    pd.testing.assert_frame_equal(first.data, second.data)
    pd.testing.assert_frame_equal(source.data, before)
    assert not first.data["pdcsap_flux"].equals(before["pdcsap_flux"])


def test_no_injection_control_exactly_preserves_source() -> None:
    source = _source()
    control = inject_variant(
        source, variant=_variant("control", "no_injection_control"), epoch=None
    )
    pd.testing.assert_frame_equal(control.data, source.data)
    assert control.data is not source.data


def test_injection_preserves_time_gaps_and_nonfinite_cadence_locations() -> None:
    source = _source()
    source.data.loc[10, "time"] = np.nan
    source.data.loc[100:, "time"] += 1.0
    original_time = source.data["time"].copy()
    injected = inject_variant(source, variant=_variant(), epoch=100.317)
    pd.testing.assert_series_equal(injected.data["time"], original_time)


def _demo_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    events = pd.DataFrame(
        [
            {
                "synthetic_event_id": "event-positive",
                "variant_id": "variant-positive",
                "object_id": "tic-1",
                "synthetic_event_class": "planet_like",
                "demo_gold_class": "positive",
                "injected_period_days": 2.0,
                "injected_epoch": 1.0,
                "injected_duration_days": 0.1,
                "injected_depth_or_amplitude": 0.02,
                "injected_secondary_depth": None,
                "injection_seed": 1,
                "difficulty": "easy",
                "variant_name": "planet",
                "generator_version": "v1",
                "dataset_role": "demo",
                "truth_source": "synthetic_injection",
            },
            {
                "synthetic_event_id": "event-negative",
                "variant_id": "variant-negative",
                "object_id": "tic-2",
                "synthetic_event_class": "eclipsing_binary_like",
                "demo_gold_class": "negative",
                "injected_period_days": 3.0,
                "injected_epoch": 1.0,
                "injected_duration_days": 0.2,
                "injected_depth_or_amplitude": 0.06,
                "injected_secondary_depth": 0.02,
                "injection_seed": 2,
                "difficulty": "easy",
                "variant_name": "eb",
                "generator_version": "v1",
                "dataset_role": "demo",
                "truth_source": "synthetic_injection",
            },
        ]
    )
    lineage = pd.DataFrame(
        [
            {
                "variant_id": "variant-positive",
                "synthetic_event_id": "event-positive",
                "object_id": "tic-1",
                "source_observation_id": "obs-1",
                "source_product_id": "mast:one",
                "source_raw_checksum": "1" * 64,
                "demo_observation_id": "demo-obs-1",
                "observation_group_id": "demo-og-1",
                "preprocessing_config_hash": "2" * 20,
                "bls_config_hash": "3" * 20,
                "preprocessing_status": "success",
                "detection_status": "success",
                "demo_processed_checksum": "4" * 64,
                "sector": 1,
            },
            {
                "variant_id": "variant-negative",
                "synthetic_event_id": "event-negative",
                "object_id": "tic-2",
                "source_observation_id": "obs-2",
                "source_product_id": "mast:two",
                "source_raw_checksum": "5" * 64,
                "demo_observation_id": "demo-obs-2",
                "observation_group_id": "demo-og-2",
                "preprocessing_config_hash": "2" * 20,
                "bls_config_hash": "3" * 20,
                "preprocessing_status": "success",
                "detection_status": "success",
                "demo_processed_checksum": "6" * 64,
                "sector": 2,
            },
        ]
    )
    frozen_candidates = pd.DataFrame(
        [
            {
                "candidate_id": "candidate-positive",
                "variant_id": "variant-positive",
                "observation_group_id": "demo-og-1",
                "rank": 1,
                "period": 2.0,
                "duration": 0.1,
                "epoch": 1.0,
                "power": 1.0,
                "depth": 0.02,
                "relation_to_stronger": None,
                "harmonic_ratio": None,
            },
            {
                "candidate_id": "candidate-negative",
                "variant_id": "variant-negative",
                "observation_group_id": "demo-og-2",
                "rank": 1,
                "period": 3.0,
                "duration": 0.2,
                "epoch": 1.0,
                "power": 1.0,
                "depth": 0.06,
                "relation_to_stronger": None,
                "harmonic_ratio": None,
            },
        ]
    )
    frozen_matches = pd.DataFrame(
        [
            {
                "candidate_id": "candidate-positive",
                "toi_id": "event-positive",
                "variant_id": "variant-positive",
                "object_id": "tic-1",
                "candidate_period": 2.0,
                "catalog_period": 2.0,
                "relative_period_error": 0.0,
                "candidate_epoch": 1.0,
                "catalog_epoch": 1.0,
                "phase_error": 0.0,
                "harmonic_ratio": 1.0,
                "match_type": "fundamental",
                "matched": True,
                "synthetic_event_class": "planet_like",
                "demo_gold_class": "positive",
            },
            {
                "candidate_id": "candidate-negative",
                "toi_id": "event-negative",
                "variant_id": "variant-negative",
                "object_id": "tic-2",
                "candidate_period": 3.0,
                "catalog_period": 3.0,
                "relative_period_error": 0.0,
                "candidate_epoch": 1.0,
                "catalog_epoch": 1.0,
                "phase_error": 0.0,
                "harmonic_ratio": 1.0,
                "match_type": "fundamental",
                "matched": True,
                "synthetic_event_class": "eclipsing_binary_like",
                "demo_gold_class": "negative",
            },
        ]
    )
    candidates, relations = build_demo_candidate_tables(
        frozen_candidates, frozen_matches, events, lineage, matching_config_hash="7" * 20
    )
    recovery = build_demo_event_recovery(events, lineage, relations)
    split = pd.DataFrame(
        {
            "object_id": ["tic-1", "tic-2"],
            "split": ["development", "development"],
        }
    )
    return candidates, relations, recovery, lineage, split


def test_candidate_labels_are_match_based_and_recovery_is_retained() -> None:
    candidates, relations, events, lineage, split = _demo_tables()
    assert (
        candidates.set_index("candidate_id").loc["candidate-positive", "gold_candidate_label"]
        == "positive"
    )
    assert (
        candidates.set_index("candidate_id").loc["candidate-negative", "gold_candidate_label"]
        == "negative"
    )
    assert set(events["recovery_state"]) == {"recovered_fundamental"}
    qa = validate_demo_dataset(candidates, relations, events, lineage, split)
    acceptance = evaluate_demo_acceptance(candidates, events, lineage, qa)
    assert acceptance["positive_gold_candidates"] == 1
    assert acceptance["negative_gold_candidates"] == 1


def test_unmatched_candidate_remains_unlabeled() -> None:
    candidates, _, events, lineage, split = _demo_tables()
    candidates.loc[:, "gold_candidate_label"] = "unlabeled"
    candidates.loc[:, "gold_training_eligible"] = False
    relations = pd.DataFrame(
        columns=["candidate_id", "synthetic_event_id", "variant_id", "object_id", "matched"]
    )
    qa = validate_demo_dataset(candidates, relations, events, lineage, split)
    assert qa["candidates"]["by_label"] == {"unlabeled": 2}


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("cross_variant", "crosses demo variants"),
        ("cross_tic", "crosses source TICs"),
        ("negative_corruption", "conflicts with matched truth"),
    ],
)
def test_demo_adversarial_relation_and_label_corruption_is_rejected(
    mutation: str, message: str
) -> None:
    candidates, relations, events, lineage, split = _demo_tables()
    positive = relations["candidate_id"] == "candidate-positive"
    if mutation == "cross_variant":
        relations.loc[positive, "variant_id"] = "variant-negative"
    elif mutation == "cross_tic":
        relations.loc[positive, "object_id"] = "tic-2"
    else:
        candidates.loc[
            candidates["candidate_id"] == "candidate-positive", "gold_candidate_label"
        ] = "negative"
    with pytest.raises(DemoValidationError, match=message):
        validate_demo_dataset(candidates, relations, events, lineage, split)


def test_demo_group_and_truth_feature_leakage_is_rejected() -> None:
    candidates, relations, events, lineage, split = _demo_tables()
    leaked_split = pd.concat(
        [split, pd.DataFrame([{"object_id": "tic-1", "split": "test"}])], ignore_index=True
    )
    with pytest.raises(DemoValidationError, match="duplicate split"):
        validate_demo_dataset(candidates, relations, events, lineage, leaked_split)
    with pytest.raises(DemoValidationError, match="model inputs"):
        validate_demo_dataset(
            candidates,
            relations,
            events,
            lineage,
            split,
            model_input_columns=("candidate_period", "injected_period_days"),
        )
    with pytest.raises(ValueError, match="Catalog evaluation"):
        assert_model_input_boundary(["injected_depth_or_amplitude"])


def test_duplicate_demo_identities_and_checksum_leakage_are_rejected() -> None:
    candidates, relations, events, lineage, split = _demo_tables()
    with pytest.raises(DemoValidationError, match="Duplicate variant"):
        validate_demo_dataset(
            candidates,
            relations,
            events,
            pd.concat([lineage, lineage.iloc[[0]]], ignore_index=True),
            split,
        )
    duplicated_events = pd.concat([events, events.iloc[[0]]], ignore_index=True)
    with pytest.raises(DemoValidationError, match="Duplicate synthetic event"):
        validate_demo_dataset(candidates, relations, duplicated_events, lineage, split)
    with pytest.raises(DemoValidationError, match="Duplicate candidate"):
        validate_demo_dataset(
            pd.concat([candidates, candidates.iloc[[0]]], ignore_index=True),
            relations,
            events,
            lineage,
            split,
        )
    leaked_lineage = lineage.copy()
    leaked_lineage.loc[1, "source_raw_checksum"] = leaked_lineage.loc[0, "source_raw_checksum"]
    leaked_split = split.copy()
    leaked_split.loc[1, "split"] = "test"
    with pytest.raises(DemoValidationError, match="Source checksum"):
        validate_demo_dataset(candidates, relations, events, leaked_lineage, leaked_split)
    observation_leakage = lineage.copy()
    observation_leakage.loc[1, "source_observation_id"] = observation_leakage.loc[
        0, "source_observation_id"
    ]
    with pytest.raises(DemoValidationError, match="Source observation"):
        validate_demo_dataset(candidates, relations, events, observation_leakage, leaked_split)


def test_no_injection_negative_and_ambiguous_eligibility_are_rejected() -> None:
    candidates, relations, events, lineage, split = _demo_tables()
    positive_candidate = candidates["candidate_id"] == "candidate-positive"
    positive_relation = relations["candidate_id"] == "candidate-positive"
    positive_event = events["synthetic_event_id"] == "event-positive"
    events.loc[positive_event, "synthetic_event_class"] = "no_injection_control"
    events.loc[positive_event, "demo_gold_class"] = "negative"
    relations.loc[positive_relation, "demo_gold_class"] = "negative"
    candidates.loc[positive_candidate, "gold_candidate_label"] = "negative"
    with pytest.raises(DemoValidationError, match="No-injection"):
        validate_demo_dataset(candidates, relations, events, lineage, split)

    candidates, relations, events, lineage, split = _demo_tables()
    negative_relation = relations["candidate_id"] == "candidate-negative"
    relations.loc[negative_relation, "matched"] = False
    with pytest.raises(DemoValidationError, match="conflicts with matched truth"):
        validate_demo_dataset(candidates, relations, events, lineage, split)

    candidates, relations, events, lineage, split = _demo_tables()
    extra_event = events.iloc[[1]].copy()
    extra_event["synthetic_event_id"] = "event-conflicting"
    extra_event["event_id"] = "event-conflicting"
    extra_event["variant_id"] = "variant-positive"
    extra_event["object_id"] = "tic-1"
    extra_event["synthetic_event_class"] = "eclipsing_binary_like"
    extra_event["demo_gold_class"] = "negative"
    events = pd.concat([events, extra_event], ignore_index=True)
    extra_relation = relations.iloc[[1]].copy()
    extra_relation["candidate_id"] = "candidate-positive"
    extra_relation["synthetic_event_id"] = "event-conflicting"
    extra_relation["event_id"] = "event-conflicting"
    extra_relation["variant_id"] = "variant-positive"
    extra_relation["object_id"] = "tic-1"
    relations = pd.concat([relations, extra_relation], ignore_index=True)
    candidates.loc[positive_candidate, "gold_candidate_label"] = "ambiguous"
    candidates.loc[positive_candidate, "gold_training_eligible"] = True
    with pytest.raises(DemoValidationError, match="eligibility"):
        validate_demo_dataset(candidates, relations, events, lineage, split)


def test_demo_identity_changes_only_with_portable_scientific_inputs() -> None:
    manifest = build_variant_manifest(_base(), settings=_settings(), master_seed=42)
    lineage = pd.DataFrame(
        {
            "source_raw_checksum": ["1" * 64],
            "demo_processed_checksum": ["2" * 64],
        }
    )
    first = _demo_identity(
        manifest,
        lineage,
        preprocessing_hash="3" * 20,
        bls_hash="4" * 20,
        matching_hash="5" * 20,
        dataset_policy_hash="6" * 20,
    )
    replay = _demo_identity(
        manifest,
        lineage.copy(),
        preprocessing_hash="3" * 20,
        bls_hash="4" * 20,
        matching_hash="5" * 20,
        dataset_policy_hash="6" * 20,
    )
    assert first == replay
    changed = _demo_identity(
        manifest,
        lineage.assign(source_raw_checksum="9" * 64),
        preprocessing_hash="3" * 20,
        bls_hash="4" * 20,
        matching_hash="5" * 20,
        dataset_policy_hash="6" * 20,
    )
    assert changed[0] != first[0]


def test_fresh_demo_processing_keeps_truth_out_of_bls_and_matches_after_freeze(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = DemoSettings(
        policy_version="demo-v1",
        generator_version="generator-v1",
        base_expansion_manifest_id="expansion-test",
        seed_component="demo",
        split_policy="all_development",
        variants=(_variant(),),
    )
    base = _base().iloc[[0]].copy()
    base["relative_raw_path"] = "tess/base.fits"
    base["sector"] = 1
    base["author"] = "TESS-SPOC"
    base["exposure_seconds"] = 200.0
    manifest = build_variant_manifest(base, settings=settings, master_seed=42)
    bls_settings = SimpleNamespace(
        model_dump=lambda mode: {"frozen": True},
    )
    preprocessing_hash = "a" * 20
    expected_bls_hash = demo_builder.content_hash(
        {"bls": {"frozen": True}, "preprocessing_hash": preprocessing_hash}
    )
    monkeypatch.setattr(
        demo_builder,
        "load_config",
        lambda path: SimpleNamespace(demo=settings),
    )
    monkeypatch.setattr(
        demo_builder,
        "read_tess_lightcurve",
        lambda path, observation_id: _source(),
    )
    monkeypatch.setattr(
        demo_builder,
        "preprocess",
        lambda record, **kwargs: LightCurveRecord(
            record.data.assign(
                valid=True,
                normalized_flux=record.data["pdcsap_flux"] / 1000.0,
                detrended_flux=record.data["pdcsap_flux"] / 1000.0,
            ),
            record.metadata,
        ),
    )

    def fake_bls(
        cadences: pd.DataFrame,
        *,
        settings: object,
        observation_group_id: str,
        preprocessing_hash: str,
    ) -> SimpleNamespace:
        forbidden = {
            "injected_period_days",
            "injected_epoch",
            "injected_duration_days",
            "injected_depth_or_amplitude",
            "synthetic_event_class",
        }
        assert forbidden.isdisjoint(cadences.columns)
        return SimpleNamespace(
            config_hash=expected_bls_hash,
            observation_group_id=observation_group_id,
            periodogram=pd.DataFrame(
                [{"period": 2.5, "power": 1.0, "duration": 0.12, "epoch": 100.5, "depth": 0.02}]
            ),
        )

    monkeypatch.setattr(demo_builder, "run_bls", fake_bls)

    def fake_peaks(result: SimpleNamespace, settings: object) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "candidate_id": make_candidate_id(
                        observation_group_id=result.observation_group_id,
                        bls_config_hash=result.config_hash,
                        rank=1,
                    ),
                    "rank": 1,
                    "period": 2.5,
                    "duration": 0.12,
                    "epoch": 100.5,
                    "power": 1.0,
                    "depth": 0.02,
                    "relation_to_stronger": None,
                    "harmonic_ratio": None,
                }
            ]
        )

    monkeypatch.setattr(demo_builder, "extract_peaks", fake_peaks)

    def fake_match(
        candidates: pd.DataFrame, events: pd.DataFrame, settings: object
    ) -> pd.DataFrame:
        candidate_files = list(
            (tmp_path / f"data/synthetic/{manifest.manifest_id}/detection").glob(
                "*-candidates.parquet"
            )
        )
        assert len(candidate_files) == 1
        return pd.DataFrame(
            [
                {
                    "candidate_id": candidates.iloc[0]["candidate_id"],
                    "toi_id": events.iloc[0]["toi_id"],
                    "candidate_period": 2.5,
                    "catalog_period": 2.5,
                    "relative_period_error": 0.0,
                    "candidate_epoch": 100.5,
                    "catalog_epoch": float(events.iloc[0]["transit_epoch_bjd"]),
                    "phase_error": 0.0,
                    "harmonic_ratio": 1.0,
                    "match_type": "fundamental",
                    "matched": True,
                }
            ]
        )

    monkeypatch.setattr(demo_builder, "match_candidates", fake_match)
    lineage, candidates, matches, bls_hash = demo_builder._process_variants(
        tmp_path,
        manifest,
        base,
        preprocessing_hash=preprocessing_hash,
        preprocessing_settings=SimpleNamespace(
            gap_days=0.5, trend_window_cadences=101, positive_spike_mad=8.0
        ),
        bls_settings=bls_settings,
    )
    assert lineage.loc[0, "preprocessing_status"] == "success"
    assert lineage.loc[0, "detection_status"] == "success"
    assert len(candidates) == len(matches) == 1
    assert bls_hash == expected_bls_hash
    replay = demo_builder._process_variants(
        tmp_path,
        manifest,
        base,
        preprocessing_hash=preprocessing_hash,
        preprocessing_settings=SimpleNamespace(
            gap_days=0.5, trend_window_cadences=101, positive_spike_mad=8.0
        ),
        bls_settings=bls_settings,
    )
    assert replay[3] == bls_hash


def test_frozen_real_demo_replays_without_network_or_identity_changes() -> None:
    root = Path(__file__).resolve().parents[1]
    frozen = root / "data/datasets/dataset-demo-a74b6aad7c21faf15b0d"
    if not frozen.is_dir():
        pytest.skip("Frozen real-substrate demo artifacts are not present in this checkout.")
    from transiteye.demo.builder import build_demo_dataset

    first = build_demo_dataset(root)
    second = build_demo_dataset(root)
    assert first.dataset_version == second.dataset_version == "dataset-demo-a74b6aad7c21faf15b0d"
    assert first.manifest_id == second.manifest_id == "demo-manifest-9459aec7c2e26e406301"
    assert first.acceptance["ready"] is True
    assert first.qa_summary["candidates"]["by_label"] == {
        "unlabeled": 245,
        "positive": 70,
        "negative": 60,
    }
