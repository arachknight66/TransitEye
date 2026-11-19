from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from astropy.io import fits

from transiteye.domain.models import LightCurve, Provenance
from transiteye.io.lookup import ArchiveProduct, TargetReference
from transiteye.io.observation_set import build_observation_set, load_verified_observation_set


def curve(
    *, mission: str = "TESS", target: str = "42", reference: str = "2457000.0",
    checksum: str = "a" * 64, observation: int = 1,
) -> LightCurve:
    return LightCurve(
        time_days=np.array([1.0, 2.0, np.nan]),
        flux=np.array([100.0, 99.0, 98.0]),
        flux_error=None, quality=np.array([0, 1, 0]), cadence_number=None,
        centroid_column=None, centroid_row=None,
        flux_kind="PDCSAP_FLUX", flux_unit="e-/s",
        provenance=Provenance(
            source_uri=f"mast:{mission}:{observation}", source_path=None,
            checksum_sha256=checksum, product_type=f"{mission.lower()}_light_curve",
            target_id=target, sector=observation if mission == "TESS" else None,
            time_system="TDB", time_reference=reference,
            metadata={"mission": mission, "quarter": str(observation)},
        ),
        metadata={"time_unit": "d"},
    )


def test_observation_set_retains_each_source_and_converts_reference() -> None:
    first = curve()
    second = curve(observation=2, checksum="b" * 64, reference="2454833.0")
    result = build_observation_set((first, second))
    assert len(result.observations) == 2
    assert result.observations[0].source is first
    assert result.observations[1].source is second
    assert result.observations[0].time_bjd_tdb[0] == 2457001.0
    assert result.observations[1].time_bjd_tdb[0] == 2454834.0
    assert np.isnan(result.observations[0].time_bjd_tdb[-1])
    assert result.input_checksums == ("a" * 64, "b" * 64)


@pytest.mark.parametrize("bad", [
    curve(target="99", checksum="c" * 64),
    curve(mission="Kepler", checksum="d" * 64),
    curve(reference="", checksum="e" * 64),
])
def test_observation_set_rejects_incompatible_inputs(bad: LightCurve) -> None:
    with pytest.raises(ValueError):
        build_observation_set((curve(), bad))


def test_observation_set_rejects_duplicate_product() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        build_observation_set((curve(), curve()))


def _sample(
    times: list[float], *, checksum: str, width: float | None = 1.0,
    quality: list[int] | None = None, flux: list[float] | None = None,
    observation: int = 1, mission: str = "TESS",
) -> LightCurve:
    base = curve(checksum=checksum, observation=observation, mission=mission)
    metadata = dict(base.provenance.metadata)
    if width is not None:
        metadata["TIMEDEL"] = str(width)
    return replace(
        base,
        time_days=np.array(times, dtype=float),
        flux=np.array(flux if flux is not None else [100.0] * len(times)),
        quality=np.array(quality if quality is not None else [0] * len(times)),
        provenance=replace(base.provenance, metadata=metadata),
    )


def test_partial_overlap_prefers_spoc_and_retains_gaps() -> None:
    # Window geometry: unit-width samples at 0, 1, 2, 3 and 2, 3, 5.
    spoc = _sample([0, 1, 2, 3], checksum="a" * 64)
    tess_spoc = _sample([2, 3, 5], checksum="b" * 64, observation=2)
    result = build_observation_set((tess_spoc, spoc), providers=("TESS-SPOC", "SPOC"))
    assert result.observations[0].analysis_mask.tolist() == [False, False, True]
    assert result.observations[1].analysis_mask.tolist() == [True, True, True, True]
    assert result.observations[0].selected_count == 1
    assert result.observations[0].excluded_count == 2
    assert len(result.observations[0].source.time_days) == 3
    with pytest.raises(ValueError):
        result.observations[0].analysis_mask[0] = True


def test_same_provider_uses_shorter_sampling_and_stable_ties() -> None:
    coarse = _sample([0, 1], checksum="b" * 64, width=1)
    fine = _sample([0, 0.5, 1], checksum="c" * 64, width=0.5, observation=2)
    tied = _sample([0, 0.5], checksum="a" * 64, width=0.5, observation=3)
    first = build_observation_set((coarse, fine, tied), providers=("SPOC",) * 3)
    second = build_observation_set((tied, fine, coarse), providers=("SPOC",) * 3)
    selected_first = {o.source.provenance.checksum_sha256: o.analysis_mask.tolist()
                      for o in first.observations}
    selected_second = {o.source.provenance.checksum_sha256: o.analysis_mask.tolist()
                       for o in second.observations}
    assert selected_first == selected_second
    assert selected_first == {"a" * 64: [True, True], "b" * 64: [False, False],
                              "c" * 64: [False, False, True]}


def test_invalid_duplicate_is_not_selected_and_valid_duplicate_is_unique() -> None:
    invalid = _sample([0, 1, 1], checksum="a" * 64, quality=[0, 2, 0])
    other = _sample([1, 2], checksum="b" * 64, observation=2)
    result = build_observation_set((invalid, other), providers=("SPOC", "SPOC"))
    assert result.observations[0].analysis_mask.tolist() == [True, False, False]
    assert result.observations[1].analysis_mask.tolist() == [False, True]
    assert result.observations[0].selection_reason == (
        "selected", "quality_flag", "overlap_excluded",
    )
    assert result.observations[0].overlap_winner_checksum[2] == "a" * 64
    assert result.observations[1].overlap_winner_checksum[0] == "a" * 64


def test_bad_preferred_cadence_does_not_promote_lower_provider() -> None:
    preferred = _sample([0, 1, 2], checksum="a" * 64, flux=[100, np.nan, 100])
    alternate = _sample([0, 1, 2], checksum="b" * 64, observation=2)
    result = build_observation_set(
        (preferred, alternate), providers=("SPOC", "TESS-SPOC"),
    )
    assert result.observations[0].selection_reason == (
        "selected", "invalid_flux", "selected",
    )
    assert result.observations[0].overlap_mask.tolist() == [True, True, True]
    assert result.observations[1].selection_reason == ("overlap_excluded",) * 3
    assert result.observations[1].overlap_mask.tolist() == [False] * 3
    assert result.observations[1].overlap_winner_checksum == ("a" * 64,) * 3
    assert result.observations[0].analysis_mask.tolist() == [True, False, True]
    with pytest.raises(ValueError):
        result.observations[0].overlap_mask[0] = False


def test_unknown_width_with_intersecting_product_spans_is_unresolved() -> None:
    first = _sample([0.1], checksum="a" * 64, width=None)
    second = _sample([0, 0.1, 0.2], checksum="b" * 64, width=0.1, observation=2)
    result = build_observation_set((first, second))
    assert result.observations[0].cadence_source == "unknown"
    assert result.observations[0].selection_reason == ("overlap_unresolved",)
    assert "overlap_unresolved" in result.observations[1].selection_reason


def test_singleton_timedel_cannot_establish_cadence_width() -> None:
    lone = _sample([0], checksum="a" * 64, width=0.000347222, mission="Kepler")
    regular = _sample(
        [-0.0204, 0, 0.0204], checksum="b" * 64, width=0.0204,
        mission="Kepler", observation=2,
    )
    result = build_observation_set((lone, regular))
    assert result.observations[0].cadence_days is None
    assert result.observations[0].selection_reason == ("overlap_unresolved",)


def test_no_overlap_and_missing_timedel_spacing_assumption() -> None:
    first = _sample([0, 1, 2], checksum="a" * 64, width=None)
    second = _sample([10, 11, 12], checksum="b" * 64, width=None, observation=2)
    result = build_observation_set((first, second))
    assert all(observation.analysis_mask.all() for observation in result.observations)
    assert all(observation.cadence_source == "median_positive_spacing"
               for observation in result.observations)
    assert all(observation.cadence_days == 1.0 for observation in result.observations)


def test_kepler_timedel_resolution_does_not_hide_long_short_overlap() -> None:
    # Sample centers are 0.0204 d apart for the long series while TIMEDEL is
    # a much smaller resolution value. The short centers fall inside its
    # middle cadence's sampling window, so the shorter cadence wins.
    long = _sample(
        [0, 0.0204, 0.0408], checksum="a" * 64,
        width=0.000347222, mission="Kepler",
    )
    short = _sample(
        [0.0198, 0.0205, 0.0212], checksum="b" * 64,
        width=0.0007, mission="Kepler", observation=2,
    )
    result = build_observation_set((long, short), providers=("Kepler", "Kepler"))
    long_result, short_result = result.observations
    assert long_result.cadence_days == pytest.approx(0.0204, rel=1e-6)
    assert long_result.cadence_source == "median_positive_spacing"
    assert long_result.overlap_mask.tolist() == [True, False, True]
    assert long_result.overlap_winner_checksum[1] == "b" * 64
    assert short_result.overlap_mask.tolist() == [True, True, True]


def test_two_timestamp_timedel_requires_spacing_consistency() -> None:
    consistent = _sample([0, 1], checksum="a" * 64, width=1)
    inconsistent = _sample([0, 1], checksum="b" * 64, width=0.01, observation=2)
    result = build_observation_set((consistent, inconsistent))
    assert result.observations[0].cadence_source == "TIMEDEL_spacing_consistent"
    assert result.observations[1].cadence_days is None
    assert result.observations[1].cadence_source == "unknown_two_timestamps"
    assert all(reason == "overlap_unresolved"
               for reason in result.observations[1].selection_reason)


def test_load_verified_observations_requires_all_products(tmp_path) -> None:
    source = tmp_path / "kplr000000042-1_llc.fits"
    primary = fits.PrimaryHDU()
    primary.header["KEPLERID"] = 42
    primary.header["QUARTER"] = 3
    table = fits.BinTableHDU.from_columns([
        fits.Column(name="TIME", format="D", array=np.array([1., 2.])),
        fits.Column(name="SAP_FLUX", format="D", array=np.array([100., 99.])),
        fits.Column(name="SAP_QUALITY", format="K", array=np.array([0, 0])),
    ])
    table.header["BJDREFI"] = 2454833
    table.header["TIMESYS"] = "TDB"
    table.header["TIMEUNIT"] = "d"
    fits.HDUList([primary, table]).writeto(source)
    product = ArchiveProduct("mast:one", source.name, "Kepler", "42", 3, "Kepler")
    target = TargetReference.parse("KIC 42")
    with pytest.raises(ValueError, match="missing"):
        load_verified_observation_set(target, (product,), {})
    result = load_verified_observation_set(target, (product,), {"mast:one": source})
    assert result.observations[0].archive_uri == "mast:one"
    assert result.observations[0].time_bjd_tdb.tolist() == [2454834.0, 2454835.0]


def test_inspect_target_set_cli_reports_product_provenance(tmp_path, capsys) -> None:
    from transiteye import __main__

    source = tmp_path / "kplr000000042-1_llc.fits"
    primary = fits.PrimaryHDU()
    primary.header["KEPLERID"] = 42
    primary.header["QUARTER"] = 3
    table = fits.BinTableHDU.from_columns([
        fits.Column(name="TIME", format="D", array=np.array([1., 2.])),
        fits.Column(name="SAP_FLUX", format="D", array=np.array([100., 99.])),
    ])
    table.header["BJDREFI"] = 2454833
    table.header["TIMESYS"] = "TDB"
    table.header["TIMEUNIT"] = "d"
    fits.HDUList([primary, table]).writeto(source)
    from transiteye.config import AcquisitionConfig
    from transiteye.io.lookup import acquire_target_products

    class LocalArchive:
        def download(self, data_uri: str, destination) -> None:
            destination.write_bytes(source.read_bytes())

    target = TargetReference.parse("KIC 42")
    product = ArchiveProduct("mast:one", source.name, "Kepler", "42", 3, "Kepler")
    manifest, completed = acquire_target_products(
        target, (product,), LocalArchive(), tmp_path / "cache", AcquisitionConfig(max_attempts=1),
    )
    assert completed["mast:one"].is_file()
    assert __main__.main(["inspect-target-set", str(tmp_path / "cache" / manifest.manifest_id / "manifest.json")]) == 0
    output = capsys.readouterr().out
    assert '"observation_count": 1' in output
    assert '"archive_uri": "mast:one"' in output
    assert '"provider": "Kepler"' in output
    assert '"selected_cadence_count": 2' in output
    assert '"overlap_unresolved_count": 0' in output
