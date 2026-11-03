"""Operational and optimizer parity checks for the physical-fit pilot."""
from pathlib import Path
from types import SimpleNamespace

import batman
import numpy as np
import pyarrow.parquet as pq
import pytest

from transiteye.characterization.fitting import fit_batman_refinement
from transiteye.evaluation.physical_fit_pilot import params_from_vector, run, simpson_fit


def test_experimental_optimizer_matches_production_with_same_forward_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    times = np.concatenate((np.linspace(-0.2, 0.2, 80), np.linspace(2.8, 3.2, 80)))
    params = params_from_vector(np.array([0.0, 3.0, 0.1, 10.0, 88.0]), [0.3, 0.22])
    exposure = 2 / 1440
    flux = batman.TransitModel(params, times, supersample_factor=5, exp_time=exposure).light_curve(params)
    errors = np.full(len(times), 0.0001)

    def five(params: batman.TransitParams, ts: np.ndarray, exp: float, points: int) -> np.ndarray:
        return batman.TransitModel(params, ts, supersample_factor=5, exp_time=exp).light_curve(params)

    monkeypatch.setattr('transiteye.evaluation.physical_fit_pilot.simpson_flux', five)
    production = fit_batman_refinement(times, flux, errors, 0, 3, 0.1, 0.01, exposure_days=exposure)
    experimental = simpson_fit(times, flux, errors, 0, 3, 0.1, 0.01, [0.3, 0.22], exposure, 51, 200)
    assert production is not None and experimental is not None
    for key in experimental:
        assert experimental[key] == pytest.approx(production[0][key], rel=1e-8, abs=1e-8)


def test_worker_failure_records_both_methods(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = tmp_path / 'config.toml'
    config.write_text('schema_version="1"\nmax_nfev=200\nradius_ratios=[0.1]\n'
                      'impact_parameters=[0.3]\nexposure_minutes=[2.0]\ntruth_tolerance_ppm=0.1\n')
    monkeypatch.setattr('transiteye.evaluation.physical_fit_pilot.subprocess.run',
                        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout='', stderr='failed'))
    output = tmp_path / 'output'
    summary = run(config, output)
    rows = pq.read_table(output / 'fits.parquet').to_pylist()
    assert summary['rows'] == 2 and summary['usable'] == 0
    assert {row['method'] for row in rows} == {'production_five', 'simpson_51'}
    assert all(row['error'] and row['native_diagnostics'] == 'failed' for row in rows)
    with pytest.raises(FileExistsError):
        run(config, output)


def test_repeated_grid_preserves_noise_replicates_and_pairing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json

    config = tmp_path / 'config.toml'
    config.write_text('schema_version="1"\nmax_nfev=200\nradius_ratios=[0.1]\n'
                      'impact_parameters=[0.3]\nexposure_minutes=[2.0]\n'
                      'noise_levels_ppm=[50.0,200.0]\nreplicates=3\ntruth_tolerance_ppm=0.1\n')

    def fake_worker(args: list[str], **kwargs: object) -> SimpleNamespace:
        case_config = json.loads(args[3])
        index = int(args[7])
        base = {'case_index': index, 'radius_ratio': float(args[4]), 'impact_parameter': float(args[5]),
                'exposure_minutes': float(args[6]), 'noise_ppm': case_config['noise_ppm'],
                'replicate': case_config['replicate'], 'seed': 10 + index,
                'truth_reference_difference_ppm': 0.0, 'error': None}
        rows = [{**base, 'method': method} for method in ('production_five', 'simpson_51')]
        return SimpleNamespace(returncode=0, stdout='ROWS=' + json.dumps({'rows': rows}), stderr='')

    monkeypatch.setattr('transiteye.evaluation.physical_fit_pilot.subprocess.run', fake_worker)
    output = tmp_path / 'output'
    summary = run(config, output)
    rows = pq.read_table(output / 'fits.parquet').to_pylist()
    assert summary['rows'] == 12 and summary['usable'] == 12
    assert {(r['noise_ppm'], r['replicate']) for r in rows} == {
        (noise, replicate) for noise in (50.0, 200.0) for replicate in range(3)
    }
    assert all(len({r['seed'] for r in rows if r['case_index'] == i}) == 1 for i in range(6))


@pytest.mark.parametrize('mode', ['invalid', 'timeout'])
def test_invalid_or_timed_out_worker_is_accounted_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str,
) -> None:
    import subprocess

    config = tmp_path / 'config.toml'
    config.write_text('schema_version="1"\nmax_nfev=200\nradius_ratios=[0.1]\n'
                      'impact_parameters=[0.3]\nexposure_minutes=[2.0]\ntruth_tolerance_ppm=0.1\n')

    def fake_worker(*args: object, **kwargs: object) -> SimpleNamespace:
        if mode == 'timeout':
            raise subprocess.TimeoutExpired('worker', 120)
        return SimpleNamespace(returncode=0, stdout='ROWS={invalid json}', stderr='')

    monkeypatch.setattr('transiteye.evaluation.physical_fit_pilot.subprocess.run', fake_worker)
    output = tmp_path / mode
    summary = run(config, output)
    rows = pq.read_table(output / 'fits.parquet').to_pylist()
    assert summary['rows'] == 2 and summary['usable'] == 0
    assert all(row['error'] for row in rows)
