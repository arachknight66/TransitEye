"""Checks for the independent annular transit diagnostic."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import pytest

from transiteye.evaluation.physical_contact import radial_flux, run


def test_uniform_star_matches_analytic_overlap() -> None:
    radius = 0.3
    for separation in (0.0, 0.4, 0.7, 0.9, 1.3, 1.5):
        if separation <= 1.0 - radius:
            overlap = np.pi * radius**2
        elif separation >= 1.0 + radius:
            overlap = 0.0
        else:
            z = separation
            overlap = (radius**2 * np.arccos((z*z + radius*radius - 1)/(2*z*radius))
                       + np.arccos((z*z + 1 - radius*radius)/(2*z))
                       - 0.5 * np.sqrt((-z + radius + 1)*(z + radius - 1)
                                       *(z - radius + 1)*(z + radius + 1)))
        assert radial_flux(separation, radius, 0, 0) == pytest.approx(1 - overlap / np.pi, abs=1e-10)


def test_invalid_geometry_is_rejected() -> None:
    with pytest.raises(ValueError):
        radial_flux(float('nan'), 0.1, 0.3, 0.22)


def test_native_warning_marks_case_unusable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = tmp_path / 'config.toml'
    config.write_text('schema_version="1"\nradius_ratios=[0.1]\ncontact_offsets=[0.0]\n'
                      'absolute_tolerance_ppm=0.01\n')
    payload = json.dumps({'radius_ratio': 0.1, 'offset': 0.0, 'requested_z': 0.9,
                          'actual_z': 0.9, 'batman_flux': 1.0, 'radial_flux': 1.0,
                          'absolute_error_ppm': 0.0})
    monkeypatch.setattr('transiteye.evaluation.physical_contact.subprocess.run',
                        lambda *args, **kwargs: SimpleNamespace(returncode=0,
                            stdout=f'ROW={payload}\nnative warning\n', stderr=''))
    output = tmp_path / 'results'
    summary = run(config, output)
    row = pq.read_table(output / 'cases.parquet').to_pylist()[0]
    assert summary['unusable'] == 1 and summary['within_tolerance'] == 0
    assert row['native_diagnostics'] == 'native warning'
    with pytest.raises(FileExistsError):
        run(config, output)
