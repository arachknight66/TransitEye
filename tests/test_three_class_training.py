"""Source isolation and label-match checks for the research-only trainer."""
import json
from pathlib import Path

from transiteye.evaluation.three_class_training import INPUTS, SPLITS, _period_matches, prepare


def test_catalog_matching_allows_only_declared_aliases() -> None:
    assert _period_matches(2.01, 2.0)
    assert _period_matches(1.0, 2.0)
    assert _period_matches(4.0, 2.0)
    assert not _period_matches(2.2, 2.0)


def test_prepare_excludes_reserved_and_conflicting_sources(tmp_path: Path) -> None:
    manifests = tmp_path / 'manifests'
    manifests.mkdir()
    for i, name in enumerate(SPLITS):
        rows = [{'source_group_id': 'tic-1', 'split': 'test'}] if i == 3 else []
        (manifests / f'{name}.splits.json').write_text(json.dumps({'assignments': rows}))
    data = tmp_path / '.transiteye-workspace' / 'data'
    (data / 'gaia_variable_acquisition').mkdir(parents=True)
    (data / 'gaia_variable_acquisition' / 'tic-gaia-dr2.json').write_text('[]')
    for label, folder in INPUTS:
        path = data / folder
        path.mkdir(parents=True)
        tic = '1' if label == 'transit_like' else '2'
        records = [{'tic_id': tic, 'filename': f'{tic}.fits', 'sha256': 'abc'}]
        (path / 'selection.json').write_text(json.dumps({'records': records}))
    output = tmp_path / 'output'
    result = prepare(tmp_path, output)
    assert result['records'] == []
    assert len(result['excluded']) == len(INPUTS)
    assert {r['reason'] for r in result['excluded']} == {
        'source has an existing non-training partition assignment',
        'conflicting source-level catalog classes',
    }
    assert (output / 'manifest.json').exists()
