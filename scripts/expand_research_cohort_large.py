"""Frozen three-class development expansion; never consumes sealed source observations."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pyarrow.parquet as pq
import requests
from astropy.io import fits
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from transiteye.evaluation.three_class_training import LABELS, _extract
from transiteye.provenance import code_fingerprint

ROOT = Path.cwd()
DATA = ROOT / '.transiteye-workspace/data'
PREVIOUS = ROOT / '.transiteye-workspace/remediation/three-class-research-v3'
EXPANSION = ROOT / '.transiteye-workspace/remediation/three-class-expansion-20261102-v2'
COUNTS = {'transit_like': 38, 'eclipse_like': 60, 'stellar_variability': 2}
TEST_COUNTS = {'transit_like': 6, 'eclipse_like': 6, 'stellar_variability': 1}
DOWNLOAD = 'https://stpubdata.s3.amazonaws.com/tess/public/tid'


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def save(path: Path, data: Any) -> None:
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temp.replace(path)


def prepare(output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use an empty output directory.')
    output.mkdir(parents=True, exist_ok=True)
    sources = sorted(DATA.glob('*/selection.json'))
    splits = sorted((ROOT / 'manifests').glob('*.splits.json'))
    previous_manifest = load(PREVIOUS / 'manifest.json')
    if previous_manifest['backend_sha256'] != code_fingerprint():
        raise ValueError('Prior features were extracted by different backend code.')
    old = [row for p in sources for row in load(p).get('records', [])]
    old.extend(load(EXPANSION / 'manifest.json')['records'])
    blocked_tic = {str(row.get('tic_id')) for row in old}
    blocked_gaia = {str(row['gaia_dr2']) for row in old if row.get('gaia_dr2')}
    for row in previous_manifest['records']:
        blocked_tic.add(str(row['tic_id']))
        if row.get('gaia_dr2'):
            blocked_gaia.add(str(row['gaia_dr2']))
    for p in splits:
        for row in load(p).get('assignments', []):
            blocked_tic.add(str(row.get('source_group_id', '')).removeprefix('tic-'))
    mapping = DATA / 'gaia_variable_acquisition/tic-gaia-dr2.json'
    for row in load(mapping):
        if str(row['tic_id']) in blocked_tic and row.get('gaia_dr2'):
            blocked_gaia.add(str(row['gaia_dr2']))
    staged = {p.name for p in DATA.glob('*/staging/*.fits')}
    planet_catalog = DATA / 'confirmed_transiting_hosts_v2/candidate-catalog.json'
    planet_discovery = DATA / 'confirmed_transiting_hosts_v2/mast-discovery.json'
    inventories = [DATA / 'eclipse_like_gaia_v2/candidate-inventory.json',
                   DATA / 'stellar_variability_gaia_v2/candidate-inventory.json']
    discovery = load(planet_discovery)
    candidates = []
    for row in load(planet_catalog):
        p = row['planet']
        candidates.append({'tic_id': str(row['tic']), 'gaia_dr2': str(row['gaia_dr2']),
            'label': 'transit_like', 'catalog_period_days': p['pl_orbper'],
            'catalog_epoch_bjd_tdb': p['pl_tranmid'], 'products': discovery.get(row['tic'], []),
            'catalog_name': p['pl_name']})
    for path, label in zip(inventories, LABELS[1:], strict=True):
        for row in load(path):
            candidates.append({'tic_id': str(row['tic_id']), 'gaia_dr2': str(row['gaia_dr2']),
                'label': label, 'catalog_period_days': row.get('gaia_period_days'),
                'catalog_epoch_bjd_tdb': None, 'products': row['products'],
                'catalog_name': row['class']})
    eligible = [r for r in candidates if r['tic_id'] not in blocked_tic
        and r['gaia_dr2'] not in blocked_gaia and r['gaia_dr2'].isdigit() and r['products']
        and all(uri.rsplit('/', 1)[-1] not in staged for uri in r['products'])]
    classes: dict[str, set[str]] = defaultdict(set)
    for row in eligible:
        classes['tic:' + row['tic_id']].add(row['label'])
        classes['gaia:' + row['gaia_dr2']].add(row['label'])
    eligible = [r for r in eligible if len(classes['tic:' + r['tic_id']]) == 1
                and len(classes['gaia:' + r['gaia_dr2']]) == 1]
    records = []
    seen_tic, seen_gaia = set(), set()
    for label in LABELS:
        pool = sorted((r for r in eligible if r['label'] == label),
                      key=lambda r: hashlib.sha256(('expansion-v1:' + r['gaia_dr2']).encode()).hexdigest())
        chosen = []
        for row in pool:
            if row['tic_id'] in seen_tic or row['gaia_dr2'] in seen_gaia:
                continue
            seen_tic.add(row['tic_id'])
            seen_gaia.add(row['gaia_dr2'])
            chosen.append(row)
            if len(chosen) == COUNTS[label]:
                break
        if len(chosen) < COUNTS[label]:
            raise ValueError(f'Only {len(chosen)} eligible sources for {label}.')
        for i, row in enumerate(chosen):
            records.append({**row, 'partition': 'test' if i < TEST_COUNTS[label] else 'train',
                            'group': 'gaia-dr2-' + row['gaia_dr2'],
                            'products': sorted(row['products'])[:4]})
    paths = sources + splits + [mapping, planet_catalog, planet_discovery, *inventories,
                                PREVIOUS / 'manifest.json', PREVIOUS / 'feature-results.parquet',
                                EXPANSION / 'manifest.json', EXPANSION / 'model-lock.json',
                                *(EXPANSION / 'checkpoints' / f"{r['tic_id']}.json"
                                  for r in load(EXPANSION / 'manifest.json')['records']
                                  if r['partition'] == 'train'),
                                Path(__file__).resolve()]
    manifest = {'protocol': 'three-class-large-expansion-v1', 'backend_sha256': code_fingerprint(),
        'selection': '38 transit, 60 eclipse, 2 variability; deterministic Gaia-hash order; '
            'exclude earlier selections, split identities, known Gaia aliases and staged products',
        'input_sha256': {str(p.resolve()): digest(p) for p in paths},
        'limits': ['catalog classes are proxies, not independently adjudicated',
            'known TIC/Gaia mappings cannot rule out all source aliases',
            'period/epoch matching selects successful detections, not all sources',
            'three classes, no other_artifact; uncalibrated; not an application bundle',
            'source inventories and earlier data are exploratory, not a blinded survey'],
        'records': records}
    save(output / 'manifest.json', manifest)
    save(output / 'manifest.sha256.json', {'sha256': digest(output / 'manifest.json')})
    print('Frozen:', dict(Counter((r['partition'], r['label']) for r in records)), flush=True)


def extract_source(row: dict[str, Any], output: Path) -> dict[str, Any]:
    errors = []
    for uri in row['products']:
        name = uri.rsplit('/', 1)[-1]
        match = re.search(r'-s(\d{4})-(\d{16})-', name)
        if not match:
            errors.append(f'{name}: no sector in filename')
            continue
        path = output / 'fits' / name
        part = path.with_suffix('.part')
        try:
            if not path.exists():
                tic_parts = '/'.join(match.group(2)[i:i + 4] for i in range(0, 16, 4))
                url = f'{DOWNLOAD}/s{match.group(1)}/{tic_parts}/{name}'
                with requests.get(url, timeout=(20, 120), stream=True) as response:
                    response.raise_for_status()
                    with part.open('wb') as target:
                        for chunk in response.iter_content(1024 * 1024):
                            target.write(chunk)
                part.replace(path)
            with fits.open(path, memmap=False) as hdus:
                hdus.verify('exception')
                if int(hdus[0].header['TICID']) != int(row['tic_id']) or int(hdus[0].header['SECTOR']) != int(match.group(1)):
                    raise ValueError('FITS TIC/sector mismatch')
                checks = [h._calculate_checksum(h._calculate_datasum()) == h.header['CHECKSUM']
                          for h in hdus if 'CHECKSUM' in h.header]
                if not checks or not all(checks):
                    raise ValueError('FITS checksum missing or invalid')
            item = {**row, 'path': str(path), 'sha256': digest(path)}
            result = _extract(item)
            result.update(path=str(path), sha256=item['sha256'], mast_data_uri=uri,
                          fits_checksum_verified=True, sha256_origin='computed from MAST bytes',
                          attempts=errors)
            return result
        except Exception as error:
            part.unlink(missing_ok=True)
            errors.append(f'{name}: {type(error).__name__}: {error}')
    return {'tic_id': row['tic_id'], 'label': row['label'], 'group': row['group'],
            'partition': row['partition'], 'status': 'download_or_validation_failed',
            'error': '; '.join(errors), 'features': None}


def run(output: Path, phase: str, workers: int) -> None:
    manifest_path = output / 'manifest.json'
    if digest(manifest_path) != load(output / 'manifest.sha256.json')['sha256']:
        raise ValueError('Frozen manifest checksum mismatch.')
    manifest = load(manifest_path)
    if manifest['backend_sha256'] != code_fingerprint():
        raise ValueError('Backend changed after protocol freeze.')
    for source, expected in manifest['input_sha256'].items():
        if digest(Path(source)) != expected:
            raise ValueError(f'Frozen input changed: {source}')
    if phase == 'test' and not (output / 'model-lock.json').is_file():
        raise ValueError('Train and lock the model before accessing test data.')
    if phase == 'test':
        lock = load(output / 'model-lock.json')
        if lock['model_sha256'] != digest(output / 'research-estimator.joblib') or lock['manifest_sha256'] != digest(manifest_path):
            raise ValueError('Model or manifest differs from lock.')
    (output / 'fits').mkdir(exist_ok=True)
    checkpoints = output / 'checkpoints'
    checkpoints.mkdir(exist_ok=True)
    records = [r for r in manifest['records'] if r['partition'] == phase]
    pending = [r for r in records if not (checkpoints / f"{r['tic_id']}.json").exists()]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for row, result in zip(pending, pool.map(extract_source, pending, [output] * len(pending)), strict=True):
            save(checkpoints / f"{row['tic_id']}.json", result)
            print(phase, row['tic_id'], result['status'], flush=True)
    results = [load(checkpoints / f"{r['tic_id']}.json") for r in records]
    if phase == 'train':
        prior = [r for r in pq.read_table(PREVIOUS / 'feature-results.parquet').to_pylist()
                 if r['partition'] == 'train' and r['status'] == 'usable']
        extra = [load(EXPANSION / 'checkpoints' / f"{r['tic_id']}.json")
                 for r in load(EXPANSION / 'manifest.json')['records'] if r['partition'] == 'train']
        extra = [r for r in extra if r['status'] == 'usable']
        usable = prior + extra + [r for r in results if r['status'] == 'usable']
        if {r['group'] for r in usable} & {r['group'] for r in manifest['records'] if r['partition'] == 'test'}:
            raise ValueError('Training and test source groups overlap.')
        model = RandomForestClassifier(n_estimators=200, max_depth=8, class_weight='balanced',
                                       random_state=42, n_jobs=1)
        model.fit(np.array([r['features'] for r in usable]), [r['label'] for r in usable])
        joblib.dump(model, output / 'research-estimator.joblib')
        save(output / 'model-lock.json', {'model_sha256': digest(output / 'research-estimator.joblib'),
            'manifest_sha256': digest(manifest_path), 'previous_train_usable': len(prior),
            'earlier_expansion_train_usable': len(extra),
            'new_train_statuses': dict(Counter(r['status'] for r in results)),
            'training_support': dict(Counter(r['label'] for r in usable))})
        print('Model frozen', len(usable), flush=True)
    else:
        lock = load(output / 'model-lock.json')
        if lock['model_sha256'] != digest(output / 'research-estimator.joblib') or lock['manifest_sha256'] != digest(manifest_path):
            raise ValueError('Model or manifest differs from lock.')
        model = joblib.load(output / 'research-estimator.joblib')
        usable = [r for r in results if r['status'] == 'usable']
        summary: dict[str, Any] = {'selected': len(results), 'statuses': dict(Counter(r['status'] for r in results)),
            'support': dict(Counter(r['label'] for r in usable)), 'model_lock': lock,
            'limits': manifest['limits'], 'records': results}
        if usable:
            actual = [r['label'] for r in usable]
            predicted = model.predict(np.array([r['features'] for r in usable]))
            summary['classification_report'] = classification_report(actual, predicted, labels=list(LABELS), output_dict=True, zero_division=0)
            summary['confusion_matrix'] = confusion_matrix(actual, predicted, labels=list(LABELS)).tolist()
            for row, prediction in zip(usable, predicted, strict=True):
                row['prediction'] = str(prediction)
        save(output / 'test-report.json', summary)
        print('Test:', summary['statuses'], summary['support'], flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare', 'train', 'test'])
    parser.add_argument('output', type=Path)
    parser.add_argument('--workers', type=int, default=4, choices=range(1, 5))
    args = parser.parse_args()
    if args.phase == 'prepare':
        prepare(args.output.resolve())
    else:
        if any(os.environ.get(n) != '1' for n in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')):
            raise ValueError('Set BLAS and OpenMP thread counts to 1.')
        run(args.output.resolve(), args.phase, args.workers)


if __name__ == '__main__':
    main()
