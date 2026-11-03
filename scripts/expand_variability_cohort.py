"""Freeze, download and train a further source-separated Gaia variability expansion."""
from __future__ import annotations

import argparse
import hashlib
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import astropy.units as u
import joblib
import numpy as np
import pyarrow.parquet as pq
from astropy.coordinates import SkyCoord
from expand_research_cohort_large import digest, extract_source, load, save
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from transiteye.evaluation.three_class_training import LABELS
from transiteye.provenance import code_fingerprint

ROOT = Path.cwd()
DATA = ROOT / '.transiteye-workspace/data'
DISCOVERY = ROOT / '.transiteye-workspace/remediation/variable-discovery-20261102'
BASE = ROOT / '.transiteye-workspace/remediation/three-class-research-v3'
EXPANSIONS = [ROOT / '.transiteye-workspace/remediation/three-class-expansion-20261102-v2',
              ROOT / '.transiteye-workspace/remediation/three-class-larger-20261102']


def prepare(output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a fresh output directory.')
    output.mkdir(parents=True, exist_ok=True)
    paths = sorted(DATA.glob('*/selection.json'))
    split_paths = sorted((ROOT / 'manifests').glob('*.splits.json'))
    prior = [r for p in paths for r in load(p).get('records', [])]
    manifests = [load(p / 'manifest.json') for p in EXPANSIONS]
    prior.extend(r for m in manifests for r in m['records'])
    previous_manifest = load(BASE / 'manifest.json')
    prior.extend(previous_manifest['records'])
    excluded_tic = {str(r['tic_id']) for r in prior if r.get('tic_id')}
    excluded_gaia = {str(r['gaia_dr2']) for r in prior if r.get('gaia_dr2')}
    for p in split_paths:
        excluded_tic.update(str(r.get('source_group_id', '')).removeprefix('tic-')
                            for r in load(p).get('assignments', []))
    mapping = DATA / 'gaia_variable_acquisition/tic-gaia-dr2.json'
    for r in load(mapping):
        if str(r['tic_id']) in excluded_tic and r.get('gaia_dr2'):
            excluded_gaia.add(str(r['gaia_dr2']))
    planet = DATA / 'confirmed_transiting_hosts_v2/catalog-time-systems.json'
    excluded_tic.update(str(r.get('tic_id', '')).replace('TIC ', '').strip() for r in load(planet)['rows'])
    eb = DATA / 'eclipse_like_gaia_v2/candidate-inventory.json'
    for r in load(eb):
        excluded_tic.add(str(r['tic_id']))
        excluded_gaia.add(str(r['gaia_dr2']))
    discovery_files = [DISCOVERY / name for name in ('gaia.json', 'neighbors.json', 'tic.json', 'products.json', 'inventory.json', 'discover.py', 'discover_fast.py')]
    positions = {r['dr3']: r for r in load(DISCOVERY / 'gaia.json')['rows']}
    staged = {p.name for p in DATA.glob('*/staging/*.fits')}
    selected = []
    for r in load(DISCOVERY / 'inventory.json')['rows']:
        if r['tic_id'] in excluded_tic or r['dr2'] in excluded_gaia:
            continue
        if not r['products'] or any(x.rsplit('/', 1)[-1] in staged for x in r['products']):
            continue
        gaia = positions[r['dr3']]
        separation = SkyCoord(gaia['ra']*u.deg, gaia['dec']*u.deg).separation(
            SkyCoord(r['ra']*u.deg, r['dec']*u.deg)).arcsec
        if separation > 1.0:
            continue
        selected.append({'tic_id': r['tic_id'], 'gaia_dr2': r['dr2'], 'gaia_dr3': r['dr3'],
            'gaia_tic_separation_arcsec': float(separation), 'gaia_class': r['class'],
            'gaia_score': r['score'], 'label': 'stellar_variability',
            'catalog_period_days': None, 'catalog_epoch_bjd_tdb': None,
            'products': sorted(r['products'])[:4], 'group': 'gaia-dr2-' + r['dr2']})
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        groups[row['gaia_dr2']].append(row)
    selected = [rows[0] for rows in groups.values() if len(rows) == 1]
    selected.sort(key=lambda r: hashlib.sha256(('variability-expansion-v1:' + r['gaia_dr2']).encode()).hexdigest())
    selected = selected[:150]
    if len(selected) < 20:
        raise ValueError(f'Only {len(selected)} qualified sources available; need at least 20.')
    test_count = max(4, len(selected)//5)
    for i, row in enumerate(selected):
        row['partition'] = 'test' if i < test_count else 'train'
    files = paths + split_paths + discovery_files + [planet, eb, mapping, BASE/'manifest.json',
        BASE/'feature-results.parquet', Path(__file__).resolve(), ROOT/'scripts/expand_research_cohort_large.py']
    for p, manifest in zip(EXPANSIONS, manifests, strict=True):
        files += [p/'manifest.json', p/'model-lock.json']
        files += [p/'checkpoints'/f"{r['tic_id']}.json" for r in manifest['records'] if r['partition']=='train']
    if previous_manifest['backend_sha256'] != code_fingerprint() or any(m['backend_sha256'] != code_fingerprint() for m in manifests):
        raise ValueError('Prior feature vectors do not match the current backend.')
    manifest = {'protocol':'variability-expansion-v1', 'backend_sha256':code_fingerprint(),
        'selection':'Gaia classifier >0.5; unique Gaia DR2 neighbor within 0.5 arcsec; unique TIC; '
            'Gaia DR3/TIC separation <=1 arcsec; exclude prior source identities, confirmed planet '
            'hosts and Gaia eclipsing-binary inventory; deterministic Gaia hash, first fifth held out',
        'input_sha256':{str(p.resolve()):digest(p) for p in files}, 'records':selected,
        'limits':['Gaia classifications are proxies; signal origin has not been adjudicated',
            'only stellar-variability examples added; test measures only this class',
            'possible unknown source aliases; catalog and archive selection biases',
            'three-class uncalibrated research model, not four-class application model']}
    save(output/'manifest.json', manifest)
    save(output/'manifest.sha256.json', {'sha256':digest(output/'manifest.json')})
    print('Frozen',len(selected),'sources:',len(selected)-test_count,'train,',test_count,'test',flush=True)


def run(output: Path, phase: str, workers: int) -> None:
    manifest_path = output/'manifest.json'
    if digest(manifest_path) != load(output/'manifest.sha256.json')['sha256']:
        raise ValueError('Frozen manifest checksum mismatch.')
    manifest=load(manifest_path)
    if manifest['backend_sha256'] != code_fingerprint():
        raise ValueError('Backend changed after selection.')
    for p, expected in manifest['input_sha256'].items():
        if digest(Path(p)) != expected:
            raise ValueError(f'Frozen input changed: {p}')
    if phase == 'test':
        if not (output/'model-lock.json').is_file():
            raise ValueError('Lock model before test access.')
        lock=load(output/'model-lock.json')
        if lock['manifest_sha256'] != digest(manifest_path) or lock['model_sha256'] != digest(output/'research-estimator.joblib'):
            raise ValueError('Locked model or manifest changed.')
    (output/'fits').mkdir(exist_ok=True)
    checkpoints=output/'checkpoints'
    checkpoints.mkdir(exist_ok=True)
    rows=[r for r in manifest['records'] if r['partition']==phase]
    pending=[r for r in rows if not (checkpoints/f"{r['tic_id']}.json").exists()]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for row,result in zip(pending,pool.map(extract_source,pending,[output]*len(pending)),strict=True):
            save(checkpoints/f"{row['tic_id']}.json",result)
            print(phase,row['tic_id'],result['status'],flush=True)
    results=[load(checkpoints/f"{r['tic_id']}.json") for r in rows]
    if phase=='train':
        prior=[r for r in pq.read_table(BASE/'feature-results.parquet').to_pylist() if r['partition']=='train' and r['status']=='usable']
        for folder in EXPANSIONS:
            m=load(folder/'manifest.json')
            for row in m['records']:
                if row['partition']=='train':
                    result=load(folder/'checkpoints'/f"{row['tic_id']}.json")
                    if result['status']=='usable':
                        prior.append(result)
        usable=prior+[r for r in results if r['status']=='usable']
        if len({r['group'] for r in usable}) != len(usable) or {r['group'] for r in usable} & {r['group'] for r in manifest['records'] if r['partition']=='test'}:
            raise ValueError('Duplicate or overlapping source groups.')
        model=RandomForestClassifier(n_estimators=200,max_depth=8,class_weight='balanced',random_state=42,n_jobs=1)
        model.fit(np.array([r['features'] for r in usable]),[r['label'] for r in usable])
        joblib.dump(model,output/'research-estimator.joblib')
        save(output/'model-lock.json',{'model_sha256':digest(output/'research-estimator.joblib'),
            'manifest_sha256':digest(manifest_path),'prior_usable':len(prior),
            'new_train_statuses':dict(Counter(r['status'] for r in results)),
            'training_support':dict(Counter(r['label'] for r in usable))})
        print('Model frozen on',len(usable),'sources',flush=True)
    else:
        lock=load(output/'model-lock.json')
        model=joblib.load(output/'research-estimator.joblib')
        usable=[r for r in results if r['status']=='usable']
        summary={'selected':len(results),'statuses':dict(Counter(r['status'] for r in results)),
            'support':dict(Counter(r['label'] for r in usable)),'model_lock':lock,
            'limits':manifest['limits'],'records':results}
        if usable:
            actual=[r['label'] for r in usable]
            predicted=model.predict(np.array([r['features'] for r in usable]))
            summary['classification_report']=classification_report(actual,predicted,labels=list(LABELS),output_dict=True,zero_division=0)
            summary['confusion_matrix']=confusion_matrix(actual,predicted,labels=list(LABELS)).tolist()
            for row,prediction in zip(usable,predicted,strict=True):
                row['prediction']=str(prediction)
        save(output/'test-report.json',summary)
        print('Test:',summary['statuses'],summary['support'],flush=True)


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['prepare','train','test'])
    p.add_argument('output',type=Path)
    p.add_argument('--workers',type=int,default=4,choices=range(1,5))
    args=p.parse_args()
    if args.phase=='prepare':
        prepare(args.output.resolve())
    else:
        if any(os.environ.get(n)!='1' for n in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')):
            raise ValueError('Set BLAS and OpenMP thread counts to 1.')
        run(args.output.resolve(),args.phase,args.workers)


if __name__=='__main__':
    main()
