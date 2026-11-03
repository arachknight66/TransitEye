"""Train an isolated, uncalibrated three-class development estimator.

This is not an application model or an independent evaluation. Existing reserved
source groups are excluded using split metadata; their observations are not opened.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import logging
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

from transiteye.characterization.fitting import refine_candidate
from transiteye.classification.models import MORPHOLOGY_FEATURE_NAMES
from transiteye.detection.peaks import find_candidates
from transiteye.features.extractors import extract_features, stellar_temperature
from transiteye.io.fits import read_tess_light_curve
from transiteye.preprocessing.detrend import two_pass_detrend
from transiteye.preprocessing.representation import prepare_light_curve
from transiteye.provenance import code_fingerprint

LOGGER = logging.getLogger(__name__)
LABELS = ('transit_like', 'eclipse_like', 'stellar_variability')
SPLITS = ('real_catalog_cohort.v1', 'real_catalog_cohort.v2',
          'real_catalog_cohort.v3', 'real_catalog_tess_ebs.v1')
INPUTS = (('transit_like', 'confirmed_transiting_hosts_v1'),
          ('eclipse_like', 'eclipse_like_gaia_v1'),
          ('stellar_variability', 'stellar_variability_gaia_v1'),
          ('transit_like', 'confirmed_transiting_hosts_v2'),
          ('eclipse_like', 'eclipse_like_gaia_v2'),
          ('stellar_variability', 'stellar_variability_gaia_v2'))


def prepare(repo: Path, output: Path) -> dict[str, Any]:
    """Freeze source selections without opening reserved split observations or labels."""
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('output must be empty when freezing selection')
    output.mkdir(parents=True, exist_ok=True)
    assignments: dict[str, set[str]] = defaultdict(set)
    split_hashes = {}
    for split in SPLITS:
        path = repo / 'manifests' / f'{split}.splits.json'
        raw = path.read_bytes()
        split_hashes[split] = hashlib.sha256(raw).hexdigest()
        for row in json.loads(raw)['assignments']:
            assignments[row['source_group_id']].add(row['split'])
    catalog_dir = repo / '.transiteye-workspace' / 'data'
    gaia_path = catalog_dir / 'gaia_variable_acquisition' / 'tic-gaia-dr2.json'
    gaia_ids = {r['tic_id']: r['gaia_dr2'] for r in json.loads(gaia_path.read_text())}
    candidates: list[dict[str, Any]] = []
    selection_hashes = {}
    for label, folder in INPUTS:
        path = catalog_dir / folder / 'selection.json'
        raw = path.read_bytes()
        selection_hashes[folder] = hashlib.sha256(raw).hexdigest()
        for row in json.loads(raw)['records']:
            tic = str(row['tic_id'])
            candidates.append({'tic_id': tic, 'source_group': f'tic-{tic}',
                'gaia_dr2': str(row.get('gaia_dr2') or gaia_ids.get(tic) or ''),
                'label': label, 'path': str((catalog_dir / folder / 'fits' / row['filename']).resolve()),
                'sha256': row['sha256'], 'catalog_period_days': row.get('catalog_period_days')
                if label == 'transit_like' else row.get('gaia_period_days'),
                'catalog_epoch_bjd_tdb': row.get('catalog_epoch_bjd_tdb') if label == 'transit_like' else None})
    tic_classes: dict[str, set[str]] = defaultdict(set)
    gaia_classes: dict[str, set[str]] = defaultdict(set)
    for row in candidates:
        tic_classes[row['tic_id']].add(row['label'])
        if row['gaia_dr2']:
            gaia_classes[row['gaia_dr2']].add(row['label'])
    records, excluded = [], []
    used_gaia: set[str] = set()
    for row in candidates:
        tic, gaia = row['tic_id'], row['gaia_dr2']
        prior = assignments.get(row['source_group'], set())
        if len(tic_classes[tic]) > 1 or (gaia and len(gaia_classes[gaia]) > 1):
            reason = 'conflicting source-level catalog classes'
        elif any(part != 'train' for part in prior):
            reason = 'source has an existing non-training partition assignment'
        elif gaia and gaia in used_gaia:
            reason = 'duplicate Gaia source'
        else:
            reason = ''
        if reason:
            excluded.append({'tic_id': tic, 'label': row['label'], 'reason': reason})
            continue
        if gaia:
            used_gaia.add(gaia)
        group = f'gaia-dr2-{gaia}' if gaia else row['source_group']
        fold = 'train' if prior else ('validation' if int(hashlib.sha256(
            f'transiteye-three-class-v1:{group}'.encode()).hexdigest()[:8], 16) % 5 == 0 else 'train')
        records.append({**row, 'group': group, 'partition': fold,
                        'existing_training_group': bool(prior)})
    manifest = {'schema_version': '1', 'labels': LABELS,
        'scope': 'three-class uncalibrated research estimator, not an application ModelBundle',
        'selection_hashes': selection_hashes, 'split_manifest_hashes': split_hashes,
        'backend_sha256': code_fingerprint(), 'records': records, 'excluded': excluded,
        'validation_limit': 'already opened, selected sources; not an independent test set'}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    return manifest


def _period_matches(found: float, catalog: float) -> bool:
    return any(abs(found / (catalog * factor) - 1) <= 0.02 for factor in (0.5, 1.0, 2.0))


def _extract(row: dict[str, Any]) -> dict[str, Any]:
    """Blindly extract features, then use catalog ephemerides only to check label match."""
    result: dict[str, Any] = {'tic_id': row['tic_id'], 'label': row['label'],
        'partition': row['partition'], 'group': row['group'], 'status': 'error', 'error': None,
        'features': None}
    try:
        path = Path(row['path'])
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(chunk)
        if h.hexdigest() != row['sha256']:
            raise ValueError('input SHA-256 mismatch')
        lc = read_tess_light_curve(path)
        if lc.provenance.target_id != row['tic_id']:
            raise ValueError('input target mismatch')
        prepared = prepare_light_curve(lc)
        dual = two_pass_detrend(prepared, baseline_window_days=0.75, device='cpu')
        det = find_candidates(dual, max_candidates=1, snr_threshold=1.5, device='cpu',
                              min_period=0.3, max_period=5.0)
        if not det.candidates:
            result.update(status='no_candidate', error='no candidate at SNR threshold 1.5')
            return result
        cand = det.candidates[0]
        if row['catalog_period_days'] is not None and not _period_matches(
                float(cand.period_days or 0), float(row['catalog_period_days'])):
            result.update(status='catalog_mismatch', error='candidate period does not match catalog or half/double alias')
            return result
        if row['label'] == 'transit_like':
            if lc.provenance.time_reference is None or cand.epoch_days is None:
                result.update(status='catalog_mismatch', error='missing transit time reference')
                return result
            period = float(row['catalog_period_days'])
            epoch = float(row['catalog_epoch_bjd_tdb']) - float(lc.provenance.time_reference)
            delta = abs(((float(cand.epoch_days) - epoch + period / 2) % period) - period / 2)
            if delta > max(0.5 / 24, float(cand.duration_hours or 0) / 48):
                result.update(status='catalog_mismatch', error='candidate epoch does not match catalog transit phase')
                return result
        events = ((float(cand.epoch_days or 0), float(cand.period_days or 1),
                   float(cand.duration_hours or 2)),)
        dual_final = two_pass_detrend(prepared, candidate_events=events, device='cpu')
        crowdsap_str = lc.metadata.get('crowdsap')
        crowdsap = float(crowdsap_str) if crowdsap_str and crowdsap_str != 'None' else None
        refined = refine_candidate(cand, dual_final.prepared.time_days, dual_final.detrended_flux,
            dual_final.prepared.flux_error_normalized, crowdsap=crowdsap,
            batman_snr_threshold=7.0, teff=stellar_temperature(lc))
        features = extract_features(dual_final, refined, det.variability, lc)
        vector = [float(features.get(name, 0.0)) for name in MORPHOLOGY_FEATURE_NAMES]
        if not np.all(np.isfinite(vector)):
            raise ValueError('non-finite feature vector')
        result.update(status='usable', features=vector)
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    return result


def run(output: Path, workers: int = 4) -> dict[str, Any]:
    """Extract features, count every exclusion, and train a standalone estimator."""
    raw = (output / 'manifest.json').read_bytes()
    manifest = json.loads(raw)
    if manifest['backend_sha256'] != code_fingerprint():
        raise ValueError('backend differs from frozen manifest')
    if not 1 <= workers <= 4:
        raise ValueError('worker count must be between 1 and 4')
    records = manifest['records']
    checkpoints = output / 'feature-checkpoints'
    checkpoints.mkdir(exist_ok=True)
    results = []
    missing = []
    for row in records:
        path = checkpoints / f"{row['tic_id']}.json"
        if path.exists():
            results.append(json.loads(path.read_text()))
        else:
            missing.append(row)
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        future_map = {pool.submit(_extract, row): row for row in missing}
        for done in concurrent.futures.as_completed(future_map):
            row = future_map[done]
            try:
                result = done.result()
            except Exception as exc:
                result = {'tic_id':row['tic_id'], 'label':row['label'], 'partition':row['partition'],
                          'group':row['group'], 'status':'error', 'features':None,
                          'error':f'worker: {type(exc).__name__}: {exc}'}
            path = checkpoints / f"{row['tic_id']}.json"
            temp = path.with_suffix('.tmp')
            temp.write_text(json.dumps(result, allow_nan=False, sort_keys=True) + '\n')
            temp.replace(path)
            results.append(result)
            LOGGER.info('feature extraction %d/%d: %s %s', len(results), len(records),
                        row['tic_id'], result['status'])
    results.sort(key=lambda x: (x['label'], int(x['tic_id'])))
    pq.write_table(pa.Table.from_pylist(results), output / 'feature-results.parquet')
    train = [r for r in results if r['status']=='usable' and r['partition']=='train']
    validation = [r for r in results if r['status']=='usable' and r['partition']=='validation']
    summary: dict[str, Any] = {'manifest_sha256': hashlib.sha256(raw).hexdigest(),
        'backend_sha256': manifest['backend_sha256'], 'selected': len(records),
        'manifest_excluded': len(manifest['excluded']), 'statuses': dict(Counter(r['status'] for r in results)),
        'support': {p: dict(Counter(r['label'] for r in rows)) for p, rows in
                    (('train', train), ('validation', validation))},
        'feature_names': list(MORPHOLOGY_FEATURE_NAMES), 'labels': LABELS,
        'limitations': ['three classes only; other_artifact is untrained',
            'no independent test; source labels and localization are provisional',
            'uncalibrated; not compatible with the four-class application model',
            'candidate-period matching is label selection, never a model feature']}
    if {r['group'] for r in train} & {r['group'] for r in validation}:
        raise ValueError('source leakage between train and validation')
    if {r['label'] for r in train} != set(LABELS):
        summary['training_error'] = 'one or more classes lack usable training sources'
        (output/'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True)+'\n')
        return summary
    estimator = RandomForestClassifier(n_estimators=200, max_depth=8, class_weight='balanced',
                                       random_state=42, n_jobs=1)
    x_train = np.array([r['features'] for r in train])
    y_train = [r['label'] for r in train]
    groups = [r['group'] for r in train]
    if min(Counter(y_train).values()) >= 5:
        folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        cv_predictions = cross_val_predict(estimator, x_train, y_train, groups=groups, cv=folds, n_jobs=1)
        summary['grouped_cv'] = {'classification_report': classification_report(
            y_train, cv_predictions, labels=list(LABELS), output_dict=True, zero_division=0),
            'confusion_matrix': confusion_matrix(y_train, cv_predictions, labels=list(LABELS)).tolist(),
            'warning': 'development cross-validation on previously selected/opened sources, not independent'}
    else:
        summary['grouped_cv'] = {'error': 'insufficient examples per class for five-fold grouped CV'}
    estimator.fit(x_train, y_train)
    model_path = output / 'research-estimator.joblib'
    joblib.dump(estimator, model_path)
    summary['model_sha256'] = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if validation:
        y = [r['label'] for r in validation]
        prediction = estimator.predict(np.array([r['features'] for r in validation]))
        summary['exploratory_validation'] = {'classification_report': classification_report(
            y, prediction, labels=list(LABELS), output_dict=True, zero_division=0),
            'confusion_matrix': confusion_matrix(y, prediction, labels=list(LABELS)).tolist(),
            'warning': 'exploratory selected sources, not an independent test'}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run'))
    parser.add_argument('output', type=Path)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.action == 'prepare':
        result = prepare(args.repo.resolve(), args.output.resolve())
        LOGGER.info('manifest: %d selected, %d excluded', len(result['records']), len(result['excluded']))
    else:
        required = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')
        if any(os.environ.get(name) != '1' for name in required):
            raise ValueError('set OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, MKL_NUM_THREADS to 1')
        result = run(args.output.resolve(), args.workers)
        LOGGER.info('completed: %s', result['support'])


if __name__ == '__main__':
    main()
