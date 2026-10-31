#!/usr/bin/env python3
"""Audit frozen inputs, staged geometry, source records and float previews."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'pytorch-CycleGAN-and-pix2pix'))
from models.region_completion import ARMS, path_key
from models.completion_importance import _validate_candidate, prioritize_completion
from models.contour_completion import completion_comparison
from models.fragment_completion import _old_locks, _conflict, _occupy
from semantic_preview.checkpoint import sha256_file
from preview_region_completion import OUTPUT, IMPORTANCE, read
from preview_completion_recall import load_input
from preview_contour_completion import _image01


def _check_paths(record, base, ridges, label):
    n = len(base['candidates'])
    assert record['candidates'][:n] == base['candidates'], (label, 'original C/R')
    assert record['endpoints'][:len(base['endpoints'])] == base['endpoints'], (label, 'original anchors')
    occupied, painted, diagonals = _old_locks(base, ridges.shape)
    keys = set(); mask = np.zeros(ridges.shape, bool)
    counts = Counter()
    for c in record['candidates'][n:]:
        p = _validate_candidate(c, ridges.shape)
        key = path_key(p)
        assert key not in keys, (label, 'duplicate geometry'); keys.add(key)
        assert len(np.unique(p, axis=0)) == len(p) and not ridges[tuple(p[1:-1].T)].any(), (label, 'non-simple/crossed original')
        assert p[[0,-1]].tolist() == [record['endpoints'][i]['point'] for i in c['endpoints']]
        r = c['region_evidence']
        assert all(np.isfinite(v) for v in r.values()) and 0 <= r['support'] <= 1 and r['noise_floor'] > 0
        if not r['available']: assert r['support'] == 0
        if r['support'] > 0:
            assert r['signed_improvement'] > 0
            assert r['null_loss'] > r['split_loss'] + r['complexity_penalty']
        assert all(np.isfinite(c[k]) and 0 <= c[k] <= 1 for k in ('reliability_score','strength','importance_gain','add_importance'))
        if record['region_completion']['arm'] != 'dedup' and c['reliability_rejections']:
            assert not c['geometry_selected'], (label, 'rejected candidate occupied site')
        if c['geometry_selected']:
            assert _conflict(p, occupied, painted, diagonals) is None, (label, 'selected crossing/site reuse')
            _occupy(p, occupied, painted, diagonals)
        if c['reliability_accepted']:
            counts['accepted'] += 1
            counts[c.get('evidence_route','legacy_C')] += 1
            assert not c['reliability_rejections']
            if c.get('evidence_route') == 'regional_gap':
                assert r['available'] and r['support'] >= record['region_completion']['settings']['minimum_region_support']
                assert c['anchor_evidence']['two_measured_anchors']
            mask[tuple(p[1:-1].T)] = True
    assert counts['accepted'] == record['region_completion']['new_accepted']
    assert record['region_completion']['parameters_learned'] == 0
    return mask, counts


def verify(output):
    manifest = read(output/'run_manifest.json'); plan = manifest['plan']; summary = manifest['summary']
    status = read(output/'status.json')
    assert status['phase'] == 'completed' and status['completed'] == len(manifest['frames'])
    assert read(output/'frozen_plan.json') == plan
    canonical = hashlib.sha256(json.dumps(plan, sort_keys=True, allow_nan=False).encode()).hexdigest()
    assert canonical == summary['frozen_plan_canonical_sha256']
    assert plan['fixed_before_processing'] and not plan['test_split_access']
    assert plan['learned_parameters_added'] == 0 and not plan['D_training_changed']
    for name, expected in plan['code_sha256'].items():
        path = (ROOT/name).resolve()
        assert path.is_relative_to(ROOT) and sha256_file(path) == expected, ('source recipe', name)
    source = Path(plan['source_fragment_directory']); weak = Path(plan['source_weak_directory'])
    assert sha256_file(source/'run_manifest.json') == plan['source_identity']['fragment_manifest_sha256']
    source_manifest = read(source/'run_manifest.json')
    assert plan['controls'] == [c for c in source_manifest['plan']['controls'] if c['prefix'] in {r['prefix'] for r in plan['sources']}]
    source_frames = {r['prefix']: r for r in plan['sources']}
    dataset_counts = defaultdict(Counter); targets = []; frame_checks = []
    for f in manifest['frames']:
        prefix = f['prefix']; row = source_frames[prefix]
        if row['dataset'] == 'FLIR-development':
            for key, path in row['paths'].items(): assert sha256_file(path) == row['sha256'][key]
        else: assert sha256_file(row['path']) == row['sha256']
        paths = dict(fragment=source/'diagnostics'/f'{prefix}_fragment.json', fragment_maps=source/'maps'/f'{prefix}_fragments.npz',
            weak=weak/'diagnostics'/f'{prefix}_paths.json', weak_maps=weak/'maps'/f'{prefix}_recall.npz', baseline=IMPORTANCE/'diagnostics'/f'{prefix}_paths.json')
        for name, path in paths.items(): assert sha256_file(path) == plan['source_identity']['files'][prefix][name]
        base = read(paths['baseline'])
        raw_pools = dict(fragment=read(paths['fragment']), weak=read(paths['weak']))
        audit = read(output/'diagnostics'/f'{prefix}_groups.json')
        recovered = set(); by_key = {}
        for g in audit['groups']:
            key = path_key(g['canonical_path_yx']); assert key not in by_key; by_key[key] = g
            assert list(map(list, key)) == g['canonical_path_yx']
            for member in g['members']:
                pool, index = member['pool'], member['proposal_id']; pair = (pool,index)
                assert pair not in recovered; recovered.add(pair)
                previous = raw_pools[pool]['candidates'][index]
                canonical_path = g['canonical_path_yx']
                reconstructed = canonical_path[::-1] if member['path_reversed_from_canonical'] else canonical_path
                assert reconstructed == previous['path_yx']
                for name, value in previous.items():
                    if name in ('path_yx','endpoints'): continue
                    assert member[name] == value, (prefix,pair,name)
                assert [audit['proposal_endpoints'][i]['point'] for i in member['endpoints']] == [raw_pools[pool]['endpoints'][i]['point'] for i in previous['endpoints']]
        n = len(base['candidates'])
        expected = {(pool,c['id']) for pool, record in raw_pools.items() for c in record['candidates'][n:]}
        assert recovered == expected, (prefix,'lost source metadata')
        with np.load(output/'maps'/f'{prefix}_regions.npz', allow_pickle=False) as z:
            maps = {k:z[k].copy() for k in z.files}
        E, ridges = maps['observed'], maps['trace_ridges']
        for name, value in maps.items():
            assert np.isfinite(value).all()
            if name != 'TIR_uint8': assert ((value >= 0)&(value <= 1)).all(), (prefix,name)
        for folder, filename, names in ((source, f'{prefix}_fragments.npz', dict(observed='observed',baseline='baseline',fragment='fragment',trace_ridges='trace_ridges',TIR_uint8='TIR_uint8')),
            (weak, f'{prefix}_recall.npz', dict(weak='recall'))):
            with np.load(folder/'maps'/filename, allow_pickle=False) as z:
                for key, previous in names.items(): np.testing.assert_array_equal(maps[key], z[previous])
        for arm in ARMS:
            record = read(output/'diagnostics'/f'{prefix}_{arm}.json')
            mask, counts = _check_paths(record, base, ridges, (prefix,arm))
            np.testing.assert_array_equal(mask, maps[arm+'_mask'])
            assert np.all(maps[arm] >= maps['baseline'])
            assert counts['accepted'] == f[arm+'_accepted']
            for c in record['candidates'][n:]:
                group = by_key[path_key(c['path_yx'])]
                assert c['region_evidence'] == group['region_evidence']
                refs = {(m['pool'],m['id']) for m in c['source_members']}
                assert refs <= {(m['pool'],m['proposal_id']) for m in group['members']}
                # R is reused only on the representative's unchanged path/context.
                previous = raw_pools[c['pool']]['candidates'][c['proposal_id']]
                for key in ('strength','importance_gain','add_importance','importance_features'):
                    assert c[key] == previous[key], (prefix,arm,'amplitude/R',key)
            dataset_counts[row['dataset']].update({arm+'_accepted':counts['accepted'], arm+'_regional_gap':counts['regional_gap']})
        yy,xx = np.indices(E.shape)
        for t in manifest['targets']:
            if row['dataset'] != 'FLIR-development' or t['filename'] != row['filename']: continue
            x0,y0,x1,y1 = t['box']; box = (xx+.5>=x0)&(xx+.5<x1)&(yy+.5>=y0)&(yy+.5<y1)
            item = dict(id=t['id'], category=t['category'])
            for arm in ('fragment','weak',*ARMS):
                item[arm+'_visible'] = int(((maps[arm]-maps['baseline']>=1/255)&box).sum())
            targets.append(item)
        frame_checks.append(dict(prefix=prefix, input_and_source_provenance=True, all_arm_paths_and_float_maps=True))
    control_results = defaultdict(list)
    for control in plan['controls']:
        name = control['id']
        with np.load(output/'controls'/f'{name}.npz', allow_pickle=False) as z: maps = {k:z[k].copy() for k in z.files}
        data = load_input(source_frames[control['prefix']])
        gray, E, _, _, structure, clutter, p, q, original_ridges = data
        np.testing.assert_array_equal(maps['TIR_uint8'], gray)
        expected_E = E.copy(); expected_E[maps['erased']] = 0
        np.testing.assert_array_equal(maps['observed'], expected_E)
        old, original_control = completion_comparison(_image01(gray), expected_E,
            trace_ridges=original_ridges & ~maps['erased'])
        old_maps, control_base = prioritize_completion(expected_E, original_control, structure, clutter, p, q)
        np.testing.assert_array_equal(maps['baseline'], old_maps['priority'])
        # Compare through the same JSON representation as the persisted record;
        # original trace_yx contains tuples before export, lists after export.
        control_base = json.loads(json.dumps(control_base, allow_nan=False))
        for folder, kind, previous_arm in ((source,'fragment','fragment'),(weak,'weak','recall')):
            path = folder/'controls'/f'{name}.npz'
            assert sha256_file(path) == plan['source_identity']['controls'][name][kind]
            with np.load(path, allow_pickle=False) as z:
                for key in ('TIR_uint8','truth','erased','observed','baseline'): np.testing.assert_array_equal(maps[key],z[key])
                np.testing.assert_array_equal(maps[kind],z[previous_arm])
        for arm in ARMS:
            stats = next(r for r in manifest['controls'] if r['id'] == name and r['arm'] == arm)
            record = read(output/'controls'/f'{name}_{arm}.json')
            mask, _ = _check_paths(record, control_base, old['trace_ridges'], (name,arm))
            np.testing.assert_array_equal(mask, maps[arm+'_mask'])
            assert np.isfinite(maps[arm]).all() and np.all(maps[arm] >= maps['baseline']) and np.all(maps[arm] <= 1)
            control_results[arm].append(stats['recall_visible_coverage_1px'])
    report = dict(format='regional_completion_verification_v1', verified=True, frames=len(frame_checks), targets=len(targets),
        controls=len(plan['controls']), source_provenance_preserved=True, observed_and_old_C_R_preserved=True,
        exact_duplicate_sources_reconstructable=True, selected_paths_simple_and_nonintersecting=True,
        parameters_learned=0, D_training_changed=False, independent_accuracy_verified=False,
        dataset_counts={k:dict(v) for k,v in dataset_counts.items()},
        target_boxes_with_visible_additions={arm:sum(r[arm+'_visible']>0 for r in targets) for arm in ('fragment','weak',*ARMS)},
        target_categories={category:{arm:sum(r[arm+'_visible']>0 for r in targets if r['category']==category) for arm in ('fragment','weak',*ARMS)} for category in sorted({r['category'] for r in targets})},
        erasure_visible_coverage_mean={k:float(np.mean(v)) for k,v in control_results.items()},
        plan_sha256=canonical, verifier_sha256=sha256_file(Path(__file__)), frame_checks=frame_checks)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=OUTPUT)
    p.add_argument('--report',type=Path)
    args = p.parse_args()
    report = verify(args.output_dir)
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='frame_checks'},ensure_ascii=False,indent=2))
