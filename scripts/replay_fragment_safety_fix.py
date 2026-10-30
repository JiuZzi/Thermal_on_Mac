#!/usr/bin/env python3
"""Hash-verified replay of frozen searches after simple-path validation repair."""

import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'pytorch-CycleGAN-and-pix2pix'))
from models.fragment_completion import FragmentSettings,sanitize_cached_fragments
from models.contour_completion import _render_paths
import preview_fragment_completion as preview
from preview_contour_completion import _image01
from semantic_preview.checkpoint import sha256_file
from preview_semantics import write_json

SOURCE=ROOT/'runs/edge_audit/fragment_completion_seed42_v2'
OUTPUT=ROOT/'runs/edge_audit/fragment_completion_seed42_v3'
VALIDATION=ROOT/'runs/edge_audit/fragment_validation_20261030'


def load_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def render_cached_R(E,metadata):
    """Same renderer/float arithmetic as prioritize_completion; R is unchanged."""
    candidates=metadata['candidates'];width=metadata['settings']['soft_width']
    for c in candidates:
        gain=c.get('importance_gain')
        if not isinstance(gain,(int,float)) or not np.isfinite(gain) or not 0<=gain<=1:
            raise ValueError('A validated cached path importance gain is required')
    _,C_added=_render_paths(E.shape,candidates,width)
    weighted=[dict(c,strength=c['strength']*c['importance_gain']) for c in candidates]
    _,R_added=_render_paths(E.shape,weighted,width)
    before=np.maximum(E,C_added);after=np.maximum(E,R_added)
    if (after<E).any() or (after>before).any():raise AssertionError('Invalid R replay bound')
    old_mass=float((before-E).sum(dtype=np.float64));new_mass=float((after-E).sum(dtype=np.float64))
    metadata['importance']['uniform_mean_control_gain']=min(1.,max(0.,new_mass/old_mass)) if old_mass>0 else 1.
    return after


def replay(args):
    if args.output_dir.exists():raise FileExistsError('Preserve existing output')
    manifest=load_json(args.source_dir/'run_manifest.json');source_plan=manifest['plan']
    if load_json(args.source_dir/'status.json').get('phase')!='completed':raise ValueError('Source search is incomplete')
    snapshots={'pytorch-CycleGAN-and-pix2pix/models/fragment_completion.py':VALIDATION/'fragment_before_simple_path_fix.py',
               'scripts/preview_fragment_completion.py':VALIDATION/'launcher_before_simple_path_fix.py'}
    for name,expected in source_plan['code_sha256'].items():
        path=snapshots.get(name,ROOT/name)
        if sha256_file(path)!=expected:raise ValueError(f'Frozen search source identity changed: {name}')
    hashes={}
    for row in source_plan['sources']:
        preview.verify_source_files(row)
        prefix=row['prefix'];paths=dict(maps=args.source_dir/'maps'/f'{prefix}_fragments.npz',
            fragments=args.source_dir/'diagnostics'/f'{prefix}_fragment.json',
            rematch=args.source_dir/'diagnostics'/f'{prefix}_rematch.json',
            baseline=preview.IMPORTANCE/'diagnostics'/f'{prefix}_paths.json')
        hashes[prefix]={key:sha256_file(path) for key,path in paths.items()}
    provenance=dict(format='simple_path_search_cache_replay_v1',entry_source=str(Path(__file__).relative_to(ROOT)),
        entry_sha256=sha256_file(Path(__file__)),source_directory=str(args.source_dir.resolve()),
        source_manifest_sha256=sha256_file(args.source_dir/'run_manifest.json'),file_sha256=hashes,
        original_recipe_snapshots={name:dict(path=str(path.relative_to(ROOT)),sha256=sha256_file(path)) for name,path in snapshots.items()},
        changes='Reject coordinate revisits, recompute C/matching; reuse unchanged path-only R; fresh full-search equivalence on fixed frame 01',
        controls='All 12 erased cases recomputed fresh, at the unchanged frozen sites')
    settings=FragmentSettings(**source_plan['settings'])
    launch=argparse.Namespace(completion_dir=preview.SOURCE,semantic_preview_dir=preview.SEMANTICS,
        clarity_dir=preview.CLARITY,importance_dir=preview.IMPORTANCE,weak_dir=preview.WEAK,output_dir=args.output_dir,
        external_samples=6,erasure_samples=12,annotations=None,zoom=6,
        max_anchors=settings.max_anchors,max_searches=settings.max_searches)
    if len(source_plan['sources'])!=52 or len(source_plan['controls'])!=12:raise ValueError('This repair expects the complete fixed review pack')
    equivalents=[]
    def builder(data,current_settings,prefix):
        if asdict(current_settings)!=asdict(settings):raise ValueError('Replay search settings must stay frozen')
        gray,E,_,_,_,_,_,_,ridges=data
        paths=dict(maps=args.source_dir/'maps'/f'{prefix}_fragments.npz',fragments=args.source_dir/'diagnostics'/f'{prefix}_fragment.json',
            rematch=args.source_dir/'diagnostics'/f'{prefix}_rematch.json',baseline=preview.IMPORTANCE/'diagnostics'/f'{prefix}_paths.json')
        for key,path in paths.items():
            if sha256_file(path)!=hashes[prefix][key]:raise ValueError('Cached data changed after preflight')
        with np.load(paths['maps'],allow_pickle=False) as z:maps={key:z[key].copy() for key in z.files if key not in ('TIR_uint8','trace_ridges')}
        with np.load(paths['maps'],allow_pickle=False) as z:
            np.testing.assert_array_equal(z['TIR_uint8'],gray);np.testing.assert_array_equal(z['observed'],E)
            np.testing.assert_array_equal(z['trace_ridges'],ridges)
        base=load_json(paths['baseline']);cached=load_json(paths['fragments']);rematch=load_json(paths['rematch'])
        clean=sanitize_cached_fragments(_image01(gray),E,ridges,base,cached)
        probe=copy.deepcopy(clean);n=len(base['candidates'])
        for c in probe['candidates'][n:]:c['strength']=c['observation_amplitude']
        probe['fragment_completion']['encoding']='observation_amplitude_probe'
        maps['fragment']=render_cached_R(E,clean);maps['observation']=render_cached_R(E,probe)
        mask=np.zeros(E.shape,bool)
        for c in clean['candidates'][n:]:
            if c['reliability_accepted']:
                points=np.asarray(c['path_yx'],dtype=int)[1:-1];mask[tuple(points.T)]=True
        maps['fragment_mask']=mask
        records=dict(baseline=base,fragment=clean,observation=probe,rematch=rematch)
        if prefix=='01':
            fresh,fresh_records=preview.apply_arms(data,current_settings)
            for name in ('baseline','fragment','observation','rematch','fragment_mask','rematch_mask','weak_ridges'):
                np.testing.assert_array_equal(fresh[name],maps[name],err_msg='Fresh/replay mismatch: '+name)
            assert fresh_records['fragment']['candidates']==records['fragment']['candidates']
            assert fresh_records['observation']['candidates']==records['observation']['candidates']
            equivalents.append(dict(prefix=prefix,all_float_conditions_bitwise_equal=True,all_candidate_C_R_and_flags_equal=True))
            write_json(VALIDATION/'fresh_replay_equivalence_01.json',equivalents[-1])
        return maps,records
    preview.run(launch,cached_arm_builder=builder,replay_provenance=provenance)
    if not equivalents:raise AssertionError('Required fresh computation was not verified')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir',type=Path,default=SOURCE);p.add_argument('--output-dir',type=Path,default=OUTPUT)
    replay(p.parse_args())
