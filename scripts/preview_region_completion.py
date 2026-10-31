#!/usr/bin/env python3
"""Three non-learning completion stages on frozen, hash-checked proposals."""

import argparse
import copy
import csv
from dataclasses import asdict
import hashlib
import html
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pytorch-CycleGAN-and-pix2pix'))
from models.region_completion import RegionSettings, ARMS, R_KEYS, completion_stages, path_key
from models.completion_importance import prioritize_completion
from models.contour_completion import completion_comparison
from models.fragment_completion import FragmentSettings, fragment_completion
from models.completion_recall import recall_completion, RecallSettings
from preview_completion_importance import preflight, SOURCE, SEMANTICS, CLARITY
from preview_completion_recall import IMPORTANCE, load_input, verify_source_files, source_identity, write_annotation_page
from preview_fragment_completion import weak_identity
from replay_fragment_safety_fix import render_cached_R
from preview_contour_completion import _image01, write_table
from preview_semantic_contours import make_sheet
from semantic_preview.checkpoint import sha256_file
from semantic_preview.review import uint8_image
from preview_semantics import write_json
from completion_recall_audit import erasure_mask, control_statistics, read_annotations, annotation_statistics

FRAGMENTS = ROOT / 'runs/edge_audit/fragment_completion_seed42_v3'
WEAK = ROOT / 'runs/edge_audit/completion_recall_seed42_v2'
OUTPUT = ROOT / 'runs/edge_audit/region_completion_seed42_v1'
LABELS = dict(baseline='Old C/R', fragment='Previous fragments', weak='Previous weak tracing',
    dedup='1 Exact dedup / old C', observed_control='C-first / no region route',
    region='2 Regional evidence / C', weak_region='Weak pool / same new C', joint='3 Joint pool / same new C')
GROUPS = dict(comparison=('fragment', 'region', 'joint'), dedup=('fragment', 'dedup', 'region'),
    complement=('weak_region', 'region', 'joint'), attribution=('dedup', 'observed_control', 'region'))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fragment-dir', type=Path, default=FRAGMENTS)
    p.add_argument('--weak-dir', type=Path, default=WEAK)
    p.add_argument('--output-dir', type=Path, default=OUTPUT)
    p.add_argument('--annotations', type=Path)
    p.add_argument('--limit', type=int, help='First N frozen frames for a functional smoke only; default uses all 52')
    p.add_argument('--zoom', type=int, default=6)
    args = p.parse_args(argv)
    if args.output_dir.exists(): p.error('Preserve previous outputs; choose a new directory')
    if args.limit is not None and not 1 <= args.limit <= 52: p.error('Limit must be 1..52')
    if not 1 <= args.zoom <= 12: p.error('Zoom must be 1..12')
    return args


def _frame_paths(args, prefix):
    return dict(fragment=args.fragment_dir/'diagnostics'/f'{prefix}_fragment.json',
        fragment_maps=args.fragment_dir/'maps'/f'{prefix}_fragments.npz',
        weak=args.weak_dir/'diagnostics'/f'{prefix}_paths.json',
        weak_maps=args.weak_dir/'maps'/f'{prefix}_recall.npz',
        baseline=IMPORTANCE/'diagnostics'/f'{prefix}_paths.json')


def preflight_all(args):
    prior = read(args.fragment_dir/'run_manifest.json')
    status = read(args.fragment_dir/'status.json')
    if status.get('phase') != 'completed' or status.get('completed') != len(prior['frames']):
        raise ValueError('Complete safe fragment preview required')
    if prior['summary'].get('truncated_frames') != 0:
        raise ValueError('Use the complete, untruncated frozen candidate pool')
    for name, digest in prior['plan']['code_sha256'].items():
        path = (ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or sha256_file(path) != digest:
            raise ValueError('Frozen proposal recipe changed: '+name)
    legacy_args = SimpleNamespace(completion_dir=SOURCE, semantic_preview_dir=SEMANTICS, clarity_dir=CLARITY,
        importance_dir=IMPORTANCE, weak_dir=args.weak_dir)
    original, development = preflight(legacy_args)
    rows = prior['plan']['sources']
    if rows[:len(development)] != development or len(rows) != 52 or len(development) != 40:
        raise ValueError('Require the same 40 development and 12 external fallback frames')
    importance_identity = source_identity(legacy_args, rows)
    weak_manifest, weak_source = weak_identity(legacy_args, rows)
    controls = prior['plan']['controls']
    if controls != weak_manifest['plan']['controls']:
        raise ValueError('Erasure sites must remain identical across mechanisms')
    if args.limit is not None:
        rows = rows[:args.limit]
        controls = [c for c in controls if c['prefix'] in {r['prefix'] for r in rows}]
    hashes = {}
    for row in rows:
        verify_source_files(row)
        hashes[row['prefix']] = {k: sha256_file(p) for k, p in _frame_paths(args, row['prefix']).items()}
    control_hashes = {c['id']: dict(
        fragment=sha256_file(args.fragment_dir/'controls'/f'{c["id"]}.npz'),
        weak=sha256_file(args.weak_dir/'controls'/f'{c["id"]}.npz')) for c in controls}
    return original, rows, controls, prior, dict(fragment_manifest_sha256=sha256_file(args.fragment_dir/'run_manifest.json'),
        weak=weak_source, importance=importance_identity, files=hashes, controls=control_hashes)


def apply_stages(data, fragments, weak, *, cached_baseline=None):
    gray, E, _, metadata, structure, clutter, p, q, ridges = data
    if cached_baseline is not None:
        if [{k:v for k,v in c.items() if k not in R_KEYS} for c in cached_baseline['candidates']] != metadata['candidates']:
            raise ValueError('Cached original C and current source differ')
        metadata = cached_baseline
    outputs, audit = completion_stages(_image01(gray), E, ridges, metadata, fragments, weak)
    n = len(metadata['candidates'])
    all_cached = all(all(k in c for k in R_KEYS) for record in outputs.values() for c in record['candidates'])
    if all_cached and 'importance' in metadata:
        base = copy.deepcopy(metadata)
    else:
        # Compute path-only R once on the union. No class enters C or the pool.
        _, joint = prioritize_completion(E, outputs['joint'], structure, clutter, p, q)
        gains = {path_key(c['path_yx']): {k:c[k] for k in R_KEYS} for c in joint['candidates'][n:]}
        base = copy.deepcopy(metadata); base['candidates'] = copy.deepcopy(joint['candidates'][:n])
        base['importance'] = copy.deepcopy(joint['importance'])
        for record in outputs.values():
            record['importance'] = copy.deepcopy(joint['importance'])
            record['candidates'][:n] = copy.deepcopy(base['candidates'])
            for c in record['candidates'][n:]: c.update(copy.deepcopy(gains[path_key(c['path_yx'])]))
    maps = dict(observed=E.copy(), baseline=render_cached_R(E, base))
    for arm, record in outputs.items():
        if record['candidates'][:n] != base['candidates']:
            raise AssertionError('Original C/R or accepted paths changed')
        maps[arm] = render_cached_R(E, record)
        mask = np.zeros(E.shape, bool)
        for c in record['candidates'][n:]:
            if c['reliability_accepted']:
                points = np.asarray(c['path_yx'], int)[1:-1]; mask[tuple(points.T)] = True
        maps[arm+'_mask'] = mask
        if not np.isfinite(maps[arm]).all() or (maps[arm] > 1).any() or (maps[arm] < maps['baseline']).any():
            raise AssertionError('Regional output must preserve the old baseline and remain bounded')
    return maps, outputs, base, audit


def save_sheets(output, prefix, gray, maps, *, folder='frames', crop=None, box=None, zoom=1):
    for group, modes in GROUPS.items():
        name = prefix+'.png' if folder == 'targets' and group == 'comparison' else prefix+'_'+group+'.png'
        make_sheet(gray, maps, crop=crop, box=box, zoom=zoom, modes=modes, labels=LABELS,
            importance_key='joint_mask', importance_caption='Joint new centers ONLY').save(output/folder/name)


def append_candidates(path, rows):
    if not rows:
        return
    exists = path.exists()
    with path.open('a', encoding='utf-8' if exists else 'utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        if not exists: writer.writeheader()
        writer.writerows(rows)


def write_page(output, frames, targets, controls, summary):
    cards = []
    for f in frames:
        prefix = f['prefix']
        crops = ''.join(f'<h4>{html.escape(t["id"])} / {html.escape(t["category"])}</h4><img loading="lazy" src="targets/{t["id"]}.png">'
            for t in targets if t['filename'] == f['filename'] and f['dataset'] == 'FLIR-development')
        cards.append(f'<article id="f{prefix}"><h2>{prefix} / {html.escape(f["dataset"])} / {html.escape(f["filename"])}</h2>'
            f'<p>新增接纳：去重 {f["dedup_accepted"]}；区域判断 {f["region_accepted"]}；弱边相同规则 {f["weak_region_accepted"]}；联合筛选 {f["joint_accepted"]}。条数不代表正确率。</p>'
            f'<img loading="lazy" src="frames/{prefix}_comparison.png">'
            f'<details><summary>步骤1、2：原片段法 → 只去重 → 区域判断</summary><img loading="lazy" src="frames/{prefix}_dedup.png"></details>'
            f'<details><summary>归因控制：去重 → 新观测决策与C优先 → 再允许区域证据补线</summary><img loading="lazy" src="frames/{prefix}_attribution.png"></details>'
            f'<details><summary>步骤3：两个候选池使用同一新规则，再联合筛选</summary><img loading="lazy" src="frames/{prefix}_complement.png"></details>'
            f'<details><summary>实际新增强度，固定[0,1]</summary><img loading="lazy" src="frames/{prefix}_increments.png"></details>'
            f'<details><summary>蓝色：观测追踪；橙色：区域证据补线；等亮仅定位</summary><img loading="lazy" src="frames/{prefix}_locator.png"></details>'
            f'<details><summary>原固定目标裁剪，框不参与算法</summary>{crops}</details>'
            f'<p><a href="diagnostics/{prefix}_joint.json">联合判断</a> · <a href="diagnostics/{prefix}_groups.json">全部原始来源及区域拟合</a></p></article>')
    erased = ''.join(f'<details><summary>{html.escape(c["id"])} / 原位置擦边辅助控制</summary><img loading="lazy" src="controls/{c["id"]}.png"><img loading="lazy" src="controls/{c["id"]}_crop.png"></details>' for c in controls)
    intro = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>区域证据补边：前三步对照</title>
<style>body{font:16px/1.7 system-ui;background:#f3f5f7;color:#182536;margin:18px}main{max-width:1800px;margin:auto}article,.notice{background:white;padding:18px;margin:18px 0;border:1px solid #ccd4df;border-radius:8px}.notice{background:#fff2d5}img{max-width:100%}h2{overflow-wrap:anywhere}summary{cursor:pointer;color:#155e91}nav a{padding:4px;display:inline-block}</style><main>
<h1>候选去重 → 区域证据可靠性 → 两类候选统一筛选</h1><div class="notice">
<p>主图依次：原TIR、上轮片段法、区域证据判断、联合弱边与片段候选、联合新增中心线定位。展开可查看单独去重及共同规则下的候选互补。所有灰度和红色叠加使用原[0,1]数值。</p>
<p>不增加网络权重。H0用一个平面解释局部红外，H1在连接两侧分别拟合平面；交换棋盘格训练／留出像素，按噪声计复杂度代价。样本不足或区域模型不支持时，可由直接观测规则判断，也可以不补。区域证据不确认物体身份，仍可能支持真实但不重要的道路纹理。</p>
<p>只合并完全相同的路径及其逆序，原始端口、来源、旧C/R均保留。区域组先判断可靠性再占用新连接位置，原有几何锁定不变。两组候选不直接合并接纳结果，模型分数未校准为正确概率。</p>
<p>40张FLIR是开发样本；外部12张只检查无语义结构回退。12个擦边控制保留TIR，仅删除观察边，不能当自然缺口精度或泛化证明。独立标注为空时不报告真实准确率；可使用<a href="annotate_gaps.html">原图缺口标记工具</a>。</p>
<p><a href="frame_statistics.csv">逐图</a> · <a href="video_statistics.csv">逐视频</a> · <a href="target_statistics.csv">原定位框</a> · <a href="candidate_statistics.csv">新候选</a> · <a href="erasure_controls.csv">同位置控制</a> · <a href="gap_statistics.csv">独立标记</a> · <a href="run_manifest.json">冻结计划与统计</a></p></div>'''
    nav = ' '.join(f'<a href="#f{f["prefix"]}">{f["prefix"]}</a>' for f in frames)
    (output/'index.html').write_text(intro+'<nav>'+nav+'</nav>'+''.join(cards)+'<article><h2>固定擦边控制</h2>'+erased+'</article><pre>'+html.escape(json.dumps(summary, ensure_ascii=False, indent=2))+'</pre></main></html>', encoding='utf-8')


def run(args):
    if args.output_dir.exists(): raise FileExistsError('Preserve previous outputs')
    torch.set_num_threads(2)
    source, rows, controls, prior, identity = preflight_all(args)
    annotations = read_annotations(args.annotations, {r['prefix'] for r in rows})
    settings = RegionSettings()
    code = dict(prior['plan']['code_sha256'])
    for path in (Path(__file__), ROOT/'pytorch-CycleGAN-and-pix2pix/models/region_completion.py'):
        code[str(path.relative_to(ROOT))] = sha256_file(path)
    plan = dict(format='regional_completion_review_v1', seed=42, settings=asdict(settings), code_sha256=code,
        sources=rows, controls=controls, source_identity=identity, source_fragment_directory=str(args.fragment_dir.resolve()),
        source_weak_directory=str(args.weak_dir.resolve()), fragment_search_settings=prior['plan']['settings'],
        weak_search_settings=asdict(RecallSettings()), fixed_before_processing=True, test_split_access=False,
        limit=args.limit, functional_subset_only=args.limit is not None, annotations=annotations,
        annotation_sha256=sha256_file(args.annotations) if args.annotations else None,
        arms=dict(dedup='Exact fragment geometry dedup; original C and matching',
            observed_control='Same new observation utility and C-first matching, but regional-gap route disabled',
            region='Same dedup pool; held-out region evidence, explicit routes, abstention and C-first matching',
            weak_region='Weak pool with the same regional rules', joint='Union of exact geometry hypotheses with the same regional rules'),
        learned_parameters_added=0, D_training_changed=False,
        limits='Uncalibrated proxy; no independent natural-gap accuracy or complete external semantic generalization')
    output = args.output_dir; output.mkdir(parents=True, exist_ok=False)
    for folder in ('frames', 'targets', 'maps', 'diagnostics', 'controls'): (output/folder).mkdir()
    write_json(output/'frozen_plan.json', plan)
    canonical = hashlib.sha256(json.dumps(plan, sort_keys=True, allow_nan=False).encode()).hexdigest()
    frames, targets, control_stats, gaps = [], [], [], []
    started = time.perf_counter(); completed = 0
    write_json(output/'status.json', dict(phase='running', completed=0, total=len(rows)))
    try:
        for row in rows:
            prefix = row['prefix']; verify_source_files(row)
            paths = _frame_paths(args, prefix)
            for k, path in paths.items():
                if sha256_file(path) != identity['files'][prefix][k]: raise ValueError('Frozen pool changed after plan')
            data = load_input(row); gray, E, _, metadata, _, _, _, _, ridges = data
            fragments = read(paths['fragment']); weak = read(paths['weak']); baseline = read(paths['baseline'])
            maps, records, base, audit = apply_stages(data, fragments, weak, cached_baseline=baseline)
            for name, key, path in (('fragment', 'fragment', paths['fragment_maps']), ('weak', 'recall', paths['weak_maps'])):
                with np.load(path, allow_pickle=False) as z:
                    for k, value in (('TIR_uint8', gray), ('observed', E), ('trace_ridges', ridges), ('baseline', maps['baseline'])):
                        np.testing.assert_array_equal(z[k], value)
                    maps[name] = z[key].copy()
            frame = dict(prefix=prefix, filename=row['filename'], dataset=row['dataset'], video_id=row.get('video_id', 'external_sequence_unknown'))
            n = len(metadata['candidates'])
            frame_candidates = []
            for arm in ARMS:
                details = records[arm]['region_completion']
                frame[arm+'_accepted'] = details['new_accepted']
                frame[arm+'_candidates'] = details['new_candidates']
                frame[arm+'_duplicates'] = details['exact_duplicate_records']
                frame[arm+'_region_routes'] = details['accepted_routes'].get('regional_gap', 0)
                write_json(output/'diagnostics'/f'{prefix}_{arm}.json', records[arm])
                for c in records[arm]['candidates'][n:]:
                    frame_candidates.append(dict(prefix=prefix, arm=arm, id=c['id'], selected=c['geometry_selected'], accepted=c['reliability_accepted'],
                        route=c.get('evidence_route', 'legacy_C'), C=c['reliability_score'], R=c['add_importance'],
                        strength=c['strength'], region_support=c['region_evidence']['support'], region_available=c['region_evidence']['available'],
                        unique_region_pixels=c['region_evidence']['unique_pixels'], member_count=len(c['source_members']),
                        rejection='|'.join(c['reliability_rejections']), geometry_rejection=c['geometry_rejection']))
            append_candidates(output/'candidate_statistics.csv', frame_candidates)
            for arm in ('fragment', 'weak', *ARMS):
                delta = maps[arm]-maps['baseline']
                frame[arm+'_strength_change'] = float(delta.sum(dtype=np.float64))
                frame[arm+'_visible_pixels'] = int((delta >= 1/255).sum())
            frames.append(frame)
            write_json(output/'diagnostics'/f'{prefix}_groups.json', audit)
            Image.fromarray(gray).save(output/'frames'/f'{prefix}_TIR.png')
            for arm in ('baseline', 'fragment', 'weak', *ARMS): uint8_image(maps[arm]).save(output/'frames'/f'{prefix}_{arm}_condition.png')
            save_sheets(output, prefix, gray, maps)
            inc = dict(maps, **{k:maps[k]-E for k in ('fragment', 'region', 'joint')})
            make_sheet(gray, inc, modes=GROUPS['comparison'], labels=LABELS, importance_key='joint_mask', importance_caption='Joint new centers ONLY').save(output/'frames'/f'{prefix}_increments.png')
            local = np.repeat(gray[..., None], 3, axis=2).astype(np.float32)/255
            colors = dict(observed=np.array([0., .8, 1.]), regional_gap=np.array([1., .6, 0.]))
            for c in records['joint']['candidates'][n:]:
                if c['reliability_accepted']:
                    points = np.asarray(c['path_yx'], int)[1:-1]; yy, xx = points.T
                    local[yy, xx] = .2*local[yy, xx]+.8*colors[c['evidence_route']]
            uint8_image(local).save(output/'frames'/f'{prefix}_locator.png')
            np.savez_compressed(output/'maps'/f'{prefix}_regions.npz', TIR_uint8=gray, trace_ridges=ridges, **maps)
            yy, xx = np.indices(E.shape)
            for t in source['targets']:
                if row['dataset'] != 'FLIR-development' or t['filename'] != row['filename']: continue
                save_sheets(output, t['id'], gray, maps, folder='targets', crop=t['crop'], box=t['box'], zoom=args.zoom)
                x0, y0, x1, y1 = t['box']; mask = (xx+.5 >= x0)&(xx+.5 < x1)&(yy+.5 >= y0)&(yy+.5 < y1)
                item = dict(id=t['id'], prefix=prefix, category=t['category'])
                for arm in ('fragment', 'weak', *ARMS):
                    delta = maps[arm]-maps['baseline']
                    item[arm+'_visible_pixels'] = int(((delta >= 1/255)&mask).sum())
                    item[arm+'_strength_change'] = float(delta[mask].sum(dtype=np.float64))
                    if arm in ARMS: item[arm+'_centers'] = int((maps[arm+'_mask']&mask).sum())
                targets.append(item)
            for gap in annotations:
                if gap['prefix'] == prefix:
                    for arm in ARMS:
                        gaps.append(dict(arm=arm, **annotation_statistics(gap, E.shape, base, records[arm], maps['baseline']-E, maps[arm]-E)))
            completed += 1
            write_json(output/'status.json', dict(phase='running', completed=completed, total=len(rows)))
            print(f'[REGION {completed:02d}/{len(rows)}] {prefix} | C accepted dedup={frame["dedup_accepted"]} region={frame["region_accepted"]} weak={frame["weak_region_accepted"]} joint={frame["joint_accepted"]}', flush=True)
        for i, control in enumerate(controls, 1):
            row = next(r for r in rows if r['prefix'] == control['prefix']); verify_source_files(row)
            gray, E, _, _, structure, clutter, p, q, ridges = load_input(row)
            erase = erasure_mask(E.shape, control['polyline_yx'], control['radius']); truth = ridges&erase
            masked = E.copy(); masked[erase] = 0
            old, metadata = completion_comparison(_image01(gray), masked, trace_ridges=ridges&~erase)
            fragments, _, _ = fragment_completion(_image01(gray), masked, old['trace_ridges'], metadata,
                settings=FragmentSettings(**prior['plan']['settings']))
            _, weak = recall_completion(_image01(gray), masked, old['trace_ridges'], metadata)
            data = (gray, masked, old['reliable'], metadata, structure, clutter, p, q, old['trace_ridges'])
            maps, records, base, audit = apply_stages(data, fragments, weak)
            for name, key, folder in (('fragment', 'fragment', args.fragment_dir), ('weak', 'recall', args.weak_dir)):
                path = folder/'controls'/f'{control["id"]}.npz'
                if sha256_file(path) != identity['controls'][control['id']][name]: raise ValueError('Control changed after plan')
                with np.load(path, allow_pickle=False) as z:
                    for k, value in (('TIR_uint8', gray), ('truth', truth), ('erased', erase), ('observed', masked), ('baseline', maps['baseline'])):
                        np.testing.assert_array_equal(z[k], value)
                    maps[name] = z[key].copy()
                # Verify fresh search/importance against the unchanged old control.
                previous = fragments if name == 'fragment' else weak
                previous_maps, _ = prioritize_completion(masked, previous, structure, clutter, p, q)
                np.testing.assert_array_equal(maps[name], previous_maps['priority'])
            for arm in ARMS:
                stats = control_statistics(truth, erase, ridges, base, records[arm], maps['baseline']-masked, maps[arm]-masked)
                control_stats.append(dict(id=control['id'], prefix=control['prefix'], arm=arm, truth_pixels=int(truth.sum()), **stats))
                write_json(output/'controls'/f'{control["id"]}_{arm}.json', records[arm])
            write_json(output/'controls'/f'{control["id"]}_groups.json', audit)
            cmaps = dict(maps, truth=truth.astype(np.float32))
            make_sheet(gray, cmaps, modes=GROUPS['comparison'], labels=LABELS, importance_key='truth', importance_caption='Erased ridge').save(output/'controls'/f'{control["id"]}.png')
            points = np.argwhere(erase); low = np.maximum(0, points.min(0)-8); high = np.minimum(E.shape, points.max(0)+9)
            crop = [int(low[1]), int(low[0]), int(high[1]), int(high[0])]
            make_sheet(gray, cmaps, crop=crop, zoom=args.zoom, modes=GROUPS['comparison'], labels=LABELS, importance_key='truth', importance_caption='Erased ridge').save(output/'controls'/f'{control["id"]}_crop.png')
            np.savez_compressed(output/'controls'/f'{control["id"]}.npz', TIR_uint8=gray, truth=truth, erased=erase, **maps)
            print(f'[CONTROL {i:02d}/{len(controls)}] {control["id"]}', flush=True)
        video = []
        for dataset, v in sorted({(f['dataset'], f['video_id']) for f in frames}):
            group = [f for f in frames if f['dataset'] == dataset and f['video_id'] == v]
            keys = [k for k in group[0] if k not in ('prefix', 'filename', 'dataset', 'video_id')]
            video.append(dict(dataset=dataset, video_id=v, frames=len(group), **{k:sum(f[k] for f in group) for k in keys}))
        for name, values, empty in (('frame_statistics', frames, ('prefix',)), ('target_statistics', targets, ('id',)),
            ('video_statistics', video, ('dataset', 'video_id')),
            ('erasure_controls', control_stats, ('id', 'arm')), ('gap_statistics', gaps, ('id', 'arm', 'label'))):
            write_table(output/f'{name}.csv', values, empty)
        if not (output/'candidate_statistics.csv').exists(): write_table(output/'candidate_statistics.csv', [], ('prefix', 'arm'))
        selected_targets = [t for t in source['targets'] if t['id'] in {r['id'] for r in targets}]
        summary = dict(frames=len(frames), development_frames=sum(f['dataset'] == 'FLIR-development' for f in frames),
            target_crops=len(targets), auxiliary_erasures=len(controls), real_gap_annotations=len(annotations), real_gap_quality_verified=False,
            external_semantic_generalization_verified=False, parameters_learned=0, D_training_changed=False,
            original_C_R_preserved=True, frozen_plan_canonical_sha256=canonical, seconds=time.perf_counter()-started,
            accepted={arm:sum(f[arm+'_accepted'] for f in frames) for arm in ARMS},
            erasure_visible_coverage_mean={arm:float(np.mean([r['recall_visible_coverage_1px'] for r in control_stats if r['arm'] == arm])) for arm in ARMS} if control_stats else None)
        write_json(output/'run_manifest.json', dict(plan=plan, summary=summary, frames=frames, targets=selected_targets, controls=control_stats))
        write_page(output, frames, selected_targets, controls, summary); write_annotation_page(output, frames)
        write_json(output/'status.json', dict(phase='completed', completed=completed, total=len(rows), **summary))
    except BaseException as error:
        write_json(output/'status.json', dict(phase='failed', completed=completed, total=len(rows), error=str(error)))
        raise
    print(output/'index.html', flush=True)


if __name__ == '__main__':
    run(parse_args())
