#!/usr/bin/env python3
"""Fragment-pair proposals, isolated fixed-C rematch and amplitude controls."""

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import html
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'pytorch-CycleGAN-and-pix2pix'))
from models.fragment_completion import FragmentSettings,fragment_completion,rematch_fixed_C
from models.completion_importance import ImportanceSettings,prioritize_completion
from models.contour_completion import completion_comparison
from preview_completion_importance import SOURCE,SEMANTICS,CLARITY,preflight,select_external_training
from preview_completion_recall import (IMPORTANCE,load_input,source_identity,verify_source_files,control_plan,write_annotation_page)
from preview_contour_completion import _image01,write_table
from preview_semantic_contours import make_sheet
from semantic_preview.review import uint8_image
from semantic_preview.checkpoint import sha256_file
from preview_semantics import write_json
from completion_recall_audit import erasure_mask,control_statistics,read_annotations,annotation_statistics

WEAK=ROOT/'runs/edge_audit/completion_recall_seed42_v2'
OUTPUT=ROOT/'runs/edge_audit/fragment_completion_seed42_v3'
LABELS=dict(observed='Observed E',baseline='Old C / R',weak='Weak tracing',fragment='Fragment E amp',
    rematch='Rematch ONLY',observation='Fragment obs amp')
GROUPS=dict(comparison=('baseline','weak','fragment'),rematch=('observed','baseline','rematch'),
    amplitude=('baseline','fragment','observation'))


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for key,default in (('completion-dir',SOURCE),('semantic-preview-dir',SEMANTICS),('clarity-dir',CLARITY),
                        ('importance-dir',IMPORTANCE),('weak-dir',WEAK),('output-dir',OUTPUT)):
        p.add_argument('--'+key,type=Path,default=default)
    p.add_argument('--external-samples',type=int,default=6)
    p.add_argument('--erasure-samples',type=int,default=12)
    p.add_argument('--annotations',type=Path)
    p.add_argument('--zoom',type=int,default=6)
    p.add_argument('--max-anchors',type=int,default=16384,help='Explicit computational limit, recorded in the frozen plan')
    p.add_argument('--max-searches',type=int,default=65536,help='Covers all four-neighbor pairs for at most 16384 anchors')
    a=p.parse_args(argv)
    if a.output_dir.exists():p.error('Preserve previous comparisons; choose a new output directory')
    if not 0<=a.external_samples<=24 or not 0<=a.erasure_samples<=40 or not 1<=a.zoom<=12:
        p.error('External samples 0..24, erasures 0..40 and zoom 1..12 required')
    if a.max_anchors<1 or a.max_searches<1:p.error('Positive anchor and search budgets required')
    return a


def weak_identity(args,rows):
    manifest=json.loads((args.weak_dir/'run_manifest.json').read_text())
    if json.loads((args.weak_dir/'status.json').read_text()).get('phase')!='completed':raise ValueError('Previous weak preview incomplete')
    for name,digest in manifest['plan']['code_sha256'].items():
        path=(ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or sha256_file(path)!=digest:raise ValueError(f'Weak source changed: {name}')
    by_prefix={r['prefix']:r for r in manifest['frames']}
    hashes={}
    for row in rows:
        old=by_prefix.get(row['prefix'])
        if old is None or old['filename']!=row['filename'] or old['dataset']!=row['dataset']:raise ValueError('Comparisons require identical frames')
        hashes[row['prefix']]=sha256_file(args.weak_dir/'maps'/f'{row["prefix"]}_recall.npz')
    return manifest,dict(manifest_sha256=sha256_file(args.weak_dir/'run_manifest.json'),map_sha256=hashes)


def apply_arms(data,fragment_settings):
    gray,E,_,metadata,structure,clutter,p,q,ridges=data
    original=json.dumps(metadata,sort_keys=True,allow_nan=False)
    baseline,base_meta=prioritize_completion(E,metadata,structure,clutter,p,q)
    _,rematched=rematch_fixed_C(_image01(gray),E,ridges,metadata)
    rematch,rematch_meta=prioritize_completion(E,rematched,structure,clutter,p,q)
    fragments,probe,aux=fragment_completion(_image01(gray),E,ridges,metadata,settings=fragment_settings)
    fragment,fragment_meta=prioritize_completion(E,fragments,structure,clutter,p,q)
    observation,observation_meta=prioritize_completion(E,probe,structure,clutter,p,q)
    assert original==json.dumps(metadata,sort_keys=True,allow_nan=False)
    n=len(metadata['candidates'])
    assert fragment_meta['candidates'][:n]==base_meta['candidates']
    assert observation_meta['candidates'][:n]==base_meta['candidates']
    for a,b in zip(fragment_meta['candidates'],observation_meta['candidates']):
        assert {k:v for k,v in a.items() if k!='strength'}=={k:v for k,v in b.items() if k!='strength'}
    for a,b in zip(base_meta['candidates'],rematch_meta['candidates']):
        for key in ('path_yx','strength','features','reliability_score','reliability_rejections','add_importance','importance_gain'):
            assert a[key]==b[key]
        if a['reliability_accepted']:assert a==b
    maps=dict(observed=E,baseline=baseline['priority'],rematch=rematch['priority'],fragment=fragment['priority'],
        observation=observation['priority'],weak_ridges=aux['weak_ridges'])
    locators={}
    for name,records in (('fragment',fragment_meta),('rematch',rematch_meta)):
        mask=np.zeros(E.shape,bool)
        for c in records['candidates']:
            is_new=c['id']>=n if name=='fragment' else c['id'] in records['rematch']['new_accepted_ids']
            if is_new and c['reliability_accepted']:
                points=np.asarray(c['path_yx'],dtype=int)[1:-1];mask[tuple(points.T)]=True
        locators[name]=mask
    maps.update(fragment_mask=locators['fragment'],rematch_mask=locators['rematch'])
    for name in ('baseline','rematch','fragment','observation'):
        assert np.all(maps[name]>=E)
        if name!='baseline':assert np.all(maps[name]>=maps['baseline'])
        if not np.isfinite(maps[name]).all() or (maps[name]>1).any():raise FloatingPointError('Invalid fragment condition')
    return maps,dict(baseline=base_meta,rematch=rematch_meta,fragment=fragment_meta,observation=observation_meta)


def save_sheets(output,prefix,gray,maps,*,crop=None,box=None,zoom=1,subfolder='frames'):
    for group,modes in GROUPS.items():
        mask='rematch_mask' if group=='rematch' else 'fragment_mask'
        suffix='' if group=='comparison' and subfolder=='targets' else '_'+group
        make_sheet(gray,maps,crop=crop,box=box,zoom=zoom,modes=modes,labels=LABELS,
            importance_key=mask,importance_caption='New centers ONLY').save(output/subfolder/f'{prefix}{suffix}.png')


def write_page(output,frames,targets,controls,summary):
    cards=[]
    for f in frames:
        prefix=f['prefix']
        crops=''.join(f'<h4>{html.escape(t["id"])} / {html.escape(t["category"])}</h4><img loading="lazy" src="targets/{t["id"]}.png">'
            f'<details><summary>相同路径的幅值对照</summary><img loading="lazy" src="targets/{t["id"]}_amplitude.png"></details>'
            for t in targets if t['filename']==f['filename'] and f['dataset']=='FLIR-development')
        cards.append(f'<article id="f{prefix}"><h2>{prefix} / {html.escape(f["dataset"])} / {html.escape(f["filename"])}</h2>'
            f'<p>片段候选 {f["fragment_candidates"]} / 几何选用 {f["fragment_selected"]} / C通过 {f["fragment_accepted"]}；只重匹配额外接纳 {f["rematch_added"]}。数量不是正确率。</p>'
            f'<img loading="lazy" src="frames/{prefix}_comparison.png">'
            f'<details><summary>原E、旧补边、仅重匹配；旧C和R不变</summary><img loading="lazy" src="frames/{prefix}_rematch.png"></details>'
            f'<details><summary>仅改变新片段路径幅值：旧E继承 vs 原图片段支持</summary><img loading="lazy" src="frames/{prefix}_amplitude.png"></details>'
            f'<details><summary>实际新增强度，固定[0,1]</summary><img loading="lazy" src="frames/{prefix}_increments.png"></details>'
            f'<details><summary>新候选：橙色几何选用，青色C通过；等亮定位</summary><img loading="lazy" src="frames/{prefix}_locator.png"></details>'
            f'<details><summary>原固定目标裁剪</summary>{crops}</details>'
            f'<p><a href="diagnostics/{prefix}_fragment.json">片段候选与C/R</a> · <a href="diagnostics/{prefix}_rematch.json">固定C重匹配</a></p></article>')
    erased=''.join(f'<details><summary>{html.escape(c["id"])} / 擦除观察边，仅辅助检查</summary><img loading="lazy" src="controls/{c["id"]}.png"><img loading="lazy" src="controls/{c["id"]}_crop.png"></details>' for c in controls)
    intro='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>片段补全与独立重匹配对照</title>
<style>body{font:16px/1.65 system-ui;background:#f3f5f7;color:#182536;margin:18px}main{max-width:1800px;margin:auto}article,.notice{padding:18px;background:white;border:1px solid #cdd4df;border-radius:8px;margin:18px 0}.notice{background:#fff2d5}img{max-width:100%}h2{overflow-wrap:anywhere}summary{cursor:pointer;color:#155e91}nav a{display:inline-block;padding:3px 6px}</style><main>
<h1>片段之间提出短缺口，C/R与幅值分别检查</h1><div class="notice">
<p>主图依次：TIR、旧补边C/R、上一轮连续弱边追踪、新片段候选（继承E幅值）、新接纳中心线定位。两个测量片段给出位置和方向，可在断口内部缺少连续NMS时提出假设；原C仍可拒绝它。侧面灰度区域相容性只作软代价，不证明属于同一物体。</p>
<p>展开重匹配对照，仅释放旧C拒绝的几何连接，不重算原C。展开幅值对照，在完全相同的新片段路径、C和R上，比较继承已加权E与原始片段支持；更亮不能当作更正确。这两项不混入主候选机制。</p>
<p>原E和旧接纳路径均保留；片段组仍保留旧几何锁定。没有用图号、类别、框或图像上下位置搜索，不强制闭合目标，不恢复不可辨认的隐藏表面。固定局部搜索和方向规则仍有设计偏差。</p>
<p>新路径不得重复经过同一网格点，也不得在内部再次踩回原观察脊线。冻结搜索缓存的安全修复会重新计算C及配对，并用固定01号完整重算核对；不会把缓存旧接纳结果直接当最终结果。</p>
<p>40张FLIR是开发样本；12张MSRS/LLVIP为无语义回退检查。辅助擦边保留原TIR信息，不能当自然缺口真值或完整泛化。独立人工标注为空时不报告补线正确率；可使用<a href="annotate_gaps.html">原图缺口标记工具</a>。</p>
<p><a href="frame_statistics.csv">逐图</a> · <a href="video_statistics.csv">逐视频</a> · <a href="target_statistics.csv">目标定位</a> · <a href="candidate_statistics.csv">候选评分</a> · <a href="erasure_controls.csv">同位置擦边</a> · <a href="gap_statistics.csv">真实缺口标记</a> · <a href="run_manifest.json">完整冻结计划</a></p></div>'''
    nav=' '.join(f'<a href="#f{f["prefix"]}">{f["prefix"]}</a>' for f in frames)
    (output/'index.html').write_text(intro+f'<nav>{nav}</nav>'+''.join(cards)+'<article><h2>相同擦边位置控制</h2>'+erased+'</article><p>'+html.escape(json.dumps(summary,ensure_ascii=False))+'</p></main></html>',encoding='utf-8')


def run(args,*,cached_arm_builder=None,replay_provenance=None):
    if args.output_dir.exists():raise FileExistsError('Preserve existing outputs')
    torch.set_num_threads(2)
    source,development=preflight(args);external,availability=select_external_training(ROOT,args.external_samples)
    rows=development+external;identity=source_identity(args,rows)
    weak_manifest,weak_source=weak_identity(args,rows)
    controls,cache,control_availability=control_plan(rows,args.erasure_samples)
    if args.erasure_samples==12 and args.external_samples==6:
        assert controls==weak_manifest['plan']['controls'],'Keep auxiliary deletion sites identical'
    annotations=read_annotations(args.annotations,{r['prefix'] for r in rows})
    settings=FragmentSettings(max_anchors=args.max_anchors,max_searches=args.max_searches)
    code=dict(weak_manifest['plan']['code_sha256'])
    for path in (Path(__file__),ROOT/'pytorch-CycleGAN-and-pix2pix/models/fragment_completion.py'):
        code[str(path.relative_to(ROOT))]=sha256_file(path)
    if cached_arm_builder is not None:
        if not isinstance(replay_provenance,dict):raise ValueError('Cached replay requires explicit provenance')
        entry=(ROOT/replay_provenance['entry_source']).resolve()
        if not entry.is_relative_to(ROOT) or sha256_file(entry)!=replay_provenance['entry_sha256']:
            raise ValueError('Replay entry source changed')
        code[str(entry.relative_to(ROOT))]=replay_provenance['entry_sha256']
    plan=dict(format='fragment_completion_review_v1',settings=asdict(settings),importance_settings=asdict(ImportanceSettings()),seed=42,
        code_sha256=code,sources=rows,previous_importance=identity,previous_weak=weak_source,
        controls=controls,control_availability=control_availability,external_availability=availability,
        annotations=annotations,annotation_sha256=sha256_file(args.annotations) if args.annotations else None,
        fixed_before_processing=True,test_split_access=False,
        experiment_arms=dict(fragment='New fragment hypotheses, original C/R, inherited E amplitude',
            rematch='Original pool and C/R, release rejected geometry only',
            observation='Identical fragment hypotheses and C/R, change new-path amplitude only'),
        limitations='Development candidates; no real-gap quality claim, no complete external semantic inference, no D training')
    if replay_provenance is not None:plan['cached_search_replay']=replay_provenance
    output=args.output_dir;output.mkdir(parents=True,exist_ok=False)
    for folder in ('frames','maps','targets','diagnostics','controls'):(output/folder).mkdir()
    write_json(output/'frozen_plan.json',plan)
    digest=hashlib.sha256(json.dumps(plan,sort_keys=True,allow_nan=False).encode()).hexdigest()
    frames,candidates,target_stats,control_stats,gaps=[],[],[],[],[];completed=0;started=time.perf_counter()
    write_json(output/'status.json',dict(phase='running',completed=0,total=len(rows)))
    try:
        for row in rows:
            verify_source_files(row)
            data=cache.get(row['prefix'])
            if data is None:data=load_input(row)
            gray,E,old_reliable,metadata,structure,clutter,p,q,ridges=data
            maps,records=apply_arms(data,settings) if cached_arm_builder is None else cached_arm_builder(data,settings,row['prefix'])
            prefix=row['prefix'];path=args.importance_dir/'maps'/f'{prefix}_importance.npz'
            if sha256_file(path)!=identity['maps_sha256'][prefix]:raise ValueError('Old baseline changed')
            with np.load(path,allow_pickle=False) as z:np.testing.assert_array_equal(z['priority'],maps['baseline'])
            path=args.weak_dir/'maps'/f'{prefix}_recall.npz'
            if sha256_file(path)!=weak_source['map_sha256'][prefix]:raise ValueError('Previous weak result changed')
            with np.load(path,allow_pickle=False) as z:
                np.testing.assert_array_equal(z['observed'],E);np.testing.assert_array_equal(z['baseline'],maps['baseline'])
                maps['weak']=z['recall'].copy()
            details=records['fragment']['fragment_completion'];new=records['fragment']['candidates'][len(metadata['candidates']):]
            frame=dict(prefix=prefix,filename=row['filename'],dataset=row['dataset'],video_id=row.get('video_id','external_sequence_unknown'),
                old_accepted=metadata['reliability_accepted'],fragment_candidates=details['new_candidates'],
                fragment_selected=details['new_geometry_selected'],fragment_accepted=details['new_reliability_accepted'],
                rematch_added=len(records['rematch']['rematch']['new_accepted_ids']),
                truncated_anchors=details['truncated_anchors'],truncated_pairs=details['truncated_pairs'],
                observed_ports=details['observed_ports'],weak_ports=details['weak_ports'],
                baseline_strength=float((maps['baseline']-E).sum(dtype=np.float64)),semantic_mode=row['semantic_mode'])
            for arm in ('weak','fragment','rematch','observation'):
                delta=maps[arm]-maps['baseline']
                frame[arm+'_strength_change']=float(delta.sum(dtype=np.float64))
                frame[arm+'_visible_pixels']=int((delta>=1/255).sum())
            frames.append(frame)
            Image.fromarray(gray).save(output/'frames'/f'{prefix}_TIR.png')
            for arm in ('baseline','weak','fragment','rematch','observation'):uint8_image(maps[arm]).save(output/'frames'/f'{prefix}_{arm}_condition.png')
            save_sheets(output,prefix,gray,maps)
            inc=dict(maps,**{arm:maps[arm]-E for arm in ('baseline','weak','fragment','rematch','observation')})
            make_sheet(gray,inc,modes=GROUPS['comparison'],labels=LABELS,importance_key='fragment_mask',importance_caption='New centers ONLY').save(output/'frames'/f'{prefix}_increments.png')
            geometry=np.zeros(E.shape,bool)
            for c in new:
                if c['geometry_selected']:
                    points=np.asarray(c['path_yx'],dtype=int)[1:-1];geometry[tuple(points.T)]=True
            local=np.repeat(gray[...,None],3,axis=2).astype(np.float32)/255
            locator=np.where(geometry[...,None],.2*local+.8*np.array([1.,.7,0.]),local)
            locator=np.where(maps['fragment_mask'][...,None],.2*local+.8*np.array([0.,1.,1.]),locator)
            uint8_image(locator).save(output/'frames'/f'{prefix}_locator.png')
            np.savez_compressed(output/'maps'/f'{prefix}_fragments.npz',TIR_uint8=gray,trace_ridges=ridges,**maps)
            for arm in ('fragment','rematch','observation'):write_json(output/'diagnostics'/f'{prefix}_{arm}.json',records[arm])
            for c in new:
                candidates.append(dict(prefix=prefix,id=c['id'],sources='|'.join(c['anchor_sources']),junction=c['junction_anchor'],
                    selected=c['geometry_selected'],accepted=c['reliability_accepted'],geometry_rejection=c['geometry_rejection'],
                    C_rejections='|'.join(c['reliability_rejections']),C=c['reliability_score'],R=c['add_importance'],gain=c['importance_gain'],
                    inherited_amplitude=c['strength'],observation_amplitude=c['observation_amplitude'],region_disagreement=c['region_disagreement'],
                    gap=c['gap'],length=c['length'],**c['features']))
            for target in source['targets']:
                if row['dataset']!='FLIR-development' or target['filename']!=row['filename']:continue
                save_sheets(output,target['id'],gray,maps,crop=target['crop'],box=target['box'],zoom=args.zoom,subfolder='targets')
                yy,xx=np.indices(E.shape);x0,y0,x1,y1=target['box'];box=(xx+.5>=x0)&(xx+.5<x1)&(yy+.5>=y0)&(yy+.5<y1)
                stats=dict(id=target['id'],prefix=prefix,category=target['category'],fragment_center_pixels=int((maps['fragment_mask']&box).sum()),
                    rematch_center_pixels=int((maps['rematch_mask']&box).sum()))
                for arm in ('weak','fragment','rematch','observation'):
                    delta=maps[arm]-maps['baseline'];stats[arm+'_strength_change']=float(delta[box].sum(dtype=np.float64));stats[arm+'_visible_pixels']=int(((delta>=1/255)&box).sum())
                target_stats.append(stats)
            for gap in annotations:
                if gap['prefix']!=prefix:continue
                for arm in ('fragment','rematch','observation'):
                    gaps.append(dict(arm=arm,**annotation_statistics(gap,E.shape,metadata,records[arm],maps['baseline']-E,maps[arm]-E)))
            completed+=1
            write_json(output/'status.json',dict(phase='running',completed=completed,total=len(rows)))
            print(f'[FRAGMENT {completed:02d}/{len(rows)}] {prefix} | proposals={frame["fragment_candidates"]} C={frame["fragment_accepted"]} rematch={frame["rematch_added"]} truncated={frame["truncated_pairs"]}',flush=True)
        for i,control in enumerate(controls,1):
            row=next(r for r in rows if r['prefix']==control['prefix']);verify_source_files(row)
            gray,E,_,_,structure,clutter,p,q,ridges=cache[control['prefix']]
            erase=erasure_mask(E.shape,control['polyline_yx'],control['radius']);truth=ridges&erase
            masked_E=E.copy();masked_E[erase]=0
            old,metadata=completion_comparison(_image01(gray),masked_E,trace_ridges=ridges&~erase)
            data=(gray,masked_E,old['reliable'],metadata,structure,clutter,p,q,old['trace_ridges'])
            maps,records=apply_arms(data,settings)
            for arm in ('fragment','rematch','observation'):
                stats=control_statistics(truth,erase,ridges,metadata,records[arm],maps['baseline']-masked_E,maps[arm]-masked_E)
                control_stats.append(dict(id=control['id'],prefix=control['prefix'],arm=arm,truth_pixels=int(truth.sum()),**stats))
            cmaps=dict(maps,truth=truth.astype(np.float32))
            modes=('baseline','fragment','observation')
            make_sheet(gray,cmaps,modes=modes,labels=LABELS,importance_key='truth',importance_caption='Erased ridge').save(output/'controls'/f'{control["id"]}.png')
            points=np.argwhere(erase);low=np.maximum(0,points.min(0)-8);high=np.minimum(E.shape,points.max(0)+9)
            crop=[int(low[1]),int(low[0]),int(high[1]),int(high[0])]
            make_sheet(gray,cmaps,crop=crop,zoom=6,modes=modes,labels=LABELS,importance_key='truth',importance_caption='Erased ridge').save(output/'controls'/f'{control["id"]}_crop.png')
            np.savez_compressed(output/'controls'/f'{control["id"]}.npz',TIR_uint8=gray,truth=truth,erased=erase,
                **{arm:maps[arm] for arm in ('observed','baseline','fragment','rematch','observation')})
            print(f'[CONTROL {i:02d}/{len(controls)}] {control["id"]}',flush=True)
        video=[]
        for dataset,v in sorted({(f['dataset'],f['video_id']) for f in frames}):
            group=[f for f in frames if f['dataset']==dataset and f['video_id']==v]
            keys=('fragment_candidates','fragment_selected','fragment_accepted','rematch_added','fragment_strength_change','observation_strength_change')
            video.append(dict(dataset=dataset,video_id=v,images=len(group),**{k:sum(f[k] for f in group) for k in keys}))
        for name,data,empty in [('frame_statistics',frames,('prefix',)),('candidate_statistics',candidates,('prefix','id','C','R')),
            ('target_statistics',target_stats,('id','category')),('video_statistics',video,('dataset','video_id')),
            ('erasure_controls',control_stats,('id','arm')),('gap_statistics',gaps,('id','arm','label'))]:write_table(output/f'{name}.csv',data,empty)
        summary=dict(development_frames=len(development),external_structure_fallback_frames=len(external),target_crops=len(source['targets']),
            auxiliary_erasures=len(controls),real_gap_annotations=len(annotations),real_gap_quality_verified=False,
            new_fragment_candidates=sum(f['fragment_candidates'] for f in frames),new_fragment_accepted=sum(f['fragment_accepted'] for f in frames),
            fixed_C_rematch_added=sum(f['rematch_added'] for f in frames),truncated_frames=sum(f['truncated_pairs']>0 for f in frames),
            original_C_R_and_accepted_paths_preserved=True,D_training_changed=False,external_semantic_generalization_verified=False,
            frozen_plan_canonical_sha256=digest,seconds=time.perf_counter()-started)
        write_json(output/'run_manifest.json',dict(plan=plan,summary=summary,frames=frames,targets=source['targets'],controls=control_stats))
        write_page(output,frames,source['targets'],controls,summary);write_annotation_page(output,frames)
        write_json(output/'status.json',dict(phase='completed',completed=completed,total=len(rows),**summary))
    except BaseException as error:
        try:write_json(output/'status.json',dict(phase='failed',completed=completed,total=len(rows),error=str(error)))
        except OSError:pass
        raise
    print(output/'index.html',flush=True)


if __name__=='__main__':run(parse_args())
