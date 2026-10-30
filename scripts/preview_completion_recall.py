#!/usr/bin/env python3
"""Weak-observation recall, immutable old C/R, stage audit and erased controls."""

import argparse
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
from models.completion_recall import RecallSettings, recall_completion, source_diagnostics
from models.completion_importance import ImportanceSettings, prioritize_completion
from models.contour_completion import CompletionSettings, completion_comparison
from models.clarity_semantic_edges import clarity_components
from models.saliency_edges import structure_texture_importance
from models.semantic_edges import FMB_CLASS_NAMES, SETTINGS
from preview_completion_importance import (SOURCE, SEMANTICS, CLARITY, preflight,
    load_development, select_external_training)
from preview_contour_completion import _image01, write_table
from preview_semantic_contours import make_sheet
from semantic_preview.review import uint8_image
from semantic_preview.checkpoint import sha256_file
from preview_semantics import write_json
from completion_recall_audit import (select_erasure, erasure_mask, control_statistics,
    read_annotations, annotation_statistics, aggregate_diagnostics)

OUTPUT=ROOT/'runs/edge_audit/completion_recall_seed42_v2'
IMPORTANCE=ROOT/'runs/edge_audit/completion_importance_seed42_v1'
MODES=('observed','baseline','recall')
LABELS=dict(observed='Observed E',baseline='Old C / R',recall='Weak + C / R')


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--completion-dir',type=Path,default=SOURCE)
    p.add_argument('--semantic-preview-dir',type=Path,default=SEMANTICS)
    p.add_argument('--clarity-dir',type=Path,default=CLARITY)
    p.add_argument('--importance-dir',type=Path,default=IMPORTANCE)
    p.add_argument('--output-dir',type=Path,default=OUTPUT)
    p.add_argument('--external-samples',type=int,default=6)
    p.add_argument('--erasure-samples',type=int,default=12)
    p.add_argument('--annotations',type=Path,help='Optional independently marked real gaps; never changes the algorithm')
    p.add_argument('--zoom',type=int,default=6)
    a=p.parse_args(argv)
    if a.output_dir.exists():p.error('Preserve existing outputs; choose a new output directory')
    if not 0<=a.external_samples<=24 or not 0<=a.erasure_samples<=40 or not 1<=a.zoom<=12:
        p.error('External samples 0..24, erasures 0..40 and zoom 1..12 required')
    return a


def load_external_with_trace(row):
    if sha256_file(row['path'])!=row['sha256']:raise ValueError('External source changed')
    with Image.open(row['path']) as im:
        if np.asarray(im).dtype!=np.uint8:raise ValueError('Explicit normalization needed for non-8-bit TIR')
        im=im.convert('L');factor=min(1.,360/max(im.size))
        if factor<1:im=im.resize(tuple(max(8,int(round(v*factor))) for v in im.size),Image.Resampling.BILINEAR)
        gray=np.asarray(im).copy()
    image=_image01(gray)
    p=np.full((len(FMB_CLASS_NAMES),*gray.shape),1/len(FMB_CLASS_NAMES),np.float32)
    clarity,_=clarity_components(image,p,include_probes=False)
    ridges=(clarity['clarity_v1_fine_ridges']>0)|(clarity['clarity_v1_coarse_ridges']>0)|(clarity['raw_fine_ridges']>0)
    old,metadata=completion_comparison(image,clarity['clarity_v1'],trace_ridges=ridges)
    structure,clutter=structure_texture_importance(image,clarity['raw_fine'],clarity['raw_coarse'])
    structure=np.clip((structure-SETTINGS['importance_min'])/(SETTINGS['importance_max']-SETTINGS['importance_min']),0,1).astype(np.float32)
    return gray,old['observed'],old['reliable'],metadata,structure,clutter,None,None,old['trace_ridges']


def load_input(row):
    if row['dataset']!='FLIR-development':return load_external_with_trace(row)
    values=load_development(row)
    with np.load(row['paths']['floating'],allow_pickle=False) as z:ridges=z['trace_ridges'].astype(bool)
    return (*values,ridges)


def source_identity(args,rows):
    manifest=json.loads((args.importance_dir/'run_manifest.json').read_text())
    status=json.loads((args.importance_dir/'status.json').read_text())
    if status.get('phase')!='completed':raise ValueError('Previous importance preview is incomplete')
    for relative,digest in manifest['plan']['code_sha256'].items():
        path=(ROOT/relative).resolve()
        if not path.is_relative_to(ROOT) or sha256_file(path)!=digest:
            raise ValueError(f'Original C/R recipe changed: {relative}')
    previous={f['prefix']:f for f in manifest['frames']}
    hashes={};diagnostic_hashes={}
    for row in rows:
        old=previous.get(row['prefix'])
        if old is None or old['filename']!=row['filename'] or old['dataset']!=row['dataset']:
            raise ValueError('Recall and importance comparison must share the same inputs')
        path=args.importance_dir/'maps'/f'{row["prefix"]}_importance.npz'
        hashes[row['prefix']]=sha256_file(path)
        diagnostic_hashes[row['prefix']]=sha256_file(args.importance_dir/'diagnostics'/f'{row["prefix"]}_paths.json')
    return dict(manifest_sha256=sha256_file(args.importance_dir/'run_manifest.json'),maps_sha256=hashes,
        diagnostics_sha256=diagnostic_hashes)


def verify_source_files(row):
    if row['dataset']=='FLIR-development':
        for key,path in row['paths'].items():
            if sha256_file(path)!=row['sha256'][key]:raise ValueError(f'Input changed after plan: {row["prefix"]}/{key}')
    elif sha256_file(row['path'])!=row['sha256']:
        raise ValueError('External image changed after plan')


def control_plan(rows,count):
    # At most one frame per existing FLIR video/external dataset. These are
    # development controls, not held-out-video or full semantic OOD tests.
    groups={}
    for row in rows:
        key=(row['dataset'],row.get('video_id','external_sequence_unknown'))
        groups.setdefault(key,[]).append(row)
    rng=np.random.default_rng(42)
    keys=sorted(groups)
    order=rng.permutation(len(keys))
    controls,cache,availability=[],{},[]
    for group_index in order[:count]:
        group=sorted(groups[keys[int(group_index)]],key=lambda r:r['prefix'])
        row=group[int(rng.integers(len(group)))]
        data=load_input(row);cache[row['prefix']]=data
        length=(6,10,14)[len(availability)%3]
        points=select_erasure(data[8],data[1],42+int(group_index),length)
        availability.append(dict(prefix=row['prefix'],requested_length=length,available=points is not None))
        if points is not None:
            controls.append(dict(id=f'{row["prefix"]}_erase{length}',prefix=row['prefix'],length=length,
                polyline_yx=points,radius=1,source='seeded observed degree-two ridge; NOT independent boundary truth'))
    return controls,cache,availability


def write_annotation_page(output,frames):
    # Blind source-first marking: method output is hidden until explicitly
    # requested. Coordinates are always native TIR pixels, independent of zoom.
    records=[dict(prefix=f['prefix'],filename=f['filename']) for f in frames]
    template='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>独立缺口标记</title>
<style>body{font:16px/1.6 system-ui;margin:24px;max-width:1100px}canvas{width:min(100%,720px);image-rendering:pixelated;border:1px solid #aaa;display:block}button,select{margin:5px;padding:8px}img{max-width:100%}#status{white-space:pre-wrap}</style>
<h1>真实缺口复核：先看原始红外</h1><p>先在原图上点击需要检查的缺口，按顺序标出折线，然后保存该条。也应标记不应补线的负例和无法辨认的位置。框和标记不进入算法。下载JSON后通过 --annotations 另存复核结果；当前预览不假造人工准确率。</p>
<select id="frame"></select><select id="label"><option value="visible_boundary">可见边界缺口</option><option value="no_boundary">此处不应补边</option><option value="uncertain">无法判断</option></select>
<button id="undo">撤销一点</button><button id="save">保存当前标记</button><button id="download">下载标注JSON</button><canvas id="canvas"></canvas><p id="status"></p>
<details><summary>完成原图标记后，查看算法对照</summary><img id="result"></details><script>
const frames=__FRAMES__,gaps=[];let points=[];const select=document.getElementById('frame'),canvas=document.getElementById('canvas'),ctx=canvas.getContext('2d'),image=new Image();
frames.forEach((f,i)=>{const o=new Option(f.prefix+' / '+f.filename,i);select.add(o)});
function draw(){ctx.drawImage(image,0,0);if(points.length){ctx.strokeStyle='#ff7700';ctx.lineWidth=1;ctx.beginPath();points.forEach((p,i)=>i?ctx.lineTo(p[1]+.5,p[0]+.5):ctx.moveTo(p[1]+.5,p[0]+.5));ctx.stroke();points.forEach(p=>{ctx.fillStyle='#00ffff';ctx.fillRect(p[1]-1,p[0]-1,3,3)})}document.getElementById('status').textContent='本条 '+points.length+' 点；已保存 '+gaps.length+' 条';}
function load(){points=[];const f=frames[Number(select.value)];image.onload=()=>{canvas.width=image.width;canvas.height=image.height;draw()};image.src='frames/'+f.prefix+'_TIR.png';document.getElementById('result').src='frames/'+f.prefix+'_comparison.png';}
canvas.onclick=e=>{const r=canvas.getBoundingClientRect();points.push([Math.min(canvas.height-1,Math.max(0,Math.floor((e.clientY-r.top)*canvas.height/r.height))),Math.min(canvas.width-1,Math.max(0,Math.floor((e.clientX-r.left)*canvas.width/r.width)))]);draw()};
document.getElementById('undo').onclick=()=>{points.pop();draw()};document.getElementById('save').onclick=()=>{if(points.length<2){alert('至少标两个点');return}gaps.push({id:'gap_'+String(gaps.length+1).padStart(3,'0'),prefix:frames[Number(select.value)].prefix,label:document.getElementById('label').value,polyline_yx:points.map(p=>p.slice()),notes:''});points=[];draw()};
document.getElementById('download').onclick=()=>{const b=new Blob([JSON.stringify({format:'completion_gap_annotations_v1',note:'Human source-first annotations; not used for algorithm fitting',gaps},null,2)],{type:'application/json'}),u=URL.createObjectURL(b),a=document.createElement('a');a.href=u;a.download='completion_gap_annotations.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000)};select.onchange=load;load();</script></html>'''
    (output/'annotate_gaps.html').write_text(template.replace('__FRAMES__',json.dumps(records,ensure_ascii=False).replace('<','\\u003c')),encoding='utf-8')


def write_page(output,frames,targets,controls,summary):
    cards=[]
    for f in frames:
        p=f['prefix']
        crops=''.join(f'<h4>{html.escape(t["id"])} / {html.escape(t["category"])}</h4><img loading="lazy" src="targets/{t["id"]}.png">'
            for t in targets if t['filename']==f['filename'] and f['dataset']=='FLIR-development')
        cards.append(f'<article id="f{p}"><h2>{p} / {html.escape(f["dataset"])} / {html.escape(f["filename"])}</h2>'
            f'<p>旧接纳路径 {f["old_accepted"]} 全保留；新提出 {f["new_candidates"]} / 几何选用 {f["new_selected"]} / C通过 {f["new_accepted"]}。这些数量不是正确率。</p>'
            f'<img loading="lazy" src="frames/{p}_comparison.png">'
            f'<details><summary>实际新增强度，固定[0,1]，不拉亮</summary><img loading="lazy" src="frames/{p}_increments.png"></details>'
            f'<details><summary>本轮新接纳中心线定位：黄色等亮，仅供找位置</summary><img loading="lazy" src="frames/{p}_locator.png"></details>'
            f'<details><summary>原固定裁剪，框不参与算法</summary>{crops}</details>'
            f'<p><a href="diagnostics/{p}_paths.json">原路径、新候选、C/R与停止原因</a></p></article>')
    erased=''.join(f'<details><summary>{html.escape(c["id"])} / 擦除已观测边，不是真实缺口真值</summary><img loading="lazy" src="controls/{c["id"]}.png"></details>' for c in controls)
    intro='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>弱边候选召回与漏补诊断</title><style>body{font:16px/1.65 system-ui;background:#f3f5f7;color:#182536;margin:18px}main{max-width:1800px;margin:auto}article,.notice{padding:18px;background:white;border:1px solid #cdd4df;border-radius:8px;margin:18px 0}.notice{background:#fff2d5}img{max-width:100%}h2{overflow-wrap:anywhere}summary{cursor:pointer;color:#155e91}nav a{display:inline-block;padding:3px 6px}</style>
<main><h1>弱边候选召回：原C、R和几何决策锁定</h1><div class="notice">
<p>左至右：TIR、原始观察轮廓、不加新候选的旧C/R补边、加入原图弱边候选后使用相同C/R、本轮新通过中心线定位。只从未占用的已有端点沿短距离、连续的红外脊线延伸。没有原图证据就停止，不按人车形状、图号、位置或框补线。</p>
<p>旧路径、旧C、旧R保持。可靠性拒绝后尝试其他配对不在这轮实施，因此旧几何锁定仍可能限制召回。弱边延伸不等于恢复遮挡后的隐藏结构，也不保证边界归属正确。</p>
<p>40张FLIR是既有开发样本；MSRS／LLVIP只检查无语义回退，未验证完整外部语义链。擦除已观测轮廓的实验只验证可恢复性，不能当作真实脚部／轮胎准确率、独立泛化或RGB收益。</p>
<p>真实缺口的独立标注为空时不输出准确率。可使用<a href="annotate_gaps.html">原图缺口标记工具</a>标记可见缺口、负例和不确定位置。算法对照默认折叠，减少按结果标注的倾向。</p>
<p><a href="frame_statistics.csv">逐图</a> · <a href="video_statistics.csv">逐视频</a> · <a href="endpoint_diagnostics.csv">端点漏补环节</a> · <a href="candidate_statistics.csv">新路径C/R</a> · <a href="erasure_controls.csv">擦边控制</a> · <a href="gap_statistics.csv">独立缺口复核</a> · <a href="run_manifest.json">冻结计划</a></p></div>'''
    nav=' '.join(f'<a href="#f{f["prefix"]}">{f["prefix"]}</a>' for f in frames)
    (output/'index.html').write_text(intro+f'<nav>{nav}</nav>'+''.join(cards)+'<article><h2>辅助擦边控制</h2>'+erased+'</article><p>'+html.escape(json.dumps(summary,ensure_ascii=False))+'</p></main></html>',encoding='utf-8')


def run(args):
    if args.output_dir.exists():raise FileExistsError('Preserve previous output')
    torch.set_num_threads(2)
    source,development=preflight(args)
    external,external_availability=select_external_training(ROOT,args.external_samples)
    rows=development+external
    identity=source_identity(args,rows)
    annotations=read_annotations(args.annotations,{r['prefix'] for r in rows})
    controls,cache,control_availability=control_plan(rows,args.erasure_samples)
    recall_settings=RecallSettings();importance_settings=ImportanceSettings()
    files=[Path(__file__),ROOT/'scripts/completion_recall_audit.py',ROOT/'pytorch-CycleGAN-and-pix2pix/models/completion_recall.py']
    old_manifest=json.loads((args.importance_dir/'run_manifest.json').read_text())
    code=dict(old_manifest['plan']['code_sha256'])
    code.update({str(p.relative_to(ROOT)):sha256_file(p) for p in files})
    plan=dict(format='completion_recall_review_v1',seed=42,recall_settings=asdict(recall_settings),
        importance_settings=asdict(importance_settings),C_settings=asdict(CompletionSettings()),
        code_sha256=code,sources=rows,previous_importance=identity,controls=controls,
        control_selection_availability=control_availability,external_availability=external_availability,
        annotations=annotations,annotation_sha256=sha256_file(args.annotations) if args.annotations else None,
        parameters_fixed_before_processing=True,test_split_access=False,
        limits='Development preview, same C/R and old geometry; erased observed ridges are not real-gap annotations; external structural fallback only')
    output=args.output_dir;output.mkdir(parents=True,exist_ok=False)
    for folder in ('frames','targets','maps','diagnostics','controls'):(output/folder).mkdir()
    write_json(output/'frozen_plan.json',plan)
    plan_sha=hashlib.sha256(json.dumps(plan,sort_keys=True,allow_nan=False).encode()).hexdigest()
    frames,candidates,endpoints,targets,gaps,control_stats,baseline_diagnostics=[],[],[],[],[],[],[]
    started=time.perf_counter();completed=0
    write_json(output/'status.json',dict(phase='running',completed=0,total=len(rows)))
    try:
        for row in rows:
            verify_source_files(row)
            data=cache.get(row['prefix'])
            if data is None:data=load_input(row)
            gray,observed,old_reliable,metadata,structure,clutter,p,q,ridges=data
            original=json.dumps(metadata,sort_keys=True,allow_nan=False)
            old_R,old_updated=prioritize_completion(observed,metadata,structure,clutter,p,q,settings=importance_settings)
            np.testing.assert_array_equal(old_R['baseline'],old_reliable)
            path=args.importance_dir/'maps'/f'{row["prefix"]}_importance.npz'
            if sha256_file(path)!=identity['maps_sha256'][row['prefix']]:raise ValueError('Prior comparison changed')
            with np.load(path,allow_pickle=False) as z:
                np.testing.assert_array_equal(z['observed'],observed)
                np.testing.assert_array_equal(z['priority'],old_R['priority'])
            path=args.importance_dir/'diagnostics'/f'{row["prefix"]}_paths.json'
            if sha256_file(path)!=identity['diagnostics_sha256'][row['prefix']]:raise ValueError('Original candidate record changed')
            previous_metadata=json.loads(path.read_text())
            assert previous_metadata['candidates']==old_updated['candidates']
            assert json.dumps(previous_metadata['endpoints'],sort_keys=True)==json.dumps(old_updated['endpoints'],sort_keys=True)
            recalled,updated=recall_completion(_image01(gray),observed,ridges,metadata,settings=recall_settings)
            new_R,new_updated=prioritize_completion(observed,updated,structure,clutter,p,q,settings=importance_settings)
            assert original==json.dumps(metadata,sort_keys=True,allow_nan=False)
            assert updated['candidates'][:len(metadata['candidates'])]==metadata['candidates']
            assert new_updated['candidates'][:len(metadata['candidates'])]==old_updated['candidates']
            maps=dict(observed=observed,baseline=old_R['priority'],recall=new_R['priority'],
                baseline_increment=old_R['priority']-observed,recall_increment=new_R['priority']-observed,
                recall_only_increment=new_R['priority']-old_R['priority'],recall_mask=recalled['recall_mask'],
                weak_ridges=recalled['weak_ridges'])
            assert np.all(maps['recall']>=maps['baseline']) and np.all(maps['baseline']>=observed)
            prefix=row['prefix'];details=updated['weak_recall']
            frame=dict(prefix=prefix,filename=row['filename'],dataset=row['dataset'],video_id=row.get('video_id','external_sequence_unknown'),
                old_accepted=metadata['reliability_accepted'],new_candidates=details['new_candidates'],
                new_selected=details['new_geometry_selected'],new_accepted=details['new_reliability_accepted'],
                new_center_pixels=int(maps['recall_mask'].sum()),new_strength=float(maps['recall_only_increment'].sum(dtype=np.float64)),
                old_strength=float(maps['baseline_increment'].sum(dtype=np.float64)),
                new_visible_pixels=int((maps['recall_only_increment']>=1/255).sum()),semantic_mode=row['semantic_mode'])
            frames.append(frame);baseline_diagnostics.append(source_diagnostics(metadata))
            Image.fromarray(gray).save(output/'frames'/f'{prefix}_TIR.png')
            for mode in MODES:uint8_image(maps[mode]).save(output/'frames'/f'{prefix}_{mode}.png')
            make_sheet(gray,maps,modes=MODES,labels=LABELS,importance_key='recall_mask',importance_caption='New centers ONLY').save(output/'frames'/f'{prefix}_comparison.png')
            inc=dict(maps,observed=np.zeros_like(observed),baseline=maps['baseline_increment'],recall=maps['recall_increment'])
            make_sheet(gray,inc,modes=MODES,labels=LABELS,importance_key='recall_mask',importance_caption='New centers ONLY').save(output/'frames'/f'{prefix}_increments.png')
            locator=np.repeat(gray[...,None],3,axis=2).astype(np.float32)/255
            mask=maps['recall_mask'][...,None]
            uint8_image(np.where(mask,.2*locator+.8*np.array([1.,.7,0.]),locator)).save(output/'frames'/f'{prefix}_locator.png')
            np.savez_compressed(output/'maps'/f'{prefix}_recall.npz',TIR_uint8=gray,trace_ridges=ridges,**maps)
            write_json(output/'diagnostics'/f'{prefix}_paths.json',new_updated)
            for e in details['endpoint_ledger']:
                e=dict(e);e['stops']=json.dumps(e.get('stops',{}),sort_keys=True)
                endpoints.append(dict(prefix=prefix,**e))
            for c in new_updated['candidates'][len(metadata['candidates']):]:
                candidates.append(dict(prefix=prefix,id=c['id'],selected=c['geometry_selected'],accepted=c['reliability_accepted'],
                    geometry_rejection=c['geometry_rejection'],C_rejections='|'.join(c['reliability_rejections']),C=c['reliability_score'],
                    R=c['add_importance'],gain=c['importance_gain'],strength=c['strength'],length=c['length'],gap=c['gap'],**c['features']))
            for t in source['targets']:
                if row['dataset']!='FLIR-development' or t['filename']!=row['filename']:continue
                make_sheet(gray,maps,crop=t['crop'],box=t['box'],zoom=args.zoom,modes=MODES,labels=LABELS,
                    importance_key='recall_mask',importance_caption='New centers ONLY').save(output/'targets'/f'{t["id"]}.png')
                yy,xx=np.indices(gray.shape);x0,y0,x1,y1=t['box']
                box=(xx+.5>=x0)&(xx+.5<x1)&(yy+.5>=y0)&(yy+.5<y1)
                targets.append(dict(id=t['id'],prefix=prefix,category=t['category'],new_center_pixels=int((maps['recall_mask']&box).sum()),
                    new_strength=float(maps['recall_only_increment'][box].sum(dtype=np.float64))))
            for gap in annotations:
                if gap['prefix']==prefix:gaps.append(annotation_statistics(gap,gray.shape,metadata,updated,maps['baseline_increment'],maps['recall_increment']))
            completed+=1
            write_json(output/'status.json',dict(phase='running',completed=completed,total=len(rows)))
            print(f'[RECALL {completed:02d}/{len(rows)}] {prefix} | new proposals={frame["new_candidates"]} selected={frame["new_selected"]} C accepted={frame["new_accepted"]}',flush=True)
        for index,control in enumerate(controls,1):
            row=next(r for r in rows if r['prefix']==control['prefix'])
            verify_source_files(row)
            gray,observed,_,_,structure,clutter,p,q,ridges=cache[control['prefix']]
            erased=erasure_mask(gray.shape,control['polyline_yx'],control['radius'])
            truth=ridges&erased;masked_E=observed.copy();masked_E[erased]=0
            masked_ridges=ridges&~erased
            old,metadata=completion_comparison(_image01(gray),masked_E,trace_ridges=masked_ridges)
            _,recalled_metadata=recall_completion(_image01(gray),masked_E,old['trace_ridges'],metadata,settings=recall_settings)
            old_R,_=prioritize_completion(masked_E,metadata,structure,clutter,p,q,settings=importance_settings)
            new_R,_=prioritize_completion(masked_E,recalled_metadata,structure,clutter,p,q,settings=importance_settings)
            stats=control_statistics(truth,erased,ridges,metadata,recalled_metadata,old_R['priority']-masked_E,new_R['priority']-masked_E)
            control_stats.append(dict(id=control['id'],prefix=control['prefix'],dataset=row['dataset'],length=control['length'],truth_pixels=int(truth.sum()),**stats))
            cmaps=dict(observed=masked_E,baseline=old_R['priority'],recall=new_R['priority'],truth=truth.astype(np.float32))
            make_sheet(gray,cmaps,modes=MODES,labels=LABELS,importance_key='truth',importance_caption='Erased ridge').save(output/'controls'/f'{control["id"]}.png')
            np.savez_compressed(output/'controls'/f'{control["id"]}.npz',TIR_uint8=gray,erased=erased,truth=truth,**{k:cmaps[k] for k in MODES})
            print(f'[ERASURE {index:02d}/{len(controls)}] {control["id"]} | visible coverage {stats["baseline_visible_coverage_1px"]:.3f} -> {stats["recall_visible_coverage_1px"]:.3f}',flush=True)
        video=[]
        for dataset,v in sorted({(f['dataset'],f['video_id']) for f in frames}):
            group=[f for f in frames if f['dataset']==dataset and f['video_id']==v]
            video.append(dict(dataset=dataset,video_id=v,images=len(group),**{key:sum(f[key] for f in group) for key in ('new_candidates','new_selected','new_accepted','new_center_pixels','new_strength')}))
        for name,records,empty in [('frame_statistics',frames,('prefix',)),('video_statistics',video,('dataset','video_id')),
            ('candidate_statistics',candidates,('prefix','id','C','R')),('endpoint_diagnostics',endpoints,('prefix','endpoint_id','status','stops')),
            ('target_statistics',targets,('id','category','new_center_pixels')),('erasure_controls',control_stats,('id','prefix','truth_pixels')),
            ('gap_statistics',gaps,('id','prefix','label','notes'))]:
            write_table(output/f'{name}.csv',records,empty)
        diagnostic=aggregate_diagnostics(baseline_diagnostics)
        write_json(output/'baseline_stage_diagnostics.json',diagnostic)
        write_json(output/'gap_annotations_template.json',dict(format='completion_gap_annotations_v1',gaps=[]))
        summary=dict(development_frames=len(development),external_structure_fallback_frames=len(external),
            old_accepted_paths_preserved=sum(f['old_accepted'] for f in frames),
            new_candidates=sum(f['new_candidates'] for f in frames),new_accepted=sum(f['new_accepted'] for f in frames),
            real_gap_annotations=len(annotations),real_gap_quality_verified=False,auxiliary_erasure_cases=len(control_stats),
            erasure_visible_coverage_mean={name:float(np.mean([c[f'{name}_visible_coverage_1px'] for c in control_stats])) for name in ('baseline','recall')} if control_stats else None,
            C_R_and_old_paths_preserved=True,D_training_changed=False,full_external_semantic_generalization_verified=False,
            seconds=time.perf_counter()-started,frozen_plan_canonical_sha256=plan_sha)
        write_json(output/'run_manifest.json',dict(plan=plan,summary=summary,frames=frames,targets=source['targets'],controls=control_stats))
        write_page(output,frames,source['targets'],controls,summary)
        write_annotation_page(output,frames)
        write_json(output/'status.json',dict(phase='completed',completed=completed,total=len(rows),**summary))
    except BaseException as error:
        try:write_json(output/'status.json',dict(phase='failed',completed=completed,total=len(rows),error=str(error)))
        except OSError:pass
        raise
    print(output/'index.html',flush=True)


if __name__=='__main__':run(parse_args())
