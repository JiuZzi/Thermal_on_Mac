#!/usr/bin/env python3
"""Fixed-path relevance comparison, matched-mean control and fallback stress."""

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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'pytorch-CycleGAN-and-pix2pix'))
from models.completion_importance import ImportanceSettings, prioritize_completion
from models.contour_completion import completion_comparison, CompletionSettings
from models.clarity_semantic_edges import clarity_components
from models.saliency_edges import structure_texture_importance
from models.semantic_edges import FMB_CLASS_NAMES, SETTINGS
from models.learned_color_prior import semantic_reliability
from preview_contour_completion import _image01, write_table
from preview_semantic_contours import read_cache, make_sheet
from semantic_preview.checkpoint import sha256_file
from semantic_preview.review import uint8_image
from preview_semantics import write_json

SOURCE = ROOT/'runs/edge_audit/contour_completion_seed42_v2'
SEMANTICS = ROOT/'runs/edge_audit/segformer_fmb_b3_semantic_preview_seed42'
CLARITY = ROOT/'runs/edge_audit/contour_clarity_seed42_v1'
OUTPUT = ROOT/'runs/edge_audit/completion_importance_seed42_v1'
LABELS = dict(baseline='Old reliability completion', uniform='Same-mean dimming control', priority='Fixed paths + importance')
MODES = tuple(LABELS)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--completion-dir', type=Path, default=SOURCE)
    p.add_argument('--semantic-preview-dir', type=Path, default=SEMANTICS)
    p.add_argument('--clarity-dir', type=Path, default=CLARITY)
    p.add_argument('--output-dir', type=Path, default=OUTPUT)
    p.add_argument('--external-samples', type=int, default=6, help='Fixed train-side samples per available external dataset; 0 disables')
    p.add_argument('--zoom', type=int, default=6)
    args = p.parse_args(argv)
    if args.output_dir.exists(): p.error('Preserve existing previews; choose a new output directory')
    if not 0 <= args.external_samples <= 24 or not 1 <= args.zoom <= 12:
        p.error('External samples 0..24, zoom 1..12 required')
    return args


def select_external_training(root, count=6, seed=42):
    """Select by seed and filenames only; never read test split or annotations."""
    records, availability = [], {}
    for dataset, relative in (('MSRS', 'MSRS/train/ir'), ('LLVIP', 'LLVIP/infrared/train')):
        folder = root/relative
        files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in ('.png','.jpg','.jpeg','.bmp')) if folder.is_dir() else []
        availability[dataset] = dict(train_directory=relative, available_images=len(files), requested=count)
        if not files or count == 0: continue
        rng = np.random.default_rng(seed)
        chosen = sorted(rng.choice(len(files), size=min(count,len(files)), replace=False))
        for index, i in enumerate(chosen, 1):
            path = files[int(i)]
            records.append(dict(prefix=f'{dataset}_{index:02d}', dataset=dataset,
                path=str(path.resolve()), filename=path.name, sha256=sha256_file(path),
                semantic_mode='unavailable_structure_fallback'))
    return records, availability


def preflight(args):
    source = json.loads((args.completion_dir/'preview_manifest.json').read_text())
    status = json.loads((args.completion_dir/'status.json').read_text())
    if status.get('phase') != 'completed' or status.get('completed') != len(source['frames']):
        raise ValueError('Source completion is incomplete')
    if source['identity']['semantic_manifest_sha256'] != sha256_file(args.semantic_preview_dir/'preview_manifest.json'):
        raise ValueError('Semantic source changed')
    if source['identity']['clarity_manifest_sha256'] != sha256_file(args.clarity_dir/'preview_manifest.json'):
        raise ValueError('Clarity source changed')
    for name, digest in source['identity']['source_files'].items():
        path = (ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or sha256_file(path) != digest:
            raise ValueError(f'Source completion recipe changed: {name}')
    semantics = json.loads((args.semantic_preview_dir/'preview_manifest.json').read_text())
    by_name = {f['filename']: f for f in semantics['frames']}
    original = {r['filename']: r for r in source['identity']['selected']}
    if len(original) != len(source['identity']['selected']) or len(by_name) != len(semantics['frames']):
        raise ValueError('Duplicate source identities')
    records = []
    if len({f['filename'] for f in source['frames']}) != len(source['frames']):
        raise ValueError('Duplicate completion frames')
    for f in source['frames']:
        name, prefix = f['filename'], f['prefix']
        if Path(name).name != name or not prefix.isdigit() or name not in original or name not in by_name:
            raise ValueError('Unsafe/missing source identity')
        row = original[name]
        tir = ROOT/'FLIR_datasets/trainB'/name
        paths = dict(tir=tir, floating=args.completion_dir/'maps'/f'{prefix}_completion.npz',
            candidates=args.completion_dir/'diagnostics'/f'{prefix}_candidates.json',
            clarity=args.clarity_dir/'maps'/f'{row["cache_prefix"]}_contours.npz',
            semantic=args.semantic_preview_dir/'maps'/f'{by_name[name]["prefix"]}_responses.npz')
        hashes = {key:sha256_file(path) for key,path in paths.items()}
        for key, expected in (('tir',row['tir_sha256']),('clarity',row['cache_sha256']),('semantic',row['semantic_cache_sha256'])):
            if hashes[key] != expected: raise ValueError(f'Frozen input changed: {name}/{key}')
        records.append(dict(prefix=prefix, filename=name, dataset='FLIR-development', video_id=f['video_id'],
            paths={k:str(p.resolve()) for k,p in paths.items()}, sha256=hashes,
            semantic_mode='frozen_single_view_cache_Q_boundary_radius1'))
    return source, records


def load_development(row):
    for key,path in row['paths'].items():
        if sha256_file(path) != row['sha256'][key]: raise ValueError('Input changed after preflight')
    with np.load(row['paths']['floating'],allow_pickle=False) as z:
        gray, observed, baseline = z['TIR_uint8'].copy(), z['observed'].copy(), z['reliable'].copy()
    with Image.open(row['paths']['tir']) as im: np.testing.assert_array_equal(gray,np.asarray(im))
    _, p = read_cache(Path(row['paths']['semantic']),gray)
    with np.load(row['paths']['clarity'],allow_pickle=False) as z:
        np.testing.assert_array_equal(observed,z['clarity_v1'])
        structure, clutter = structure_texture_importance(_image01(gray),z['raw_fine'],z['raw_coarse'])
    structure = np.clip((structure-SETTINGS['importance_min'])/(SETTINGS['importance_max']-SETTINGS['importance_min']),0,1).astype(np.float32)
    q = semantic_reliability(torch.from_numpy(p)[None],boundary_radius=1)[0,0].numpy()
    metadata = json.loads(Path(row['paths']['candidates']).read_text())
    return gray, observed, baseline, metadata, structure, clutter, p, q


def load_external(row):
    if sha256_file(row['path']) != row['sha256']: raise ValueError('External input changed')
    with Image.open(row['path']) as im:
        if np.asarray(im).dtype != np.uint8: raise ValueError('Explicit normalization needed for non-8-bit thermal input')
        im = im.convert('L'); original_size = im.size
        factor = min(1.,360/max(im.size))
        if factor < 1: im = im.resize(tuple(max(8,int(round(v*factor))) for v in im.size),Image.Resampling.BILINEAR)
        gray = np.asarray(im).copy()
    image = _image01(gray)
    unknown = np.full((len(FMB_CLASS_NAMES),*gray.shape),1/len(FMB_CLASS_NAMES),np.float32)
    maps, _ = clarity_components(image,unknown,include_probes=False)
    ridges = (maps['clarity_v1_fine_ridges']>0)|(maps['clarity_v1_coarse_ridges']>0)|(maps['raw_fine_ridges']>0)
    old, metadata = completion_comparison(image,maps['clarity_v1'],trace_ridges=ridges)
    structure, clutter = structure_texture_importance(image,maps['raw_fine'],maps['raw_coarse'])
    structure = np.clip((structure-SETTINGS['importance_min'])/(SETTINGS['importance_max']-SETTINGS['importance_min']),0,1).astype(np.float32)
    metadata['external_preprocessing'] = dict(original_size=original_size,processed_size=tuple(gray.shape[::-1]),
        E_semantics='uniform unknown probabilities; not actual semantic inference',resize='aspect-preserving, long side at most 360; no crop')
    return gray,old['observed'],old['reliable'],metadata,structure,clutter,None,None


def write_page(output, frames, targets, summary):
    cards = []
    for f in frames:
        prefix=f['prefix']
        crops=''.join(f'<h4>{html.escape(t["id"])} / {html.escape(t["category"])}</h4><img loading="lazy" src="targets/{t["id"]}.png">'
            for t in targets if t['filename']==f['filename'] and f['dataset']=='FLIR-development')
        cards.append(f'<article id="f{prefix}"><h2>{prefix} / {html.escape(f["dataset"])} / {html.escape(f["filename"])}</h2>'
            f'<p>路径及C决策保持 {f["accepted_paths"]} 条；新增总强度保留 {f["retained_fraction"]:.1%}。这是强度分配，不是正确率。</p>'
            f'<img loading="lazy" src="frames/{prefix}_comparison.png">'
            f'<details><summary>新增部分：旧版／同均值变暗／重要性加权（固定[0,1]）</summary><img loading="lazy" src="frames/{prefix}_increments.png"></details>'
            f'<details><summary>目标裁剪，只用于定位</summary>{crops}</details>'
            f'<p><a href="diagnostics/{prefix}_paths.json">逐路径R、C与权重</a></p></article>')
    nav=' '.join(f'<a href="#f{f["prefix"]}">{f["prefix"]}</a>' for f in frames)
    intro='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>新增补边：任务重要性独立对照</title><style>body{font:16px/1.65 system-ui,sans-serif;background:#f3f5f7;color:#182536;margin:18px}main{max-width:1800px;margin:auto}article,.notice{padding:18px;background:white;border:1px solid #cdd4df;border-radius:8px;margin:18px 0}.notice{background:#fff2d5}img{max-width:100%}h2{overflow-wrap:anywhere}summary{cursor:pointer;color:#155e91}nav{background:white;padding:8px}nav a{display:inline-block;padding:3px 6px}</style>
<main><h1>新增补边的重要性：固定路径与C，单独改变R</h1><div class="notice">
<p>从左至右：TIR、旧可靠性补边、仅新增部分同均值变暗、同候选重要性加权、新增路径权重图。下排为真实强度叠加。原E不改变，不增加或移动路径，不把R叫作C。</p>
<p>类别无关的可信区域边界＋可配置任务类别＋结构回退共同确定R。道路、树冠和建筑的主边界也可保留；语义无效时退回结构，不按图号、位置或框设置规则，不强制有人车或要求每图补边。</p>
<p>同均值对照在与原E取max之后匹配真正新增强度，只用于辨别“有选择地衰减”是否不同于全体变暗。新增mask位置本来就相同，比较实际浮点强度。</p>
<p>40张FLIR开发图覆盖全部24个训练侧视频，逐视频报告仍不能作为独立泛化证明。MSRS／LLVIP仅使用固定训练侧样本，未运行语义模型，属于无语义结构回退压力测试，不能当作完整语义链跨数据集性能。</p>
<p>本版没有学习或调参搜索；固定规则也可能产生人工设计偏差。未读取正式testB，未训练diffusion，未证明边界准确率或生成质量提高。</p>
<p><a href="frame_statistics.csv">逐图</a> · <a href="video_statistics.csv">逐视频</a> · <a href="target_statistics.csv">目标定位统计</a> · <a href="semantic_proxy_statistics.csv">语义代理分布</a> · <a href="run_manifest.json">冻结设置与来源</a></p></div>'''
    (output/'index.html').write_text(intro+f'<nav>{nav}</nav>'+''.join(cards)+f'<p>{html.escape(json.dumps(summary,ensure_ascii=False))}</p></main></html>',encoding='utf-8')


def run(args):
    if args.output_dir.exists(): raise FileExistsError('Preserve existing outputs')
    torch.set_num_threads(2)
    source, development = preflight(args)
    external, availability = select_external_training(ROOT,args.external_samples)
    settings = ImportanceSettings()
    code = [Path(__file__),ROOT/'pytorch-CycleGAN-and-pix2pix/models/completion_importance.py',
        ROOT/'pytorch-CycleGAN-and-pix2pix/models/contour_completion.py',ROOT/'pytorch-CycleGAN-and-pix2pix/models/clarity_semantic_edges.py',
        ROOT/'pytorch-CycleGAN-and-pix2pix/models/saliency_edges.py',ROOT/'pytorch-CycleGAN-and-pix2pix/models/semantic_edges.py',
        ROOT/'pytorch-CycleGAN-and-pix2pix/models/learned_color_prior.py',ROOT/'scripts/preview_semantic_contours.py',
        ROOT/'scripts/preview_contour_completion.py',ROOT/'scripts/semantic_preview/review.py']
    plan = dict(format='completion_importance_review_v1',settings=asdict(settings),seed=42,
        external_completion_settings=asdict(CompletionSettings()),
        sources=development+external,external_availability=availability,
        source_manifest_sha256=sha256_file(args.completion_dir/'preview_manifest.json'),
        code_sha256={str(p.relative_to(ROOT)):sha256_file(p) for p in code},
        parameters_fixed_before_processing=True,test_split_access=False,
        generalization_scope='24 already-seen FLIR development videos; external train-side structural fallback only')
    output=args.output_dir;output.mkdir(parents=True,exist_ok=False)
    for folder in ('frames','maps','targets','diagnostics'): (output/folder).mkdir()
    write_json(output/'frozen_plan.json',plan)
    plan_digest=hashlib.sha256(json.dumps(plan,sort_keys=True,allow_nan=False).encode()).hexdigest()
    frames,stats,target_stats,semantic_stats,path_stats=[],[],[],[],[]
    started=time.perf_counter(); completed=0
    records=development+external
    write_json(output/'status.json',dict(phase='running',completed=0,total=len(records)))
    try:
        for row in records:
            gray,observed,baseline,metadata,structure,clutter,p,q = load_development(row) if row['dataset']=='FLIR-development' else load_external(row)
            original=json.dumps(metadata,sort_keys=True,allow_nan=False)
            maps,updated=prioritize_completion(observed,metadata,structure,clutter,p,q,settings=settings)
            np.testing.assert_array_equal(maps['baseline'],baseline)
            assert json.dumps(metadata,sort_keys=True,allow_nan=False)==original
            for before,after in zip(metadata['candidates'],updated['candidates']):
                assert all(after[key]==value for key,value in before.items())
            prefix=row['prefix']
            Image.fromarray(gray).save(output/'frames'/f'{prefix}_TIR.png')
            for mode in MODES: uint8_image(maps[mode]).save(output/'frames'/f'{prefix}_{mode}.png')
            make_sheet(gray,maps,modes=MODES,labels=LABELS,importance_key='gain_map',importance_caption='R gain on existing paths').save(output/'frames'/f'{prefix}_comparison.png')
            increments=dict(maps,**{mode:maps[mode+'_increment'] for mode in MODES})
            make_sheet(gray,increments,modes=MODES,labels=LABELS,importance_key='gain_map',importance_caption='R gain / not structural C').save(output/'frames'/f'{prefix}_increments.png')
            np.savez_compressed(output/'maps'/f'{prefix}_importance.npz',TIR_uint8=gray,structure=structure,clutter=clutter,**maps)
            write_json(output/'diagnostics'/f'{prefix}_paths.json',updated)
            old_mass=float(maps['baseline_increment'].sum(dtype=np.float64))
            new_mass=float(maps['priority_increment'].sum(dtype=np.float64))
            mean_error=float(abs(maps['uniform_increment'].mean(dtype=np.float64)-maps['priority_increment'].mean(dtype=np.float64)))
            assert mean_error<1e-8
            frame=dict(prefix=prefix,filename=row['filename'],dataset=row['dataset'],video_id=row.get('video_id','external_sequence_unknown'),
                accepted_paths=sum(c['reliability_accepted'] for c in metadata['candidates']),
                baseline_added_strength=old_mass,priority_added_strength=new_mass,
                retained_fraction=new_mass/old_mass if old_mass>0 else 1.,mean_match_error=mean_error,
                semantic_mode=row['semantic_mode'])
            frames.append(frame)
            for c in updated['candidates']:
                if c['reliability_accepted']:
                    path_stats.append(dict(prefix=prefix,candidate_id=c['id'],C=c['reliability_score'],R=c['add_importance'],gain=c['importance_gain'],**c['importance_features']))
            for mode in MODES:
                inc=maps[mode+'_increment']
                stats.append(dict(**frame,mode=mode,added_strength=float(inc.sum(dtype=np.float64))))
                if p is not None:
                    for i,name in enumerate(FMB_CLASS_NAMES):
                        semantic_stats.append(dict(prefix=prefix,video_id=frame['video_id'],mode=mode,semantic_class=name,
                            probability_area=float(p[i].sum(dtype=np.float64)),soft_added_strength=float((p[i]*inc).sum(dtype=np.float64))))
            if row['dataset']=='FLIR-development':
                for t in source['targets']:
                    if t['filename']!=row['filename']:continue
                    make_sheet(gray,maps,crop=t['crop'],box=t['box'],zoom=args.zoom,modes=MODES,labels=LABELS,
                        importance_key='gain_map',importance_caption='R gain / same paths').save(output/'targets'/f'{t["id"]}.png')
                    yy,xx=np.indices(gray.shape);x0,y0,x1,y1=t['box']
                    mask=(xx+.5>=x0)&(xx+.5<x1)&(yy+.5>=y0)&(yy+.5<y1)
                    for mode in MODES:
                        target_stats.append(dict(id=t['id'],category=t['category'],prefix=prefix,mode=mode,
                            added_strength=float(maps[mode+'_increment'][mask].sum(dtype=np.float64)),pixels=int(mask.sum())))
            completed+=1
            write_json(output/'status.json',dict(phase='running',completed=completed,total=len(records)))
            print(f'[IMPORTANCE {completed:02d}/{len(records)}] {prefix} | same paths={frame["accepted_paths"]} | retained={frame["retained_fraction"]:.3f}',flush=True)
        video_stats=[]
        for dataset,video in sorted({(f['dataset'],f['video_id']) for f in frames}):
            group=[f for f in frames if f['dataset']==dataset and f['video_id']==video]
            before=sum(f['baseline_added_strength'] for f in group);after=sum(f['priority_added_strength'] for f in group)
            video_stats.append(dict(dataset=dataset,video_id=video,images=len(group),baseline_added_strength=before,
                priority_added_strength=after,retained_fraction=after/before if before else 1.))
        for name,rows,columns in [('frame_statistics',stats,('prefix','mode')),('video_statistics',video_stats,('dataset','video_id')),
            ('target_statistics',target_stats,('id','mode')),('semantic_proxy_statistics',semantic_stats,('prefix','mode','semantic_class')),
            ('path_importance',path_stats,('prefix','candidate_id','C','R','gain'))]:
            write_table(output/f'{name}.csv',rows,columns)
        summary=dict(development_images=len(development),development_videos=len({r['video_id'] for r in development}),
            frozen_plan_sha256=plan_digest,
            external_fallback_images=len(external),targets=len(source['targets']),seconds=time.perf_counter()-started,
            max_mean_control_error=max((f['mean_match_error'] for f in frames),default=0.),
            unchanged_geometry_and_C=True,unmodified_observed_E=True,full_external_semantic_generalization_verified=False)
        write_json(output/'run_manifest.json',dict(plan=plan,summary=summary,frames=frames,targets=source['targets']))
        write_page(output,frames,source['targets'],summary)
        write_json(output/'status.json',dict(phase='completed',completed=completed,total=len(records),**summary))
    except BaseException as error:
        try:write_json(output/'status.json',dict(phase='failed',completed=completed,total=len(records),error=str(error)))
        except OSError:pass
        raise
    print(output/'index.html',flush=True)


if __name__=='__main__':run(parse_args())
