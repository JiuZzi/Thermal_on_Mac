"""Prepare and compare frozen, non-learning edge-use interventions.

Pretrained inference is an engineering pilot, not a trained-D quality test.
Formal inference reuses the Windows run, its P/Q, prior and checkpoint. Neither
mode changes model weights, losses, training configuration or cached conditions.
"""

import argparse
from dataclasses import asdict
import hashlib
import html
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import numpy as np
from PIL import Image, ImageDraw
import torch

from D.common import ROOT, DEFAULT_CACHE, DEFAULT_CONFIG, EDGE_SOURCES, read_json, write_json, sha256, load_config, select_sources
from D.unpaired.data import ConditionDataset
from D.unpaired.protocol import DEFAULT_DATA, DEFAULT_PRIOR, DEFAULT_MANIFEST, model_config
from D.unpaired.semantics import ConditionBuilder, default_semantic_checkpoint
from D.unpaired.losses import edge_preservation
from .edge_use import ProbeSettings, VARIANTS, interventions


LABELS = dict(original='Original E', uniform='Uniform / same edge mass', selective='Selective / fixed probe')


def code_identity():
    files = [Path(__file__), Path(__file__).with_name('edge_use.py'), ROOT/'D/common.py', ROOT/'D/models.py',
        ROOT/'D/preview.py', ROOT/'D/unpaired/model.py', ROOT/'D/unpaired/protocol.py', ROOT/'D/unpaired/losses.py',
        ROOT/'D/unpaired/data.py', ROOT/'D/unpaired/checkpoint.py', ROOT/'D/unpaired/semantics.py',
        ROOT/'D/unpaired/report.py', ROOT/'D/adaptation.py',
        ROOT/'pytorch-CycleGAN-and-pix2pix/models/learned_color_prior.py']
    return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest() for p in files}


def prepare(args):
    if args.output_dir.exists(): raise FileExistsError('Preserve existing inputs; choose a new output directory')
    if args.from_existing_audit:
        # Explicit opt-in: these are already-seen train-side development caches,
        # never silently substituted for the requested held-out validation data.
        sys.path.insert(0,str(ROOT/'scripts'))
        from types import SimpleNamespace
        from preview_completion_importance import preflight as audit_preflight, load_development, SOURCE, SEMANTICS, CLARITY
        original, rows = audit_preflight(SimpleNamespace(completion_dir=SOURCE,semantic_preview_dir=SEMANTICS,clarity_dir=CLARITY))
        rows = rows[:args.limit]
        recipe = dict(format='edge_use_probe_inputs_v1', phase='train', seed=42,
            selection='First N of the frozen 40-frame development pack; not held-out validation',
            semantic_checkpoint_loaded=False, semantic_flip_reliability=False,
            semantic_manifest_sha256=sha256(SEMANTICS/'preview_manifest.json'),
            clarity_manifest_sha256=sha256(CLARITY/'preview_manifest.json'),
            input='Frozen TIR01,clarity_v1_E01,zero/P; Q recomputed by the original single-view development recipe',
            geometry='existing native pixels, no crop or stretch', runner_code=code_identity(), frames=[])
        output=args.output_dir;output.mkdir(parents=True,exist_ok=False);(output/'maps').mkdir()
        write_json(output/'manifest.json',recipe);write_json(output/'status.json',dict(phase='preparing',completed=0,total=len(rows)))
        torch.set_num_threads(args.cpu_threads)
        for i,row in enumerate(rows,1):
            gray,E,_,_,_,_,p,q=load_development(row)
            stem=Path(row['filename']).stem;relative=f'maps/{stem}.npz'
            np.savez_compressed(output/relative,gray=gray,E=E,P=p,Q=q)
            recipe['frames'].append(dict(id=stem,filename=row['filename'],stem=stem,group=row['video_id'],
                source=row['paths']['tir'],source_sha256=row['sha256']['tir'],native_size=[gray.shape[1],gray.shape[0]],
                condition=relative,condition_sha256=sha256(output/relative),frozen_audit_inputs=row['sha256']))
            write_json(output/'manifest.json',recipe);write_json(output/'status.json',dict(phase='preparing',completed=i,total=len(rows)))
            print(f'[CACHED DEVELOPMENT {i}/{len(rows)}] {row["filename"]}',flush=True)
        write_json(output/'status.json',dict(phase='completed',completed=len(rows),total=len(rows),development_only=True))
        return output/'manifest.json'
    rows, counts = select_sources(args.data_manifest, args.phase, args.limit, seed=42)
    if not args.semantic_checkpoint.is_file(): raise FileNotFoundError('Existing complete FMB weight is required')
    torch.set_num_threads(args.cpu_threads)
    # Existing TIR segmenter only; do not load/download an additional RGB model.
    builder = ConditionBuilder(args.semantic_checkpoint, rgb=False, flip=True, device='cpu')
    recipe = dict(format='edge_use_probe_inputs_v1', phase=args.phase, seed=42, selection='seed42 balanced videos',
        data_manifest_sha256=sha256(args.data_manifest), counts=counts, tir_semantic_sha256=builder.tir.info['sha256'],
        input='TIR01,clarity_v1_E01,zero; original P and flip-reliability Q', geometry='native input pixels; no crop or stretch',
        source_code_sha256={name:sha256(ROOT/name) for name in EDGE_SOURCES}, runner_code=code_identity(), frames=[])
    output = args.output_dir; output.mkdir(parents=True, exist_ok=False); (output/'maps').mkdir()
    write_json(output/'manifest.json', recipe); write_json(output/'status.json', dict(phase='preparing', completed=0, total=len(rows)))
    for i, row in enumerate(rows, 1):
        with Image.open(row['path']) as im: pixels = np.asarray(im.convert('RGB')).copy()
        gray, E, p, q = builder.compute(pixels, 'TIR')
        relative = f'maps/{row["stem"]}.npz'
        np.savez_compressed(output/relative, gray=gray, E=E, P=p, Q=q)
        recipe['frames'].append(dict(id=row['stem'], filename=row['filename'], stem=row['stem'], group=row['group'],
            source=row['source'], source_sha256=sha256(row['path']), native_size=[gray.shape[1],gray.shape[0]],
            condition=relative, condition_sha256=sha256(output/relative)))
        write_json(output/'manifest.json', recipe); write_json(output/'status.json', dict(phase='preparing', completed=i, total=len(rows)))
        print(f'[CONDITION {i}/{len(rows)}] {row["filename"]}', flush=True)
    write_json(output/'status.json', dict(phase='completed', completed=len(rows), total=len(rows)))
    return output/'manifest.json'


class ProbeDataset:
    def __init__(self, folder):
        self.folder = Path(folder); self.manifest = read_json(self.folder/'manifest.json')
        status = read_json(self.folder/'status.json')
        self.frames = self.manifest.get('frames', [])
        if (self.manifest.get('format') != 'edge_use_probe_inputs_v1' or self.manifest.get('phase') not in ('train','val') or
            status.get('phase') != 'completed' or status.get('completed') != len(self.frames) or not self.frames):
            raise ValueError('Complete train/validation probe inputs required')
        for key in ('id','stem'):
            if len({f[key] for f in self.frames}) != len(self.frames): raise ValueError('Duplicate input/output identities')

    def __len__(self): return len(self.frames)

    def __getitem__(self, i):
        frame = self.frames[i]
        path = (self.folder/frame['condition']).resolve()
        if not path.is_relative_to(self.folder.resolve()) or sha256(path) != frame['condition_sha256']:
            raise ValueError('Condition changed or leaves the cache')
        with np.load(path, allow_pickle=False) as z: gray,E,p,q = [z[k].copy() for k in ('gray','E','P','Q')]
        from D.unpaired.protocol import validate_arrays
        validate_arrays(gray,E,p,q)
        if frame['native_size'] != [gray.shape[1],gray.shape[0]]: raise ValueError('Native geometry mismatch')
        t = torch.from_numpy(gray.astype(np.float32)/255)[None,None]; e = torch.from_numpy(E)[None,None]
        return dict(x=torch.cat((t,e,torch.zeros_like(e)),1), p=torch.from_numpy(p)[None], q=torch.from_numpy(q)[None,None], frame=frame)


def preflight(args):
    if args.backend == 'formal':
        if args.run_dir is None: raise ValueError('Supply the trained Windows D --run-dir')
        from D.unpaired.checkpoint import verify, locate
        run = read_json(args.run_dir/'run.json')
        if run.get('format') != 'D_unpaired_run_v1' or sha256(args.prior) != run['prior_sha256']:
            raise ValueError('Formal run and its original learned prior are required')
        current = code_identity()
        used = ('D/common.py','D/models.py','D/adaptation.py','D/unpaired/model.py',
            'D/unpaired/protocol.py','D/unpaired/semantics.py',
            'pytorch-CycleGAN-and-pix2pix/models/learned_color_prior.py')
        changed = [name for name in used if run.get('code_sha256',{}).get(name) != current[name]]
        if changed:
            raise ValueError('Inference source differs from this trained run; use its matching code: '+', '.join(changed))
        step = args.step
        if args.best:
            from D.unpaired.validation import selected_record
            selected_run, selected_path, record = selected_record(args.run_dir/'best.json', 'primary')
            if selected_run != args.run_dir.resolve(): raise ValueError('Best checkpoint belongs to another run')
            step = record['step']
        checkpoint, step = locate(args.run_dir, step); header = verify(args.run_dir, checkpoint)
        if header['step'] != step: raise ValueError('Checkpoint update count mismatch')
        if args.best and checkpoint.resolve() != selected_path: raise ValueError('Best location changed')
        dataset = ConditionDataset(args.conditions, 'TIR', 'val')
        if dataset.manifest['identity'] != run['data']['source']:
            raise ValueError('P/E/Q cache must match the original training recipe exactly')
        config = run['config']
        identity = dict(kind='trained_D', run_sha256=sha256(args.run_dir/'run.json'), step=step,
            checkpoint=str(checkpoint.resolve()), checkpoint_files=header['files'], prior_sha256=sha256(args.prior),
            config=config, condition_identity=run['data']['source'], independent_semantic_color_branches=True)
        return dataset, config, identity
    dataset = ProbeDataset(args.conditions)
    config = load_config(args.config)
    # Read pinned local snapshots without model loading or any Hub request.
    snapshots = {}
    for kind in ('base','control'):
        snapshot = args.cache_dir/('models--'+config[kind+'_model'].replace('/','--'))/'snapshots'/config[kind+'_revision']
        if not snapshot.is_dir(): raise FileNotFoundError('Missing existing local model snapshot: '+str(snapshot))
        files = sorted(p for p in snapshot.rglob('*') if p.is_file() and p.suffix in ('.json','.txt','.safetensors'))
        if not any(p.suffix == '.safetensors' for p in files): raise FileNotFoundError('Local pretrained weights are incomplete')
        snapshots[kind] = dict(revision=config[kind+'_revision'], files={p.relative_to(snapshot).as_posix():sha256(p) for p in files})
    return dataset, config, dict(kind='pretrained_only', config=config, snapshots=snapshots,
        independent_semantic_color_branches=False, trained_D_checkpoint_used=False,
        limits='Zero-training ControlNet diagnostic; P/Q only constructs a fixed probe; cannot establish trained-D benefit')


def _to_rgb(values):
    values = np.asarray(values)
    if values.ndim != 3 or values.shape[2] != 3 or values.dtype != np.uint8:
        raise ValueError('Unmodified uint8 RGB output required')
    return Image.fromarray(values)


def rgb_metrics(image, sample):
    """Fixed-source diagnostics, never paired quality or correctness metrics."""
    size = (sample['x'].shape[-1],sample['x'].shape[-2])
    rgb = np.asarray(image.resize(size, Image.Resampling.BICUBIC)).astype(np.float32)/255
    values = torch.from_numpy(rgb.transpose(2,0,1).copy())[None]*2-1
    with torch.no_grad(): structural = float(edge_preservation(values,sample['x'][:,:1],sample['x'][:,1:2]))
    luminance = (rgb*np.array([.299,.587,.114],np.float32)).sum(2,keepdims=True)
    chroma = rgb-luminance
    p, q = sample['p'][0].numpy(), sample['q'][0,0].numpy()
    numerator = denominator = 0.
    for axis in (0,1):
        a,b = [slice(None)]*2,[slice(None)]*2; a[axis],b[axis] = slice(1,None),slice(None,-1)
        a,b = tuple(a),tuple(b)
        pa,pb = p[(slice(None),*a)],p[(slice(None),*b)]
        quality = np.minimum(q[a],q[b]); agreement = (pa*pb).sum(0)*quality*(quality>=.5)
        numerator += float((np.abs(chroma[a]-chroma[b]).mean(2)*agreement).sum(dtype=np.float64))
        denominator += float(agreement.sum(dtype=np.float64))
    maximum, minimum = rgb.max(2),rgb.min(2)
    saturation = (maximum-minimum)/np.maximum(maximum,1e-8)
    return dict(source_edge_proxy=structural, same_semantic_chroma_variation=numerator/denominator if denominator else None,
        semantic_pair_weight=denominator, mean_saturation=float(saturation.mean()), chroma_std=float(chroma.std()))


def panel(items):
    w,h = items[0][1].size
    result = Image.new('RGB',(w*len(items),h+26),'white'); draw = ImageDraw.Draw(result)
    for i,(label,image) in enumerate(items):
        if image.size != (w,h): image = image.resize((w,h),Image.Resampling.BICUBIC)
        result.paste(image.convert('RGB'),(i*w,26)); draw.text((i*w+4,5),label,fill='black')
    return result


def _gray(values):
    return Image.fromarray(np.rint(np.clip(values,0,1)*255).astype(np.uint8)).convert('RGB')


def write_page(output, identity, records):
    pilot = identity['model']['kind'] == 'pretrained_only'
    scope = ('本页使用未进行TIR适配的预训练SD1.5/ControlNet。没有加载训练后的D权重、独立语义适配器或颜色分支；仅用于工程与条件敏感性预览。'
        if pilot else '本页固定训练后的D权重、P/Q、颜色先验、采样种子和步数，仅改变E。属于同一权重下的条件干预，不代替新模型训练消融。')
    cards = []
    for frame in identity['frames']:
        stem = frame['stem']
        items = [records.get(variant+'/'+stem) for variant in VARIANTS]
        if not all(items): continue
        cards.append(f'<article><h2>{html.escape(frame["filename"])}</h2><img src="panels/{stem}_rgb.png">'
            f'<details><summary>原轮廓、固定G、实际选择性轮廓及同强度控制</summary><img src="panels/{stem}_conditions.png"></details>'
            '<details><summary>描述性统计，不是质量评分</summary><pre>'+html.escape(json.dumps({v:r['metrics'] for v,r in zip(VARIANTS,items)},ensure_ascii=False,indent=2))+'</pre></details></article>')
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>轮廓使用门控：固定权重抑制诊断</title>'
    page += '<style>body{font:16px/1.7 system-ui;background:#f3f5f7;color:#182536;margin:20px}main{max-width:1700px;margin:auto}article,.notice{padding:18px;border:1px solid #ccd4df;background:white;margin:16px 0;border-radius:8px}.notice{background:#fff2d5}img{max-width:100%}h2{overflow-wrap:anywhere}summary{cursor:pointer}pre{white-space:pre-wrap}</style><main>'
    page += '<h1>原E → 统一减弱 → 空间选择性减弱</h1><div class="notice"><p>'+scope+'</p>'
    page += '<p>G是固定规则诊断图，尚未学习，也不是边缘正确概率。规则不按道路／植被类别清零，不使用框、图号、固定上下位置或生成RGB。</p>'
    page += '<p>每组使用相同模型对象、CPU噪声种子42、原始TIR及相同推理参数。统一减弱与选择性组在实际模型输入尺寸上匹配E总强度。RGB未作调色／锐化，另存模型分辨率原图。</p>'
    page += '<p>先检查颜色是否只变灰、人体／车辆是否位移、真实边界是否被抹掉，再看类内色度变化。统计使用同一源P/Q和原E，仍可能受错误语义影响；不报告小样本FID或伪造正确率。</p>'
    page += '<p><a href="run.json">冻结身份与范围</a> · <a href="records.json">输入输出哈希、时间与统计</a></p></div>'
    (output/'index.html').write_text(page+''.join(cards)+'</main></html>',encoding='utf-8')


def compare_samples(dataset, config, model_identity, output, predictor, *, device='cpu', resume=False, limit=4):
    """Model-independent transaction; predictor returns (model-size PIL RGB, flag)."""
    selected = dataset.frames[:limit] if limit else dataset.frames
    if not selected: raise ValueError('No inputs selected')
    if any(Path(f['stem']).name != f['stem'] or f['stem'] in ('.','..') for f in selected): raise ValueError('Unsafe output stems')
    model_size = (config['height'],config['width'])
    identity = dict(format='edge_use_intervention_v1', model=model_identity, frames=selected,
        conditions_sha256=sha256(dataset.folder/'manifest.json'), settings=asdict(ProbeSettings()),
        variants=list(VARIANTS), model_size=list(model_size), seed=42, eta=0., code_sha256=code_identity(),
        runtime=dict(device=device,python=platform.python_version(),system=platform.system(),
            versions={k:metadata.version(k) for k in ('torch','diffusers','transformers','numpy','Pillow')}),
        learned_gate=False, training_performed=False, independent_quality_verified=False)
    output = Path(output)
    if output.exists():
        if not resume or read_json(output/'run.json') != identity: raise ValueError('Preserve output; resume requires identical recipe and inputs')
        records = read_json(output/'records.json')
        artifacts = read_json(output/'artifacts.json') if (output/'artifacts.json').is_file() else {}
    else:
        if resume: raise ValueError('Nothing to resume')
        output.mkdir(parents=True)
        for sub in ('panels','conditions','maps',*VARIANTS):
            (output/sub).mkdir()
        for variant in VARIANTS:
            (output/variant/'images').mkdir();(output/variant/'model_resolution').mkdir()
        records = {}; artifacts = {}; write_json(output/'run.json',identity);write_json(output/'records.json',records)
        write_json(output/'artifacts.json',artifacts)
    total = len(selected)*len(VARIANTS)
    write_json(output/'status.json',dict(phase='running',completed=len(records),total=total))
    try:
        for i, frame in enumerate(selected):
            sample = dataset[i]
            if sample['frame'] != frame: raise ValueError('Dataset order changed')
            x,p,q = sample['x'],sample['p'],sample['q']
            before = [v.clone() for v in (x,p,q)]
            variants,maps,diagnostics = interventions(x,p,q,model_size=model_size)
            stem = frame['stem']; source_gray = _gray(x[0,0].numpy()); w,h = source_gray.size
            if stem in artifacts:
                for relative,digest in artifacts[stem].items():
                    if sha256(output/relative) != digest: raise ValueError('Saved diagnostic artifact changed: '+relative)
            has_artifacts = stem in artifacts
            images = []
            payload = dict(TIR_uint8=np.rint(x[0,0].numpy()*255).astype(np.uint8),P=p[0].numpy(),Q=q[0,0].numpy(),
                **{k:v[0,0].numpy() for k,v in maps.items()},**{v:condition[0,1].numpy() for v,condition in variants.items()})
            for variant in VARIANTS:
                key = variant+'/'+stem; target=output/variant/'images'/f'{stem}_fake_B.png'; model_target=output/variant/'model_resolution'/f'{stem}.png'
                if key in records:
                    record=records[key]
                    if sha256(target)!=record['sha256'] or sha256(model_target)!=record['model_sha256']:
                        raise ValueError('Saved RGB changed: '+key)
                    with Image.open(model_target) as im: image=im.convert('RGB')
                else:
                    tick=time.perf_counter()
                    image,flagged=predictor(variants[variant].clone(),p.clone(),q.clone())
                    if image.mode!='RGB' or image.size!=(model_size[1],model_size[0]): raise ValueError('Unexpected model output geometry/mode')
                    if not np.asarray(image).any() and not flagged: raise FloatingPointError('Unfiltered black output; check precision')
                    image.save(model_target);image.resize(tuple(frame['native_size']),Image.Resampling.BICUBIC).save(target)
                    record=dict(sha256=sha256(target),model_sha256=sha256(model_target),condition_sha256=frame['condition_sha256'],
                        metrics=None if flagged else rgb_metrics(image,sample),safety_flagged=flagged,seed=42,
                        seconds=time.perf_counter()-tick,**diagnostics[0])
                    records[key]=record;write_json(output/'records.json',records)
                    write_json(output/'status.json',dict(phase='running',completed=len(records),total=total))
                    print(f'[PROBE {len(records):02d}/{total}] {variant} | {frame["filename"]} | {record["seconds"]:.1f}s',flush=True)
                images.append((LABELS[variant],image.resize((w,h),Image.Resampling.BICUBIC)))
            if any(not torch.equal(a,b) for a,b in zip((x,p,q),before)): raise AssertionError('Original TIR/E/P/Q mutated')
            if not has_artifacts:
                np.savez_compressed(output/'maps'/f'{stem}.npz',**payload)
                panel([('TIR',source_gray),*images]).save(output/'panels'/f'{stem}_rgb.png')
                panel([('TIR',source_gray),('Original E',_gray(x[0,1].numpy())),('Fixed G / not confidence',_gray(maps['gate'][0,0].numpy())),
                    ('Uniform E',_gray(variants['uniform'][0,1].numpy())),('Selective E',_gray(variants['selective'][0,1].numpy()))]).save(output/'panels'/f'{stem}_conditions.png')
                relatives=(f'maps/{stem}.npz',f'panels/{stem}_rgb.png',f'panels/{stem}_conditions.png')
                artifacts[stem]={name:sha256(output/name) for name in relatives}
                write_json(output/'artifacts.json',artifacts)
            write_page(output,identity,records)
        write_json(output/'status.json',dict(phase='completed',completed=len(records),total=total,trained_gate=False,quality_verified=False))
    except BaseException as error:
        write_json(output/'status.json',dict(phase='failed',completed=len(records),total=total,error=str(error)))
        raise
    return output/'index.html'


def load_predictor(args, config, identity):
    from D.models import load_pipeline
    if args.backend == 'formal':
        if args.device != 'cuda:0' or not torch.cuda.is_available(): raise ValueError('Trained formal D currently requires cuda:0; use --check-only on Mac')
        from safetensors.torch import load_file
        from D.unpaired.model import FormalGenerator
        from D.unpaired.report import to_image
        print(f'[LOAD] Fixed formal D update {identity["step"]}; offline pretrained components',flush=True)
        pipeline=load_pipeline(model_config(config),args.cache_dir,offline=True)
        model=FormalGenerator(pipeline,config,args.prior,identity['condition_identity']['recipe']).to(args.device)
        model.import_state(load_file(str(Path(identity['checkpoint'])/'generator.safetensors')))
        model.eval().requires_grad_(False);del pipeline
        versions={name:p._version for name,p in model.named_parameters()}
        def predict(x,p,q):
            with torch.inference_mode():
                values=model(x.to(args.device),p.to(args.device),q.to(args.device),generator=torch.Generator('cpu').manual_seed(42))
            if versions!={name:p._version for name,p in model.named_parameters()}: raise AssertionError('Model weights changed during intervention')
            return to_image(values),None
        return predict
    if args.device=='mps' and not torch.backends.mps.is_available(): raise RuntimeError('MPS unavailable in this process; request native GPU access or use Windows CUDA')
    if args.device=='cuda:0' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    from D.preview import generate_one
    print('[LOAD] Existing pretrained SD1.5/ControlNet; offline; no trained-D weights',flush=True)
    pipeline=load_pipeline(config,args.cache_dir,offline=True);pipeline.disable_vae_tiling();pipeline.to(args.device)
    pipeline.set_progress_bar_config(disable=True)
    versions={(part,name):p._version for part in ('unet','vae','controlnet','text_encoder') for name,p in getattr(pipeline,part).named_parameters()}
    def predict(x,p,q):
        initial=Image.fromarray(np.rint(x[0,0].numpy()*255).astype(np.uint8)).convert('RGB')
        image,flagged=generate_one(pipeline,initial,x[0,1].numpy(),config,'D1')
        if args.device=='mps': torch.mps.synchronize()
        if versions!={(part,name):p._version for part in ('unet','vae','controlnet','text_encoder') for name,p in getattr(pipeline,part).named_parameters()}:
            raise AssertionError('Pretrained model weights changed')
        return image,flagged
    return predict


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='action',required=True)
    prep=commands.add_parser('prepare');prep.add_argument('--data-manifest',type=Path,default=DEFAULT_MANIFEST)
    prep.add_argument('--semantic-checkpoint',type=Path,default=default_semantic_checkpoint())
    prep.add_argument('--phase',choices=('train','val'),default='val');prep.add_argument('--limit',type=int,default=4)
    prep.add_argument('--from-existing-audit',action='store_true',help='Explicitly reuse fixed train-development P/E caches when the semantic weight is absent; NOT validation')
    prep.add_argument('--output-dir',type=Path,default=ROOT/'runs/D/edge_use_inputs_seed42_v1')
    prep.add_argument('--cpu-threads',type=int,default=2)
    run=commands.add_parser('compare');run.add_argument('--backend',choices=('formal','pretrained'),required=True)
    run.add_argument('--conditions',type=Path);run.add_argument('--run-dir',type=Path)
    choice=run.add_mutually_exclusive_group();choice.add_argument('--step',type=int);choice.add_argument('--best',action='store_true')
    run.add_argument('--prior',type=Path,default=DEFAULT_PRIOR);run.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    run.add_argument('--cache-dir',type=Path,default=DEFAULT_CACHE);run.add_argument('--device',choices=('cuda:0','mps'),default='cuda:0')
    run.add_argument('--limit',type=int,default=4);run.add_argument('--output-dir',type=Path,required=True)
    run.add_argument('--resume',action='store_true');run.add_argument('--check-only',action='store_true')
    args=parser.parse_args(argv)
    if args.limit<1: parser.error('Positive bounded diagnostic limit required; use formal evaluation for full test metrics')
    if args.action=='prepare':
        if args.cpu_threads<1: parser.error('Positive CPU thread count required')
        print(prepare(args));return
    if args.backend=='pretrained' and (args.run_dir is not None or args.step is not None or args.best): parser.error('Pretrained mode cannot claim a trained checkpoint')
    args.conditions=args.conditions or (DEFAULT_DATA if args.backend=='formal' else ROOT/'runs/D/edge_use_inputs_seed42_v1')
    if args.output_dir.exists() and not args.resume and not args.check_only: parser.error('Preserve existing output')
    torch.set_num_threads(2)
    dataset,config,identity=preflight(args)
    # Verify selected arrays before constructing any large model.
    for i in range(min(args.limit,len(dataset))):
        sample=dataset[i]; shape=sample['x'].shape[-2:]
        if shape[1]*config['height'] != shape[0]*config['width']: parser.error('Model size would stretch TIR geometry')
        if args.backend=='formal' and tuple(shape)!=(config['height'],config['width']): parser.error('Formal input size differs from the run')
        interventions(sample['x'],sample['p'],sample['q'],model_size=(config['height'],config['width']))
    if args.check_only:
        print(json.dumps(dict(backend=args.backend,frames=min(args.limit,len(dataset)),model=identity,large_model_loaded=False,output_created=False),ensure_ascii=False,indent=2));return
    torch.manual_seed(42)
    if args.device=='cuda:0':
        torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    predictor=load_predictor(args,config,identity)
    print(compare_samples(dataset,config,identity,args.output_dir,predictor,device=args.device,resume=args.resume,limit=args.limit),flush=True)


if __name__=='__main__': main()
