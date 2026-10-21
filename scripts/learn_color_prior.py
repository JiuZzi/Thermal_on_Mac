#!/usr/bin/env python3
"""Prepare grouped RGB data, cache a frozen RGB teacher, fit/evaluate colors."""

import argparse
import csv
import colorsys
import hashlib
import html
import json
import os
from pathlib import Path
import random

import numpy as np

from color_prior_learning.data import digest, load_manifest, write_json
from color_prior_learning.fit import evaluate_bank, fit_bank


def prepare(args):
    if not 0 < args.val_fraction < .5:
        raise ValueError("Use a predeclared validation group fraction in (0,.5)")
    with args.csv.open(newline="",encoding="utf-8-sig") as handle:
        records=[row for row in csv.DictReader(handle) if row[args.source_role_column]==args.source_train_value]
    if not records:
        raise ValueError("No source rows matched the explicitly selected role")
    groups=sorted({row[args.group_column] for row in records})
    if any(not value for value in groups) or (not args.evaluation_only and len(groups)<4):
        raise ValueError("Explicit video/scene IDs and enough independent groups are required")
    random.Random(args.seed).shuffle(groups)
    held=set(groups[:max(2,round(len(groups)*args.val_fraction))])
    output=args.output.resolve(); image_root=args.image_root.resolve()
    rows=[]
    for row in sorted(records,key=lambda r:r[args.image_column]):
        image=(image_root/row[args.image_column]).resolve()
        if not image.is_relative_to(image_root) or not image.is_file():
            raise ValueError("Invalid source RGB image path")
        identity=hashlib.sha256(row[args.image_column].encode()).hexdigest()[:20]
        rows.append(dict(id=identity,image=os.path.relpath(image,output.parent),label=f'labels/{identity}.png',
                         group=row[args.group_column],split='test' if args.evaluation_only else
                         ('val' if row[args.group_column] in held else 'train')))
    manifest=dict(format='rgb_chroma_manifest_v1',dataset_id=args.dataset_id,class_names=args.classes,images=rows,
                  source_csv_sha256=digest(args.csv),source_role_filter={args.source_role_column:args.source_train_value},
                  split_seed=args.seed,group_definition=args.group_column,
                  labels='pending frozen RGB teacher cache; IDs follow class_names, 255=ignore')
    write_json(output,manifest)
    load_manifest(output)
    print(f'Prepared {len(rows)} images/{len(groups)} groups: {output}',flush=True)


def cache(args):
    from PIL import Image
    import torch
    from torch.nn import functional as F
    from scipy.ndimage import binary_erosion
    try:
        from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
        from huggingface_hub import snapshot_download
        import transformers
    except ImportError as error:
        raise RuntimeError('RGB teacher needs optional scripts/requirements_learned_color_teacher.txt') from error
    if args.output_dir.exists():
        raise ValueError('Use a new teacher output directory; existing caches are preserved')
    if not (0<args.min_probability<=1 and 0<=args.min_margin<=1 and 0<=args.min_agreement<=1 and args.erosion>=0):
        raise ValueError('Invalid, predeclared teacher mask settings')
    manifest,base=load_manifest(args.manifest)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable')
    torch.set_num_threads(args.cpu_threads)
    if Path(args.teacher).is_dir():
        snapshot=Path(args.teacher).resolve()
    else:
        # Resolve once: processor and weights must come from the same commit.
        snapshot=Path(snapshot_download(args.teacher,revision=args.revision,
            local_files_only=not args.allow_download,
            allow_patterns=['config.json','preprocessor_config.json','*.safetensors','*.bin','*.index.json']))
    processor=AutoImageProcessor.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False)
    # Transformers prefers safetensors; any .bin load remains restricted.
    if tuple(int(part) for part in torch.__version__.split('+')[0].split('.')[:2]) < (2,6):
        raise ValueError('Use PyTorch >=2.6 for restricted RGB teacher checkpoint loading')
    model=AutoModelForSemanticSegmentation.from_pretrained(snapshot,local_files_only=True,
                                                         trust_remote_code=False,weights_only=True)
    model.eval().requires_grad_(False).to(args.device)
    mapping={str(name).strip().lower():int(index) for index,name in model.config.id2label.items()}
    if any(name.lower() not in mapping for name in manifest['class_names']):
        raise ValueError('Canonical names absent from RGB teacher; align mapping explicitly')
    weights=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        weights.update(name.encode()); weights.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    args.output_dir.mkdir(parents=True)
    (args.output_dir/'labels').mkdir()
    (args.output_dir/'preview').mkdir()
    preview_groups=sorted({r['group'] for r in manifest['images'] if r['split']!='test' or args.include_test})
    random.Random(42).shuffle(preview_groups)
    preview_groups=set(preview_groups[:getattr(args,'preview_groups',24)])
    previewed=set(); cards=[]
    display=np.asarray([np.asarray(colorsys.hsv_to_rgb(i/max(1,len(manifest['class_names'])),.65,.85))*255
                        for i in range(len(manifest['class_names']))],dtype=np.uint8)
    rows=[]; counts={name:0 for name in manifest['class_names']}
    for row in manifest['images']:
        if row['split']=='test' and not args.include_test:
            continue
        image_path=(base/row['image']).resolve()
        image=Image.open(image_path).convert('RGB'); width,height=image.size
        views=[]
        with torch.inference_mode():
            for view in ([image,image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)] if args.flip else [image]):
                inputs=processor(images=view,return_tensors='pt').to(args.device)
                logits=model(**inputs).logits.float()
                p=F.interpolate(logits,size=(height,width),mode='bilinear',align_corners=False).softmax(1)[0]
                views.append(p)
            if args.flip:
                views[1]=views[1].flip(-1)
            agreement=1-.5*(views[0]-views[-1]).abs().sum(0)
            p=torch.stack(views).mean(0)
            top=p.topk(2,dim=0)
            keep=(top.values[0]>=args.min_probability)&((top.values[0]-top.values[1])>=args.min_margin)&(agreement>=args.min_agreement)
            predicted=p.argmax(0).cpu().numpy(); keep=keep.cpu().numpy()
        labels=np.full((height,width),255,np.uint8)
        for index,name in enumerate(manifest['class_names']):
            interior=predicted==mapping[name.lower()]
            if args.erosion:
                interior=binary_erosion(interior,iterations=args.erosion,border_value=0)
            labels[interior&keep]=index
            counts[name]+=int((labels==index).sum())
        label_path=args.output_dir/'labels'/f"{row['id']}.png"
        Image.fromarray(labels).save(label_path)
        if row['group'] in preview_groups and row['group'] not in previewed:
            previewed.add(row['group'])
            full=np.full((height,width,3),100,np.uint8)
            core=np.full_like(full,100)
            for index,name in enumerate(manifest['class_names']):
                full[predicted==mapping[name.lower()]]=display[index]
                core[labels==index]=display[index]
            Image.fromarray(full).save(args.output_dir/'preview'/f"{row['id']}_classes.png")
            Image.fromarray(core).save(args.output_dir/'preview'/f"{row['id']}_retained.png")
            image_rel=html.escape(os.path.relpath(image_path,args.output_dir.resolve()),quote=True)
            cards.append(f'<article><h2>{html.escape(str(row["group"]))} / {row["split"]}</h2>'
                         f'<img src="{image_rel}"><img src="preview/{row["id"]}_classes.png">'
                         f'<img src="preview/{row["id"]}_retained.png"></article>')
        rows.append(dict(row,image=os.path.relpath(image_path,args.output_dir.resolve()),label=f"labels/{row['id']}.png"))
        print(f'{len(rows)} / {len(manifest["images"])} RGB teacher',flush=True)
    teacher=dict(model=args.teacher,revision_requested=args.revision,resolved_commit=snapshot.name,
                 tensor_sha256=weights.hexdigest(),transformers=transformers.__version__,torch=torch.__version__,
                 processor=processor.to_dict(),flip=args.flip,min_probability=args.min_probability,
                 min_margin=args.min_margin,min_agreement=args.min_agreement,erosion=args.erosion,
                 limits='RGB pseudo labels, not ground truth; high scores do not establish correctness')
    result=dict(manifest,images=rows,teacher=teacher,source_manifest_sha256=digest(args.manifest))
    write_json(args.output_dir/'manifest.json',result)
    write_json(args.output_dir/'cache_summary.json',dict(images=len(rows),class_pixels=counts,
               skipped_test_images=sum(r['split']=='test' for r in manifest['images']) if not args.include_test else 0))
    legend=' '.join(f'<span style="background:rgb{tuple(int(v) for v in color)}">{html.escape(name)}</span>'
                    for name,color in zip(manifest['class_names'],display))
    (args.output_dir/'index.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px system-ui;max-width:1400px;margin:20px auto}img{width:32%;height:auto}article{border:1px solid #ccc;padding:12px;margin:16px 0}span{padding:5px;margin:5px}</style>'
        '<h1>RGB教师审查</h1><p>每行：原RGB／目标类别预测／筛选后保留区域。灰色为其他或忽略；显示色不是颜色先验。分数高仍可能分错，不是精度证据。</p>'
        +legend+''.join(cards),encoding='utf-8')
    print('Cached manifest:',args.output_dir/'manifest.json',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('prepare')
    p.add_argument('--csv',type=Path,required=True); p.add_argument('--image-root',type=Path,required=True)
    p.add_argument('--dataset-id',required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--image-column',default='image'); p.add_argument('--group-column',default='group')
    p.add_argument('--source-role-column',default='split'); p.add_argument('--source-train-value',default='train')
    p.add_argument('--classes',nargs='+',required=True); p.add_argument('--val-fraction',type=float,default=.2)
    p.add_argument('--seed',type=int,default=42); p.add_argument('--evaluation-only',action='store_true')
    p=sub.add_parser('cache')
    p.add_argument('--manifest',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--teacher',required=True); p.add_argument('--revision',required=True,help='Pinned HF revision or local snapshot identity')
    p.add_argument('--allow-download',action='store_true'); p.add_argument('--device',default='cpu')
    p.add_argument('--cpu-threads',type=int,default=4); p.add_argument('--include-test',action='store_true')
    p.add_argument('--flip',action='store_true'); p.add_argument('--min-probability',type=float,default=.9)
    p.add_argument('--min-margin',type=float,default=.2); p.add_argument('--min-agreement',type=float,default=.9)
    p.add_argument('--erosion',type=int,default=2)
    p.add_argument('--preview-groups',type=int,default=24)
    p=sub.add_parser('fit')
    p.add_argument('--manifest',type=Path,required=True); p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('evaluate')
    p.add_argument('--manifest',type=Path,required=True); p.add_argument('--prior',type=Path,required=True)
    p.add_argument('--role',choices=('val','test'),default='test'); p.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.action in ('prepare','cache'):
        globals()[args.action](args)
    elif args.action=='fit':
        bank=fit_bank(args.manifest,args.output,json.loads(args.config.read_text()))
        print(json.dumps({name:dict(status=item['status'],support=item['support'],components=item.get('selected_components'))
                          for name,item in bank['classes'].items()},ensure_ascii=False,indent=2))
    else:
        print(json.dumps(evaluate_bank(args.prior,args.manifest,args.output,args.role),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
