#!/usr/bin/env python3
"""Compare C2c, semantic512 v1 and structural protection on fixed trainB frames."""

import argparse
import hashlib
import html
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
from PIL import Image
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pytorch-CycleGAN-and-pix2pix"))

from models.conditioned_generator import ConditionedGenerator
from models.semantic_edges import make_structure_semantic_contours, SETTINGS, IMPORTANT_CLASSES
from preview_semantic_contours import DEFAULT_SOURCE, DEFAULT_OUTPUT as PREVIOUS_OUTPUT
from preview_semantic_contours import preflight, read_cache, make_sheet, weighted_mean
from preview_semantics import DEFAULT_REVIEW, write_json
from semantic_preview.checkpoint import sha256_file
from semantic_preview.inference import CLASS_NAMES
from semantic_preview.review import uint8_image, red_overlay, write_csv

DEFAULT_OUTPUT = ROOT / "runs/edge_audit/semantic512_structure_joint_seed42_v2"
MODES = ("c2c", "semantic512", "joint512")
LABELS = {"c2c": "C2c", "semantic512": "Semantic v1", "joint512": "Struct + Sem"}
RULE = {
    "name": "bounded_structural_protection_v1",
    "advantage": "clip((fine - 0.5*coarse)/(fine+1e-6), 0, 1)",
    "gate": "(1-S)*(1-D)*advantage",
    "importance": "R_sem + gate*max(R_structure-R_sem, 0)",
    "new_tuned_gains": "none; candidate/fusion/floor settings inherited unchanged",
}


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic-preview-dir",type=Path,default=DEFAULT_SOURCE)
    parser.add_argument("--previous-contours-dir",type=Path,default=PREVIOUS_OUTPUT)
    parser.add_argument("--input-dir",type=Path,default=ROOT/"FLIR_datasets/trainB")
    parser.add_argument("--review-manifest",type=Path,default=DEFAULT_REVIEW)
    parser.add_argument("--protocol-manifest",type=Path,default=ROOT/"FLIR_protocol_v2/manifest.csv")
    parser.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument("--num-images",type=int,default=40)
    parser.add_argument("--zoom",type=int,default=6)
    args=parser.parse_args(argv)
    if args.num_images<1 or not 1<=args.zoom<=12:
        parser.error("Positive image count and zoom 1..12 required")
    if args.input_dir.resolve()!=(ROOT/"FLIR_datasets/trainB").resolve():
        parser.error("Only fixed trainB development images may select this variant")
    if args.output_dir.exists():
        parser.error("Use a new output directory; earlier outputs and ratings are preserved")
    return args


def previous_preflight(args,semantic_manifest,selected,cache_hashes):
    path=args.previous_contours_dir
    manifest=json.loads((path/"preview_manifest.json").read_text(encoding="utf-8"))
    status=json.loads((path/"status.json").read_text(encoding="utf-8"))
    if manifest.get("format")!="semantic_contour_preview_v1" or status.get("phase")!="completed":
        raise ValueError("A complete semantic512-v1 contour preview is required")
    if status.get("completed")!=len(manifest["frames"]):
        raise ValueError("Previous contour preview is incomplete")
    if manifest["identity"]["source_pack_id"]!=semantic_manifest["pack_id"]:
        raise ValueError("Previous contours came from a different semantic model/review")
    if manifest["identity"]["settings"]!=json.loads(json.dumps(SETTINGS)):
        raise ValueError("Previous contours used different candidate/fusion settings")
    frames={frame["filename"]:frame for frame in manifest["frames"]}
    hashes,paths={},{}
    for row in selected:
        name=row["filename"]
        if name not in frames or manifest["identity"]["cache_sha256"].get(name)!=cache_hashes[name]:
            raise ValueError(f"Previous source cache mismatch: {name}")
        prefix=frames[name]["prefix"]
        if not prefix.isdigit():raise ValueError("Previous contour prefix must be numeric")
        old=path/"maps"/f"{prefix}_contours.npz"
        with np.load(old,allow_pickle=False) as archive:
            if "semantic512" not in archive or archive["semantic512"].shape!=(288,360):
                raise ValueError(f"Invalid previous contour file: {old}")
        paths[name]=old;hashes[name]=sha256_file(old)
    return manifest,paths,hashes


def render(gray,maps,crop=None,box=None,zoom=1):
    return make_sheet(gray,maps,crop,box,zoom,modes=MODES,labels=LABELS,
                      importance_key="joint_importance",importance_caption="R joint / region")


def write_page(path,frames,targets,pack_id):
    esc=html.escape
    navigation='<nav>'+''.join(f'<a href="#frame-{f["prefix"]}">{f["prefix"]}</a>' for f in frames)+'</nav>'
    cards=[]
    for frame in frames:
        prefix=frame["prefix"]
        panels=[("TIR","原始红外"),("c2c","C2c：结构显著性"),("semantic512","语义512 v1"),
                ("joint512","结构＋语义：保护版"),("joint512_overlay","保护版叠加"),
                ("base_importance","原结构重要性 R₀"),("semantic_importance","语义重要性 Rₛ"),
                ("joint_importance","联合重要性 Rⱼ"),("structure_protection_gate","结构补充权重（非置信度）")]
        grid=''.join(f'<figure><figcaption>{esc(title)}</figcaption><a href="frames/{prefix}_{key}.png" target="_blank"><img loading="lazy" alt="{esc(title)}" src="frames/{prefix}_{key}.png"></a></figure>' for key,title in panels)
        crops=[]
        for target in targets:
            if target["filename"]!=frame["filename"]:continue
            w,h=target["box_size_pixels"];src=f'targets/{target["panel"]}'
            crops.append(f'<section id="{esc(target["id"])}"><h4>{esc(target["id"])} / {esc(target["category"])} / {w:.1f}×{h:.1f}像素</h4>'
                         '<p>从左到右：红外、C2c、语义v1、结构＋语义、联合R。上行灰度，下行叠加。黄色框只定位审查对象。</p>'
                         f'<a href="{src}" target="_blank"><img class="crop" loading="lazy" alt="{esc(target["id"])}轮廓对照" src="{src}"></a></section>')
        cards.append(f'<article id="frame-{prefix}"><h3>{prefix} / {esc(frame["group"])} / {esc(frame["filename"])}</h3><div class="grid">{grid}</div>'
                     f'<details><summary>目标裁剪（{len(crops)}个）</summary>{"".join(crops)}</details></article>')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>结构显著性＋512语义：轮廓保护对照</title><style>
body{font:16px/1.6 system-ui,sans-serif;margin:20px;background:#f4f6f8;color:#18212c}main{max-width:1320px;margin:auto}
article,.notice{background:white;padding:18px;border:1px solid #ccd4de;border-radius:8px;margin:18px 0}.notice{background:#fff4d7}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px}figure{margin:0}figure img{width:100%}
.crop{max-width:100%;height:auto;image-rendering:pixelated}section{border-top:1px solid #ddd;margin-top:18px}h3{overflow-wrap:anywhere}
nav{position:sticky;top:0;background:white;padding:8px;z-index:1}nav a{display:inline-block;padding:2px 5px}summary{cursor:pointer;color:#2459a0}
</style><main><h1>结构显著性＋512语义：轮廓保护对照</h1>
<div class="notice"><p>同40张固定trainB开发图、同512冻结语义响应；候选、D、融合与25%细边保底固定。只改变R的组合方式。</p>
<p>原语义v1是控制；结构保护只在原结构权重更高、语义支持弱且细候选占优时增加R。密集纹理的补充力度较小。</p>
<p>此次不改变语义类别，也不会恢复所有候选都遗漏的边。保护版不压弱语义v1的像素，可能同时恢复树枝或其他错误纹理。</p>
<p>重点检查02_O03、08_O09、29_O02三个误压弱行人，并检查全图中叶片/路面杂纹是否回升。框内强度与预测类别区域统计均不是准确率。</p>
<p>上行轮廓、下行红色叠加统一使用[0,1]。R与结构补充权重不是后续边缘置信度；第三通道保持零。</p>
<p>除了新候选，另存与保护版具有相同全图平均强度的语义v1整体缩放对照（在每帧展开项中）；该对照只用于排查整体变亮。</p></div>'''
    # A per-frame brightness-matched control uses only full-image mean, not target boxes.
    for i,frame in enumerate(frames):
        prefix=frame["prefix"]
        diagnostics=f'<details><summary>同全图平均强度对照</summary><div class="grid"><figure><figcaption>语义v1整体缩放（仅显示对照）</figcaption><img loading="lazy" src="frames/{prefix}_semantic_global_match.png"></figure><figure><figcaption>结构＋语义</figcaption><img loading="lazy" src="frames/{prefix}_joint512.png"></figure></div></details>'
        cards[i]=cards[i].replace('</article>',diagnostics+'</article>')
    path.write_text(page+navigation+''.join(cards)+f'<p>Pack: {esc(pack_id)}</p></main></html>',encoding="utf-8")


def match_global_mean(original,reference):
    """One global clipped gain, with no target or semantic-region information."""
    target=float(reference.mean(dtype=np.float64))
    if target==0:return np.zeros_like(original),1.0
    if not original.any():raise ValueError("Cannot scale an empty semantic map to a nonzero target")
    lo,hi=1.0,2.0
    while float(np.minimum(original*hi,1).mean(dtype=np.float64))<target:
        hi*=2
        if not np.isfinite(hi):raise ValueError("Could not bracket the global mean")
    for _ in range(45):
        mid=(lo+hi)/2
        if float(np.minimum(original*mid,1).mean(dtype=np.float64))<target:lo=mid
        else:hi=mid
    return np.minimum(original*hi,1).astype(np.float32),hi


def run(args):
    torch.set_num_threads(1)
    source,selected,source_targets,cache_paths,cache_hashes=preflight(args)
    previous,previous_paths,previous_hashes=previous_preflight(args,source,selected,cache_hashes)
    code_paths=[Path(__file__),ROOT/"scripts/preview_semantic_contours.py",
                ROOT/"pytorch-CycleGAN-and-pix2pix/models/semantic_edges.py",
                ROOT/"pytorch-CycleGAN-and-pix2pix/models/saliency_edges.py",
                ROOT/"pytorch-CycleGAN-and-pix2pix/models/conditioned_generator.py",
                ROOT/"scripts/semantic_preview/review.py",ROOT/"scripts/preview_semantics.py"]
    identity=dict(source_semantic_pack=source["pack_id"],previous_pack=previous["pack_id"],
                  source_manifest_sha256=sha256_file(args.semantic_preview_dir/"preview_manifest.json"),
                  previous_manifest_sha256=sha256_file(args.previous_contours_dir/"preview_manifest.json"),
                  semantic_cache_sha256=cache_hashes,previous_output_sha256=previous_hashes,
                  code_hashes={str(p.relative_to(ROOT)):sha256_file(p) for p in code_paths},
                  settings=SETTINGS,rule=RULE,important_classes=IMPORTANT_CLASSES,selected=selected,zoom=args.zoom)
    pack_id=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    output=args.output_dir
    for name in ("frames","maps","targets"):(output/name).mkdir(parents=True,exist_ok=False)
    controls={"c2b":ConditionedGenerator(nn.Identity(),"soft_multi",fusion_mode="direct"),
              "c2c":ConditionedGenerator(nn.Identity(),"soft_saliency",fusion_mode="direct")}
    frames,targets,frame_rows,target_rows=[],[],[],[]
    started=time.perf_counter()
    write_json(output/"status.json",dict(phase="running",completed=0,total=len(selected)))
    print(f"Preflight OK: {len(selected)} fixed frames/{len(source_targets)} targets; semantic-v1 preserved as control.",flush=True)
    try:
        for number,row in enumerate(selected,1):
            name=row["filename"]
            with Image.open(args.input_dir/name) as image:gray=np.asarray(image).copy()
            _,probabilities=read_cache(cache_paths[name],gray)
            tir=torch.from_numpy((gray.astype(np.float32)/255-.5)/.5)[None,None]
            image=np.clip((tir.numpy()[0,0]+1)/2,0,1)
            maps=make_structure_semantic_contours(image,probabilities,CLASS_NAMES)
            with np.load(previous_paths[name],allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive['TIR_uint8'],gray)
                for key in archive.files:
                    if key!='TIR_uint8':np.testing.assert_array_equal(archive[key],maps[key])
            for mode,model in controls.items():
                actual,reserved=model.make_condition(tir)
                np.testing.assert_array_equal(actual[0,0].numpy(),maps[mode])
                if torch.count_nonzero(reserved).item():raise RuntimeError("Reserved confidence channel must stay zero")
            condition=controls['c2c'](tir,edge_override=torch.from_numpy(maps['joint512'])[None,None])
            np.testing.assert_array_equal(condition[:,0:1].numpy(),tir.numpy())
            np.testing.assert_array_equal(condition[0,1].numpy(),maps['joint512'])
            if torch.count_nonzero(condition[:,2:3]).item():raise RuntimeError("Third direct channel must stay zero")
            if np.any(maps['joint512']+2e-7<maps['semantic512']):raise RuntimeError("Structural protection weakened the semantic-v1 contour")
            if np.any(maps['joint512']+1e-7<SETTINGS['fine_floor']*maps['fine']):raise RuntimeError("Fine floor violated")
            matched,gain=match_global_mean(maps['semantic512'],maps['joint512'])
            maps['semantic_global_match']=matched
            prefix=f'{number:02d}'
            Image.fromarray(gray).save(output/'frames'/f'{prefix}_TIR.png')
            for key,values in maps.items():
                uint8_image(values).save(output/'frames'/f'{prefix}_{key}.png')
                if key in MODES:red_overlay(gray,values).save(output/'frames'/f'{prefix}_{key}_overlay.png')
            render(gray,maps).save(output/'frames'/f'{prefix}_comparison.png')
            np.savez_compressed(output/'maps'/f'{prefix}_contours.npz',TIR_uint8=gray,**maps)
            frames.append(dict(prefix=prefix,filename=name,group=row['group'],video_id=row['video_id']))
            vegetation=probabilities[CLASS_NAMES.index('vegetation')]
            stats=dict(filename=name,group=row['group'],global_match_gain=gain,
                       global_match_mean_error=float(matched.mean(dtype=np.float64)-maps['joint512'].mean(dtype=np.float64)),
                       protection_gate_mean=float(maps['structure_protection_gate'].mean()))
            for mode in (*MODES,'semantic_global_match'):
                stats[mode+'_mean']=float(maps[mode].mean())
                stats[mode+'_vegetation_weighted_strength_proxy']=weighted_mean(maps[mode],vegetation)
                stats[mode+'_important_weighted_strength_proxy']=weighted_mean(maps[mode],maps['semantic_support'])
            frame_rows.append(stats)
            for item in source_targets:
                if item['filename']!=name:continue
                panel=f"{item['id']}_contours.png";x0,y0,x1,y1=item['box']
                render(gray,maps,item['crop'],item['box'],args.zoom).save(output/'targets'/panel)
                targets.append(dict(item,panel=panel,box_size_pixels=[x1-x0,y1-y0]))
                region=np.s_[int(np.floor(y0)):int(np.ceil(y1)),int(np.floor(x0)):int(np.ceil(x1))]
                ts=dict(id=item['id'],filename=name,category=item['category'],group=item['group'],
                        mean_structure_R_proxy=float(maps['base_importance'][region].mean()),
                        mean_semantic_R_proxy=float(maps['semantic_importance'][region].mean()),
                        mean_joint_R_proxy=float(maps['joint_importance'][region].mean()))
                for mode in (*MODES,'semantic_global_match'):ts[mode+'_box_mean_strength_proxy']=float(maps[mode][region].mean())
                target_rows.append(ts)
            write_json(output/'status.json',dict(phase='running',completed=number,total=len(selected)))
            print(f'[{number}/{len(selected)}] {name}',flush=True)
        write_csv(output/'frames.csv',frame_rows);write_csv(output/'targets.csv',target_rows)
        write_page(output/'index.html',frames,targets,pack_id)
        quick=[]
        for frame in frames:
            if frame['prefix'] in ('01','02','31','34'):
                with Image.open(output/'frames'/f"{frame['prefix']}_comparison.png") as image:quick.append(np.asarray(image).copy())
        if quick:Image.fromarray(np.concatenate(quick,axis=0)).save(output/'quicklook.png')
        manifest=dict(format='structure_semantic_contour_preview_v1',pack_id=pack_id,identity=identity,
                      frames=frames,targets=targets,elapsed_seconds=time.perf_counter()-started,
                      environment=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,device='cpu'),
                      controls='All previous v1 floating arrays and C2b/C2c checked exactly; [TIR,joint edge,0] verified for every frame',
                      limits=['Importance protection is fixed and not calibrated edge confidence.','Strong semantic mistakes and texture rescue can remain.',
                              'Strength proxies are not edge accuracy or RGB improvement.','Only existing candidates can be strengthened.',
                              'Boxes used solely for review/statistics; no labels train the protection rule.','No GAN training mode registered; online crop/flip/cycle integration remains separate.'])
        write_json(output/'preview_manifest.json',manifest)
        write_json(output/'status.json',dict(phase='completed',completed=len(frames),total=len(selected),targets=len(targets),elapsed_seconds=manifest['elapsed_seconds']))
        print(f"Wrote {output/'index.html'}",flush=True)
    except BaseException as error:
        write_json(output/'status.json',dict(phase='failed_or_interrupted',completed=len(frames),total=len(selected),error=type(error).__name__,message=str(error)))
        raise


if __name__=='__main__':run(parse_args())
