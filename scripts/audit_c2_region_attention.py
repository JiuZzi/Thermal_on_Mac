"""Fixed CPU-only, no-external-model region-attention feasibility audit."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation
from skimage.segmentation import mark_boundaries

from audit_c2_conditions import ROOT, load_tir
from audit_c2_importance import fuse_fixed_candidates, priority_objects, write_csv
from audit_c2_region_recall import compare_importance, render_panel
from region_attention_probe import PARAMETERS, match_mean, region_attention
from review_contour_candidates import CATEGORIES, balanced_sample, records_from_annotations, transformed_box


CASE02 = "video-DJYDQTZH7GLCqw3Tx-frame-008810-zLboSzPyqEzu5GzBJ.jpg"
WINDOWS = dict(road_mixed=[35,180,325,280], distant_mixed=[40,65,200,110],
               building_mixed=[265,35,359,105])


def object_boxes(record):
    boxes = []
    for ann in record["annotations"]:
        box, clipped = transformed_box(ann, record["source_width"], record["source_height"], "full")
        if box[2]-box[0] < 2 or box[3]-box[1] < 2:
            continue
        x0,y0,x1,y1 = box
        boxes.append(dict(annotation_id=ann["id"], category=CATEGORIES[ann["category_id"]],
                          box=[math.floor(x0),math.floor(y0),math.ceil(x1),math.ceil(y1)], clipped=clipped))
    return boxes


def mask_at(shape, box):
    mask = np.zeros(shape, bool)
    x0,y0,x1,y1 = box
    mask[y0:y1,x0:x1] = True
    return mask


def attention_stats(arrays, mask):
    selected = mask & (arrays["fine"] >= .2)
    keys = ("old_R", "recall_R", "region_R", "old_E", "recall_E", "region_E", "mean_control")
    return dict(candidate_pixels=int(selected.sum()),
                **{key: float(arrays[key][selected].mean()) if selected.any() else None for key in keys})


def columns(arrays):
    return {"TIR": arrays["TIR"], "old C2c": arrays["old_E"],
            "region_recall": arrays["recall_E"], "matched old C2c": arrays["mean_control"],
            "regional probe": arrays["region_E"]}


def draw_cues(arrays, path):
    gray = arrays["TIR"]
    seg = mark_boundaries(gray, arrays["labels"], color=(0,1,1))
    panels = {"TIR / grayscale segments": seg, "old R [0,1]": arrays["old_R"],
              "region score [0,1]": arrays["region_score"], "probe R [0,1]": arrays["region_R"]}
    h,w = gray.shape
    sheet = Image.new("RGB", (w*4,h+58), "white")
    draw = ImageDraw.Draw(sheet)
    for col,(label,values) in enumerate(panels.items()):
        if values.ndim == 2:
            values = np.repeat(values[...,None],3,-1)
        draw.text((col*w+4,4),label,fill="black")
        sheet.paste(Image.fromarray(np.uint8(np.clip(values,0,1)*255)),(col*w,23))
    draw.text((4,h+30),"Connected gray regions are NOT semantic objects. Scores are NOT confidence or edge truth.",fill="black")
    sheet.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-audit",type=Path,default=ROOT/"runs/edge_audit/c2c_region_recall_20261012_seed42")
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--allow-same-video-extras",action="store_true",
                        help="If no unseen videos exist, review 8 new frames, explicitly NOT independent-video validation")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Choose a new directory; previous evidence is never overwritten")
    args.processed_dir = ROOT/"FLIR_datasets/trainB"
    args.manifest = ROOT/"FLIR_protocol_v2/manifest.csv"
    args.coco = ROOT/"FLIR_ADAS_v2/images_thermal_train/coco.json"
    reference = json.loads((args.reference_audit/"audit_manifest.json").read_text(encoding="utf-8"))
    if reference["selection"]["seed"] != args.seed or reference["summary"]["geometry"] != "full":
        raise ValueError("Reference must use the same seed and full geometry")
    for name,digest in reference["code_hashes"].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Reference source changed: {name}")
    records = records_from_annotations(args)
    lookup = {record["filename"]:record for record in records}
    selection = reference["selection"]["samples"]
    chosen = [dict(lookup[item["filename"]],group=item["group"]) for item in selection]
    old_videos = {record["video_id"] for record in chosen}
    pool = [record for record in records if record["video_id"] not in old_videos]
    # Entire videos are new to this fixed 32-image audit, if available.
    extras = balanced_sample(pool, min(8,len({r["video_id"] for r in pool})), random.Random(args.seed))
    extra_group = "new_video_review"
    if not extras and args.allow_same_video_extras:
        used = {r["filename"] for r in chosen}
        pool = [r for r in records if r["filename"] not in used]
        extras = balanced_sample(pool,min(8,len(pool)),random.Random(args.seed))
        extra_group = "new_frame_same_video"
    chosen += [dict(record,group=extra_group) for record in extras]
    output = args.output_dir
    for name in ("frames","targets","maps"):
        (output/name).mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    frame_rows, target_rows, cards, source_hashes, map_hashes, case02 = [], [], [], {}, {}, {}
    for number,record in enumerate(chosen,1):
        source = args.processed_dir/record["filename"]
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        source_hashes[record["filename"]] = digest
        _,tir = load_tir(source,"full")
        gray = np.clip((tir[0,0].numpy()+1)/2,0,1)
        if gray.shape != (288,360):
            raise ValueError("Expected full 360x288 processed TIR")
        previous,_ = compare_importance(gray)
        if number <= len(selection):
            if reference["source_hashes"][record["filename"]] != digest:
                raise ValueError("Reference input image changed")
            with np.load(args.reference_audit/"maps"/f"{number:02d}_maps.npz") as archived:
                for key in ("fine","coarse","clutter","old_R","new_R","baseline","enhanced"):
                    np.testing.assert_array_equal(previous[key],archived[key])
                np.testing.assert_array_equal(gray,archived["TIR"])
        start = time.perf_counter()
        regional_R,cues = region_attention(gray,previous["old_R"])
        elapsed = time.perf_counter()-start
        regional_E,before_floor = fuse_fixed_candidates(previous["fine"],previous["coarse"],regional_R,previous["clutter"])
        control,gain = match_mean(previous["baseline"],regional_E)
        np.testing.assert_allclose(control.mean(dtype=np.float64),regional_E.mean(dtype=np.float64),rtol=0,atol=1e-7)
        if np.any(regional_E < .25*previous["fine"]-1e-7):
            raise AssertionError("Fine floor changed")
        absent = (previous["fine"] == 0)&(previous["coarse"] == 0)
        if np.any(regional_E[absent]):
            raise AssertionError("Probe invented candidate support")
        arrays = dict(TIR=gray,old_R=previous["old_R"],recall_R=previous["new_R"],region_R=regional_R,
                      old_E=previous["baseline"],recall_E=previous["enhanced"],region_E=regional_E,
                      mean_control=control,fine=previous["fine"],coarse=previous["coarse"],
                      clutter=previous["clutter"],before_floor=before_floor,**cues)
        prefix = f"{number:02d}"
        np.savez_compressed(output/"maps"/f"{prefix}_maps.npz",**arrays)
        map_hashes[f"maps/{prefix}_maps.npz"] = hashlib.sha256((output/"maps"/f"{prefix}_maps.npz").read_bytes()).hexdigest()
        render_panel(gray,columns(arrays),output/"frames"/f"{prefix}_comparison.png")
        draw_cues(arrays,output/"frames"/f"{prefix}_importance.png")
        frame_rows.append(dict(index=number,filename=record["filename"],video_id=record["video_id"],group=record["group"],
                               segments=int(cues["labels"].max()+1),region_probe_seconds=elapsed,gain_control=gain,
                               old_mean=float(arrays["old_E"].mean()),recall_mean=float(arrays["recall_E"].mean()),
                               region_mean=float(regional_E.mean()),
                               increase_003_fraction=float((regional_E-arrays["old_E"]>.03).mean()),
                               decrease_003_fraction=float((regional_E-arrays["old_E"]<-.03).mean()),
                               vs_mean_control_mae=float(np.abs(regional_E-control).mean())))
        boxes = object_boxes(record)
        all_boxes = np.zeros(gray.shape,bool)
        for obj in boxes:
            all_boxes |= mask_at(gray.shape,obj["box"])
        for obj in boxes:
            mask = mask_at(gray.shape,obj["box"])
            ring = binary_dilation(mask,iterations=8)&~all_boxes
            inside,outside = attention_stats(arrays,mask),attention_stats(arrays,ring)
            row = dict(frame=number,group=record["group"],filename=record["filename"],**obj)
            row["box"] = json.dumps(row["box"])
            for tag,stats in (("inside",inside),("surround",outside)):
                row.update({f"{tag}_{key}":value for key,value in stats.items()})
            for key in ("old_E","recall_E","region_E","mean_control","old_R","recall_R","region_R"):
                row[f"{key}_inside_minus_surround"] = (inside[key]-outside[key]
                    if inside[key] is not None and outside[key] is not None else None)
            target_rows.append(row)
        crop_objects = priority_objects(record,"full")
        persons = [obj for obj in boxes if obj["category"]=="person" and not obj["clipped"]]
        if persons:
            largest = max(persons,key=lambda o:(o["box"][2]-o["box"][0])*(o["box"][3]-o["box"][1]))
            if largest["annotation_id"] not in {o["annotation_id"] for o in crop_objects}:
                crop_objects.append(largest)
        crop_html = []
        for obj in crop_objects:
            x0,y0,x1,y1 = obj["box"]
            margin = max(6,round(max(x1-x0,y1-y0)*.25))
            x0,y0,x1,y1 = max(0,math.floor(x0-margin)),max(0,math.floor(y0-margin)),min(360,math.ceil(x1+margin)),min(288,math.ceil(y1+margin))
            crop = {key:value[y0:y1,x0:x1] for key,value in arrays.items()}
            name = f"targets/{prefix}_A{obj['annotation_id']}.png"
            render_panel(crop["TIR"],columns(crop),output/name,zoom=min(6,max(1,math.ceil(140/(y1-y0)))))
            crop_html.append(f'<p>{obj["category"]} / annotation {obj["annotation_id"]}</p><img loading="lazy" src="{name}">')
        if record["filename"] == CASE02:
            for label,category in (("person_boxes","person"),("car_boxes","car")):
                mask = np.zeros(gray.shape,bool)
                for obj in boxes:
                    if obj["category"] == category:
                        mask |= mask_at(gray.shape,obj["box"])
                case02[label] = attention_stats(arrays,mask)
            for label,box in WINDOWS.items():
                case02[label] = attention_stats(arrays,mask_at(gray.shape,box))
        cards.append(f'<article id="frame-{prefix}"><h2>{prefix} | {record["group"]} | {html.escape(record["filename"])}</h2>'
                     f'<img loading="lazy" src="frames/{prefix}_importance.png"><img loading="lazy" src="frames/{prefix}_comparison.png">'
                     f'<details><summary>固定目标裁剪（含最大完整行人）</summary>{"".join(crop_html)}</details></article>')
        print(f"Reviewed {number}/{len(chosen)}; regions={frame_rows[-1]['segments']}; CPU={elapsed:.2f}s",flush=True)
    groups = {}
    for group in sorted({row["group"] for row in frame_rows}):
        rows = [r for r in frame_rows if r["group"]==group]
        groups[group] = dict(frame_count=len(rows),**{key:float(np.mean([r[key] for r in rows])) for key in (
            "old_mean","recall_mean","region_mean","increase_003_fraction","decrease_003_fraction","vs_mean_control_mae","region_probe_seconds")})
    summary = dict(seed=args.seed,frame_count=len(chosen),old_frame_count=len(selection),
                   extra_frame_count=len(extras),extra_group=extra_group,
                   new_video_count=len({r["video_id"] for r in extras}-old_videos),
                   baseline_bitwise_verified=True,parameters=PARAMETERS,groups=groups,case02=case02,
                   annotation_count=len(target_rows),training_model_changed=False,external_models_used=False,
                   scope="Image-only fixed heuristic; development attention audit, not semantic/edge accuracy or RGB quality",
                   extra_video_ids=sorted({r["video_id"] for r in extras}),
                   limitations=["No semantic class supervision: gray regions need not correspond to objects.",
                                "Boxes/rings/windows are mixed regions, not pixel truth.",
                                "32 reused development frames; extras may share videos and are not a formal test set.",
                                "Low R does not mean unreliable edge; no confidence output is implemented."])
    code_files = [Path(__file__),ROOT/"scripts/region_attention_probe.py",ROOT/"scripts/audit_c2_importance.py",
                  ROOT/"scripts/audit_c2_region_recall.py",ROOT/"scripts/review_contour_candidates.py",
                  ROOT/"scripts/audit_c2_conditions.py",ROOT/"pytorch-CycleGAN-and-pix2pix/models/saliency_edges.py"]
    manifest = dict(summary=summary,source_hashes=source_hashes,map_hashes=map_hashes,
                    code_hashes={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in code_files},
                    selection=[{key:record[key] for key in ("filename","video_id","group")} for record in chosen],
                    annotation_use="evaluation/selection only, never passed into region_attention or fusion")
    for name,value in (("summary.json",summary),("audit_manifest.json",manifest)):
        (output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")
    write_csv(output/"samples.csv",frame_rows)
    write_csv(output/"targets.csv",target_rows)
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>无外部模型：区域重要性审查</title><style>body{font:16px/1.6 system-ui;margin:24px;background:#eee}article{background:white;padding:16px;margin:24px 0}img{max-width:100%;height:auto}h2{font-size:18px;overflow-wrap:anywhere}summary{cursor:pointer}</style>'
    page += f'<h1>区域重要性：固定规则可行性验证，seed42</h1><p>无外部模型、无训练、无 JR、无置信度。前32张复用开发样本，新增 {len(extras)} 张的组别为 {extra_group}；new_frame_same_video 表示同视频新帧，不是跨视频验证。区域不是语义目标；更亮/更暗不等于更正确。</p>'
    page += '<p>上排：灰度连通区域、原 R、区域分数、融合后 R。下排：红外、原 C2c、region_recall、与探针整图均值匹配的原 C2c、区域探针。所有 R/E 固定 [0,1] 显示；红色仅代表轮廓响应。</p>'
    page += '<p>固定 F/C/D 与底线，只改 R；框仅用于评价，框外也可能有有用结构。此次不是人车语义识别，不能从框内均值推断边界准确率。</p>'
    (output/"index.html").write_text(page+"\n".join(cards),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__ == "__main__":
    main()
