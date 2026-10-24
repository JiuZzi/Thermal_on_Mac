#!/usr/bin/env python3
"""Audit frozen person responses and R attenuation; never use boxes to make edges.

This is a development review and an isolated R-only probe, not a training mode.
Reuses the fixed 40 images, and checks the existing contour output pixelwise.
"""

import argparse
from collections import defaultdict
import csv
import hashlib
import html
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pytorch-CycleGAN-and-pix2pix"))
from models.recall_semantic_edges import RECALL_SETTINGS, recall_components
from models.saliency_edges import region_recall_importance, structure_texture_importance
from models.selective_semantic_edges import importance_gain
from models.semantic_edges import FMB_CLASS_NAMES, fuse_candidates, semantic_importance
from review_contour_candidates import transformed_box


AUDIT = ROOT / "runs/edge_audit"
REFERENCE = AUDIT / "contour_clarity_seed42_v1"
SEMANTIC = AUDIT / "segformer_fmb_b3_semantic_preview_seed42"
JOINT = AUDIT / "semantic512_structure_joint_seed42_v2"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def mean(value, mask):
    return float(value[mask].mean(dtype=np.float64)) if mask.any() else None


def box_mask(shape, box):
    yy, xx = np.indices(shape)
    x0, y0, x1, y1 = box
    return (xx + .5 >= x0) & (xx + .5 < x1) & (yy + .5 >= y0) & (yy + .5 < y1)


def sheet(gray, panels, crop=None, box=None):
    # Every displayed response uses the same fixed [0,1] scale.
    crop = crop or (0, 0, gray.shape[1], gray.shape[0])
    x0, y0, x1, y1 = crop
    zoom = 5 if box is not None else 1
    w, h = (x1-x0)*zoom, (y1-y0)*zoom
    canvas = Image.new("RGB", (len(panels)*w, h+30), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (name, value) in enumerate(panels):
        a = np.rint(np.clip(value[y0:y1, x0:x1], 0, 1)*255).astype(np.uint8)
        tile = Image.fromarray(a).convert("RGB").resize((w, h), Image.Resampling.NEAREST)
        if box is not None:
            bx0, by0, bx1, by1 = box
            ImageDraw.Draw(tile).rectangle(((bx0-x0)*zoom, (by0-y0)*zoom,
                                            (bx1-x0)*zoom, (by1-y0)*zoom), outline="#ffd000", width=1)
        canvas.paste(tile, (index*w, 30)); draw.text((index*w+3, 6), name, fill="black")
    return canvas


def run(output, focus):
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose an empty output directory; preserve previous reviews")
    output.mkdir(parents=True, exist_ok=True)
    for name in ("frames", "targets", "maps"): (output/name).mkdir(exist_ok=True)
    manifest = json.loads((REFERENCE/"preview_manifest.json").read_text())
    source = json.loads((SEMANTIC/"preview_manifest.json").read_text())
    assert len(manifest["frames"]) == 40
    assert manifest["identity"]["semantic_pack"] == source["pack_id"]
    coco_path = ROOT/"FLIR_ADAS_v2/images_thermal_train/coco.json"
    coco = json.loads(coco_path.read_text())
    images = {Path(row["file_name"]).name: row for row in coco["images"]}
    annotations = defaultdict(list)
    for row in coco["annotations"]:
        if row["category_id"] in (1, 74): annotations[row["image_id"]].append(row)
    frames, people, frame_rows, pages = [], [], [], []
    for frame in manifest["frames"]:
        prefix, filename = frame["prefix"], frame["filename"]
        paths = dict(semantic=SEMANTIC/"maps"/f"{prefix}_responses.npz",
                     joint=JOINT/"maps"/f"{prefix}_contours.npz",
                     clarity=REFERENCE/"maps"/f"{prefix}_contours.npz")
        assert sha(paths["semantic"]) == manifest["identity"]["semantic_cache_sha256"][filename], "Semantic cache changed since the recorded real-weight replay"
        with np.load(paths["semantic"], allow_pickle=False) as archive:
            gray = archive["TIR_uint8"]
            assert tuple(archive["class_names"].tolist()) == FMB_CLASS_NAMES
            p, detail = archive["base_probabilities"], archive["detail_probabilities"]
        with Image.open(ROOT/"FLIR_datasets/trainB"/filename) as original:
            np.testing.assert_array_equal(gray, np.asarray(original))
        # Match the online normalized TIR arithmetic rather than re-quantizing.
        image = ((gray.astype(np.float32)/255-.5)/.5+1)/2
        current, _ = recall_components(image, p)
        with np.load(paths["clarity"], allow_pickle=False) as archive:
            for key in ("clarity_v1", "joint_importance", "raw_fine", "raw_coarse"):
                np.testing.assert_array_equal(current[key], archive[key])
        structure, clutter = structure_texture_importance(image, current["raw_fine"], current["raw_coarse"])
        semantic, support = semantic_importance(p)
        with np.load(paths["joint"], allow_pickle=False) as archive:
            for key, value in (("base_importance", structure), ("clutter", clutter),
                               ("semantic_importance", semantic), ("semantic_support", support)):
                np.testing.assert_array_equal(value, archive[key])
        fine, coarse, base = (current[k] for k in ("clarity_v1_fine", "clarity_v1_coarse", "joint_importance"))
        local, local_maps = region_recall_importance(image, fine, coarse, base)
        # A diagnostic, fixed-coefficient R-only probe. Actual candidate advantage
        # prevents a larger R from reducing the current coarse route.
        limited = base + (1-clutter)*(local-base)
        def edge(weight):
            return np.maximum(current["preserved_ridge_floor"],
                importance_gain(weight)*fuse_candidates(fine, coarse, clutter, weight)).astype(np.float32)
        probe, unlimited = edge(limited), edge(local)
        np.testing.assert_array_equal(probe, current["recall_v1"])
        np.testing.assert_array_equal(limited, current["recall_importance"])
        old = current["clarity_v1"]
        assert np.all(probe >= old-1e-6) and np.all(unlimited >= old-1e-6)
        matched = old*(probe.mean(dtype=np.float64)/old.mean(dtype=np.float64) if old.any() else 1.)
        assert np.max(matched) <= 1, "Global-brightness control must not clip"
        assert abs(matched.mean(dtype=np.float64)-probe.mean(dtype=np.float64)) < 1e-7
        maps = dict(TIR_uint8=gray, person=p[8], person_1024_probe=detail[8],
                    structure_importance=structure, semantic_importance=semantic,
                    joint_importance=base, clutter=clutter, importance_gain=current["importance_gain"],
                    fine=fine, coarse=coarse, old=old, limited_probe=probe, unlimited_probe=unlimited,
                    matched_brightness=matched, limited_importance=limited, **local_maps)
        labels = p.argmax(0)
        for region, mask in (("all", np.ones(image.shape, bool)),
                             ("predicted_vegetation_not_ground_truth", labels == 6),
                             ("predicted_road_not_ground_truth", labels == 1)):
            frame_rows.append(dict(prefix=prefix, region=region, pixels=int(mask.sum()),
                old=mean(old, mask), limited_probe=mean(probe, mask),
                unlimited_probe=mean(unlimited, mask), matched_brightness=mean(matched, mask)))
        im = images[filename]; frame_targets = []
        for ann in annotations[im["id"]]:
            box, clipped = transformed_box(ann, im["width"], im["height"], "full")
            mask = box_mask(gray.shape, box)
            if not mask.any(): continue
            edges = mask & current["clarity_v1_fine_ridges"].astype(bool)
            row = dict(id=f"{prefix}_ann{ann['id']}", prefix=prefix, annotation_id=ann["id"],
                category="person" if ann["category_id"] == 1 else "rider", clipped=clipped,
                width=box[2]-box[0], height=box[3]-box[1], box_pixels=int(mask.sum()),
                fine_ridge_pixels=int(edges.sum()), person_mean=mean(p[8], mask),
                person_argmax_fraction=mean((labels == 8).astype(np.float32), mask),
                detail_person_mean=mean(detail[8], mask),
                detail_person_argmax_fraction=mean((detail.argmax(0) == 8).astype(np.float32), mask))
            for key in ("structure_importance", "semantic_importance", "joint_importance", "clutter",
                        "importance_gain", "fine", "old", "limited_probe", "matched_brightness"):
                row[key+"_on_fine_ridges"] = mean(maps[key], edges)
            people.append(row)
            if prefix in focus:
                pad = max(6, round(max(row["width"], row["height"])*.15))
                crop = (max(0, int(np.floor(box[0]))-pad), max(0, int(np.floor(box[1]))-pad),
                        min(gray.shape[1], int(np.ceil(box[2]))+pad),
                        min(gray.shape[0], int(np.ceil(box[3]))+pad))
                panels = [("TIR", image), ("P(person) 512", p[8]), ("R structure", structure),
                          ("R joint", base), ("Old contour", old), ("R-only probe", probe)]
                sheet(gray, panels, crop, box).save(output/"targets"/f"{row['id']}.png")
                frame_targets.append(row)
        if prefix in focus:
            np.savez_compressed(output/"maps"/f"{prefix}_audit.npz", **maps)
            sheet(gray, [("TIR", image), ("P(person) 512", p[8]), ("R structure", structure),
                         ("R joint", base), ("Current contour", old)]).save(output/"frames"/f"{prefix}_audit.png")
            sheet(gray, [("TIR", image), ("Current contour", old), ("Global mean control", matched),
                         ("Local R-only probe", probe), ("Probe R", limited)]).save(output/"frames"/f"{prefix}_probe.png")
            crops = ''.join(f'<h4>{row["id"]} / {row["width"]:.1f}×{row["height"]:.1f}</h4>'
                f'<img class="crop" loading="lazy" src="targets/{row["id"]}.png">' for row in frame_targets)
            pages.append(f'<article id="f{prefix}"><h2>{prefix} / {html.escape(filename)}</h2>'
                f'<p>原始红外、512行人响应、结构R、联合R、当前轮廓。</p><img src="frames/{prefix}_audit.png">'
                f'<p>原始红外、当前轮廓、相同全图均值的整体提亮、局部R补充探针、探针R。</p>'
                f'<img src="frames/{prefix}_probe.png"><details><summary>该图全部已有行人/骑行者框</summary>{crops}</details></article>')
        frames.append(dict(**frame, cache_sha256={k:sha(v) for k,v in paths.items()}))
        print(f"[{len(frames)}/40] {prefix}: old={old.mean():.5f}, probe={probe.mean():.5f}", flush=True)
    write_csv(output/"person_statistics.csv", people)
    write_csv(output/"frame_statistics.csv", frame_rows)
    summary = dict(frames=len(frames), annotated_people=len(people),
        annotated_people_without_person_argmax=sum(row["person_argmax_fraction"] == 0 for row in people),
        focus=[row for row in people if row["prefix"] in focus],
        reference_recomputed_pixelwise_equal=True,
        limits="Boxes include background, not segmentation or edge ground truth. Counts are descriptive, not person recall or boundary accuracy. All 40 are repeatedly reviewed development images.")
    (output/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n")
    provenance = dict(format="person_importance_audit_v1", focus=focus, frames=frames,
        reference_manifest_sha256=sha(REFERENCE/"preview_manifest.json"),
        semantic_pack=source["pack_id"], previous_semantic_replay=manifest["identity"]["cache_revalidation"],
        coco_sha256=sha(coco_path), code_sha256={str(path.relative_to(ROOT)):sha(path) for path in (
            Path(__file__), ROOT/"pytorch-CycleGAN-and-pix2pix/models/saliency_edges.py",
            ROOT/"pytorch-CycleGAN-and-pix2pix/models/clarity_semantic_edges.py",
            ROOT/"pytorch-CycleGAN-and-pix2pix/models/recall_semantic_edges.py")},
        probe="Existing region_recall_importance(I,current fine,current coarse,Rjoint), then Rprobe=Rjoint+(1-D)*(Rlocal-Rjoint); identical candidates, soft width, D and ridge floor. No boxes/classes/coordinates enter the probe.",
        training_variant="recall_v1", launches_training=False, settings=RECALL_SETTINGS,
        semantic_correction=False, completion=False, confidence=False)
    (output/"manifest.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2)+"\n")
    intro = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>行人重要性漏关注定位</title>
<style>body{font:16px/1.6 system-ui;background:#f3f5f8;margin:24px;color:#18212b}main{max-width:1800px;margin:auto}article,.note{background:white;padding:16px;margin:20px 0;border:1px solid #ccc;border-radius:8px}img{max-width:100%}.crop{image-rendering:pixelated}h2{overflow-wrap:anywhere}</style><main>
<h1>行人重要性漏关注：11 / 18 / 31</h1><div class="note">
<p>原清晰度轮廓、语义与R均从当前源码逐像素复现；语义使用先前已验证的冻结缓存。展示按统一[0,1]强度，不逐图拉伸。</p>
<p>本页区分：语义没认出人、已存在候选被R/g/D压弱、原候选不足。R是细/粗混合及贡献权重，不能当作完整人体分割图。</p>
<p>局部R补充为可选recall_v1：复用既有多尺度相对梯度/区域对比度重要性算法，受(1-D)限制。细/粗候选、线宽、保底和语义模型均不变，没有补边、置信度或新检测器。已接入模型可选接口，本页只预览条件，不启动GAN训练。它可能同时抬亮树叶和路面纹理。</p>
<p>所有40张固定开发图均计算统计。以下三张展示全部已有行人框，黄框只定位，不参与语义、提边或重要性计算。框内含背景，不能用这些统计宣称分割/检测召回或边界精度提升。</p>
<p><a href="person_statistics.csv">40图全部行人框统计</a> · <a href="frame_statistics.csv">全图及预测植被/道路区域统计</a> · <a href="summary.json">定位摘要</a> · <a href="manifest.json">来源与公式</a></p></div>'''
    (output/"index.html").write_text(intro+''.join(pages)+'</main></html>', encoding="utf-8")
    print(json.dumps({k:v for k,v in summary.items() if k not in ("focus", "limits")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=AUDIT/"importance_recall_seed42_v1")
    parser.add_argument("--focus", nargs="+", default=["11", "18", "31"])
    args = parser.parse_args()
    run(args.output_dir, args.focus)
