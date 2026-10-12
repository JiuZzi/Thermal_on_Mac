"""CPU-only development audit: spatial C2c importance versus its image mean.

Only R changes. Fine/coarse candidates, clutter, background gain and the
fine-edge floor are shared. Boxes locate review crops, never boundary labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn

from audit_c2_conditions import ROOT, load_tir
from review_contour_candidates import (
    CATEGORIES, NAMES, records_from_annotations, select_records, transformed_box,
)
from models.conditioned_generator import ConditionedGenerator
from models.saliency_edges import make_adaptive_saliency_edge, structure_texture_importance


def fuse_fixed_candidates(fine, coarse, importance, clutter):
    """Audit replica of the CURRENT fusion; never called by training."""
    arrays = tuple(np.asarray(value, dtype=np.float32)
                   for value in (fine, coarse, importance, clutter))
    if any(value.shape != arrays[0].shape for value in arrays):
        raise ValueError("All fixed-candidate maps must have the same shape")
    if any(not np.isfinite(value).all() or np.any((value < 0) | (value > 1))
           for value in arrays):
        raise ValueError("Fixed-candidate maps must be finite and in [0,1]")
    fine, coarse, importance, clutter = arrays
    before_floor = (importance * fine + (1 - importance) * .5 * coarse) * (1 - .4 * clutter)
    return np.clip(np.maximum(before_floor, .25 * fine), 0, 1), before_floor


def importance_ablation(gray):
    spatial, importance, fine, coarse = make_adaptive_saliency_edge(
        gray, (.12, .16, .20), .5, 1., (.7, 1., 1.6), .5,
    )
    _, clutter = structure_texture_importance(gray, fine, coarse)
    replica, before_floor = fuse_fixed_candidates(fine, coarse, importance, clutter)
    np.testing.assert_array_equal(replica, spatial)
    mean_importance = np.float32(importance.mean(dtype=np.float64))
    constant_importance = np.full_like(importance, mean_importance)
    constant, constant_before_floor = fuse_fixed_candidates(fine, coarse, constant_importance, clutter)
    return dict(fine=fine, coarse=coarse, R=importance, clutter=clutter,
                constant_R=constant_importance, spatial=spatial, constant=constant,
                before_floor=before_floor, constant_before_floor=constant_before_floor)


def map_stats(maps):
    fine, spatial, constant = (maps[key] for key in ("fine", "spatial", "constant"))
    delta = spatial - constant
    eligible = fine >= .2  # Reporting subset only, not a training label.
    stats = dict(importance_mean=float(maps["R"].mean()),
                 importance_std=float(maps["R"].std()),
                 fine_mean=float(fine.mean()), spatial_mean=float(spatial.mean()),
                 constant_mean=float(constant.mean()), mean_abs_difference=float(np.abs(delta).mean()),
                 brighter_fraction_delta_005=float((delta > .05).mean()),
                 dimmer_fraction_delta_005=float((delta < -.05).mean()),
                 spatial_ge05_fraction=float((spatial >= .5).mean()),
                 constant_ge05_fraction=float((constant >= .5).mean()),
                 candidate_pixel_count=int(eligible.sum()))
    for tag, raw in (("spatial", maps["before_floor"]), ("constant", maps["constant_before_floor"])):
        stats[f"{tag}_floor_active_on_candidates"] = (
            float((raw[eligible] <= .25 * fine[eligible]).mean()) if eligible.any() else None)
    return stats


def render_comparison(gray, maps, path, zoom=1):
    """Fixed [0,1] display, no per-map normalization; overlays are diagnostics."""
    height, width = gray.shape
    cell = max(160, width * zoom)
    sheet = Image.new("RGB", (cell * 4, height * zoom * 2 + 58), "white")
    draw = ImageDraw.Draw(sheet)
    labels = ("TIR", "fine candidate", "constant R", "spatial R (current)")
    base = np.repeat(gray[..., None], 3, -1)
    for column, (label, array) in enumerate(zip(labels, (gray, maps["fine"], maps["constant"], maps["spatial"]))):
        draw.text((column * cell + 4, 4), label, fill="black")
        top = np.repeat(array[..., None], 3, -1)
        alpha = array[..., None] * .65
        bottom = base if column == 0 else base * (1 - alpha) + np.array([1, 0, 0]) * alpha
        for values, y in ((top, 23), (bottom, 23 + height * zoom)):
            image = Image.fromarray(np.uint8(np.clip(values, 0, 1) * 255)).resize(
                (width * zoom, height * zoom), Image.Resampling.NEAREST)
            sheet.paste(image, (column * cell + (cell - image.width) // 2, y))
    draw.text((4, height * zoom * 2 + 29), "Same scale [0,1]. Red = response, NOT correctness.", fill="black")
    sheet.save(path)


def render_diagnostics(gray, maps, path):
    height, width = gray.shape
    sheet = Image.new("RGB", (width * 4, height + 55), "white")
    draw = ImageDraw.Draw(sheet)
    delta = maps["spatial"] - maps["constant"]
    colored = np.zeros((*gray.shape, 3), np.float32)
    colored[..., 0] = np.maximum(delta, 0) * 4
    colored[..., 2] = np.maximum(-delta, 0) * 4
    labels = ("TIR", "importance R [0,1]", "clutter D [0,1]", "delta: red + / blue -; x4")
    arrays = (gray, maps["R"], maps["clutter"], colored)
    for column, (label, values) in enumerate(zip(labels, arrays)):
        if values.ndim == 2:
            values = np.repeat(values[..., None], 3, -1)
        draw.text((column * width + 4, 4), label, fill="black")
        sheet.paste(Image.fromarray(np.uint8(np.clip(values, 0, 1) * 255)), (column * width, 23))
    draw.text((4, height + 30), "R is an importance heuristic, NOT object probability or confidence.", fill="black")
    sheet.save(path)


def priority_objects(record, geometry):
    objects = []
    for annotation in sorted(record["annotations"], key=lambda value: value["id"]):
        box, clipped = transformed_box(annotation, record["source_width"], record["source_height"], geometry)
        if box[2] - box[0] >= 2 and box[3] - box[1] >= 2:
            objects.append(dict(annotation_id=annotation["id"], category=CATEGORIES[annotation["category_id"]],
                                box=list(box), clipped=clipped))
    area = lambda item: (item["box"][2] - item["box"][0]) * (item["box"][3] - item["box"][1])
    chosen = []
    for pool in ([item for item in objects if item["category"] in ("person", "rider")],
                 [item for item in objects if item["category"] not in ("person", "rider")]):
        if pool:
            chosen.append(min(pool, key=lambda item: (area(item), item["annotation_id"])))
    remaining = [item for item in objects if item not in chosen]
    if remaining:
        chosen.append(max(remaining, key=lambda item: (area(item), item["annotation_id"])))
    return chosen


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "FLIR_datasets/trainB")
    parser.add_argument("--manifest", type=Path, default=ROOT / "FLIR_protocol_v2/manifest.csv")
    parser.add_argument("--coco", type=Path, default=ROOT / "FLIR_ADAS_v2/images_thermal_train/coco.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/edge_audit/c2c_importance_ablation_seed42")
    parser.add_argument("--sample-count", type=int, default=24)
    parser.add_argument("--challenge-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--geometry", choices=("full", "center_crop_256"), default="full")
    parser.add_argument("--selection-json", type=Path)
    args = parser.parse_args()
    if args.processed_dir.resolve() != (ROOT / "FLIR_datasets/trainB").resolve():
        parser.error("This development audit uses trainB only, never testB")
    if args.sample_count < 1 or args.challenge_count < 0:
        parser.error("sample-count >= 1 and challenge-count >= 0 required")
    if args.output_dir.exists():
        parser.error("Use a new output directory; existing audits will not be overwritten")
    return args


def main():
    args = parse_args()
    torch.set_num_threads(1)
    selected, selection = select_records(records_from_annotations(args), args)
    output = args.output_dir
    for name in ("frames", "maps", "targets"):
        (output / name).mkdir(parents=True, exist_ok=False)
    official = ConditionedGenerator(nn.Identity(), "soft_saliency", fusion_mode="direct")
    frames, targets, cards = [], [], []
    for index, record in enumerate(selected, 1):
        source = args.processed_dir / record["filename"]
        with Image.open(source) as image:
            if image.mode != "L" or image.size != (360, 288):
                raise ValueError(f"Expected processed L-mode 360x288 TIR: {source}")
        original, tir = load_tir(source, args.geometry)
        gray = np.clip((tir[0, 0].numpy() + 1) / 2, 0, 1)
        maps = importance_ablation(gray)
        actual, reserved = official.make_condition(tir)
        np.testing.assert_allclose(actual[0, 0].numpy(), maps["spatial"], atol=1e-7, rtol=0)
        if torch.count_nonzero(reserved).item():
            raise RuntimeError("Reserved confidence channel changed")
        prefix = f"{index:02d}"
        comparison = f"frames/{prefix}_comparison.png"
        diagnostics = f"frames/{prefix}_diagnostics.png"
        render_comparison(gray, maps, output / comparison)
        render_diagnostics(gray, maps, output / diagnostics)
        np.savez_compressed(output / "maps" / f"{prefix}_maps.npz", TIR=gray, **maps)
        item = dict(index=index, filename=record["filename"], video_id=record["video_id"], group=record["group"],
                    sha256=hashlib.sha256(source.read_bytes()).hexdigest(), **map_stats(maps))
        frames.append(item)
        crops_html = []
        for number, obj in enumerate(priority_objects(record, args.geometry), 1):
            box = obj["box"]
            margin = max(6, round(max(box[2] - box[0], box[3] - box[1]) * .25))
            crop = (max(0, math.floor(box[0] - margin)), max(0, math.floor(box[1] - margin)),
                    min(gray.shape[1], math.ceil(box[2] + margin)), min(gray.shape[0], math.ceil(box[3] + margin)))
            x1, y1, x2, y2 = crop
            sliced = {key: value[y1:y2, x1:x2] for key, value in maps.items()}
            target_id = f"{prefix}_O{number:02d}"
            crop_file = f"targets/{target_id}_comparison.png"
            zoom = min(6, max(1, math.ceil(160 / (y2 - y1))))
            render_comparison(gray[y1:y2, x1:x2], sliced, output / crop_file, zoom=zoom)
            # Constant R remains the FULL image mean, never recomputed per crop.
            targets.append(dict(id=target_id, filename=record["filename"], group=record["group"],
                                category=obj["category"], annotation_id=obj["annotation_id"],
                                clipped=obj["clipped"], box=json.dumps(box), crop=json.dumps(crop),
                                **map_stats(sliced)))
            title = f"{target_id} {NAMES[obj['category']]}" + ("（裁剪截断，只评可见部分）" if obj["clipped"] else "")
            crops_html.append(f'<h4>{html.escape(title)}</h4><img loading="lazy" src="{crop_file}">')
        title = f"{prefix} | {record['group']} | {record['filename']}"
        cards.append(f'<article><h3>{html.escape(title)}</h3><img loading="lazy" src="{comparison}">'
                     f'<details><summary>查看权重 R、纹理抑制 D 和增强／减弱位置</summary>'
                     f'<img loading="lazy" src="{diagnostics}"></details>'
                     f'<details><summary>查看固定重点目标（{len(crops_html)} 个）</summary>{"".join(crops_html)}</details></article>')
        print(f"Audited {index}/{len(selected)}: {record['filename']}", flush=True)
    fields = ("importance_mean", "importance_std", "spatial_mean", "constant_mean", "mean_abs_difference",
              "brighter_fraction_delta_005", "dimmer_fraction_delta_005", "spatial_ge05_fraction", "constant_ge05_fraction")
    aggregate = lambda rows: {key: float(np.mean([row[key] for row in rows])) for key in fields}
    summary = dict(seed=args.seed, geometry=args.geometry, sample_count=len(frames), target_count=len(targets),
                   video_counts=dict(Counter(item["video_id"] for item in frames)),
                   groups={group: dict(count=len(rows), **aggregate(rows))
                           for group in sorted({item["group"] for item in frames})
                           for rows in [[item for item in frames if item["group"] == group]]},
                   scope="trainB development audit; no checkpoint, training or testB use",
                   changed="R(x,y) versus full-image mean(R); everything else fixed",
                   limitations=["Mean R matched does NOT guarantee equal final-map brightness.",
                                "Box/crop statistics describe response, NOT boundary accuracy or semantic saliency.",
                                "R-only ablation: spatial clutter D is shared, so the constant control is not free of all spatial heuristics.",
                                "Regular and small-object-challenge groups must remain separate."])
    code_paths = (Path(__file__), ROOT / "pytorch-CycleGAN-and-pix2pix/models/saliency_edges.py",
                  ROOT / "scripts/review_contour_candidates.py")
    manifest = dict(selection=selection, source_hashes={item["filename"]: item["sha256"] for item in frames},
                    code_hashes={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in code_paths},
                    parameters=dict(highs=[.12, .16, .20], low_ratio=.5, width=1., sigmas=[.7, 1., 1.6],
                                    background_gain=.5, clutter_factor=.4, fine_floor=.25), summary=summary,
                    annotation_use="Existing boxes only select/localize review crops; never used to generate maps or as boundary truth.")
    for name, value in (("selection.json", selection), ("summary.json", summary), ("audit_manifest.json", manifest)):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(output / "samples.csv", frames)
    write_csv(output / "targets.csv", targets)
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>C2c 显著性空间权重独立审查</title>
<style>body{font:16px/1.6 system-ui,sans-serif;background:#f4f6f8;color:#18212c;margin:24px}main{max-width:1500px;margin:auto}
article{background:white;padding:18px;border:1px solid #ccd4de;border-radius:8px;margin:22px 0}img{max-width:100%;height:auto}
summary{cursor:pointer;color:#2459a0;margin:12px 0}.notice{background:#fff2cf;padding:16px}h3{overflow-wrap:anywhere}</style><main>
<h1>C2c：空间权重 R 与逐图平均 R 的独立对照</h1>
<p class="notice">seed42；trainB 开发审查，非独立测试。只改变 R，细粗候选、D、0.5 背景增益、0.25 细边底线全部固定。
无训练、无生成器权重。标注框只用于定位，不是轮廓真值；常规组与小目标挑战组分开查看。</p>
<p>对照列：红外、细候选、常数权重、当前空间权重。上行为统一 [0,1] 灰度，下行为同强度红色叠加。
红色不是正确性。重要性图不作逐图拉伸。差异图统一 ×4：红色增强，蓝色减弱。</p>
<p>先判断原图可见边界，再检查：①候选本来缺边？②空间权重误压目标？③目标增强但杂纹也增强？
不要把框内所有边都当正确，也不要把框外道路／建筑主结构都当杂纹。平均 R 相同不意味着最终轮廓图同亮度。</p>
__CARDS__</main></html>'''
    (output / "index.html").write_text(page.replace("__CARDS__", "\n".join(cards)), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(output / "index.html", flush=True)


if __name__ == "__main__":
    main()
