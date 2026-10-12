"""CPU trainB audit of the opt-in R-only region-recall prototype.

Old/new maps share candidates, clutter and fusion. A third, diagnostic-only
global gain control matches the NEW output mean, so brighter is not better.
Annotation boxes select review crops only; they are not boundary supervision.
"""

from __future__ import annotations

import argparse
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
from audit_c2_importance import fuse_fixed_candidates, priority_objects, write_csv
from review_contour_candidates import NAMES, records_from_annotations, select_records
from models.conditioned_generator import ConditionedGenerator
from models.saliency_edges import (
    make_adaptive_saliency_edge, region_recall_importance, structure_texture_importance,
)


def brightness_control(baseline, enhanced):
    """Match final-map mean by uniform gain + clipping, not by per-map display."""
    baseline, enhanced = (np.asarray(value, np.float32) for value in (baseline, enhanced))
    if baseline.shape != enhanced.shape or not baseline.size:
        raise ValueError("Brightness control requires matching nonempty maps")
    if any(not np.isfinite(value).all() or np.any((value < 0) | (value > 1))
           for value in (baseline, enhanced)):
        raise ValueError("Brightness-control maps must be finite and in [0, 1]")
    target = float(enhanced.mean(dtype=np.float64))
    if target < float(baseline.mean(dtype=np.float64)) - 1e-7:
        raise ValueError("The enhancement must not reduce the baseline mean")
    if np.any((baseline == 0) & (enhanced > 0)):
        raise ValueError("R-only enhancement cannot add support outside the baseline")
    if np.array_equal(baseline, enhanced):
        return baseline.copy(), 1.0
    lower, upper = 1.0, 2.0
    response = lambda gain: np.minimum(baseline.astype(np.float64) * gain, 1).astype(np.float32)
    while response(upper).mean(dtype=np.float64) < target and upper < 2**40:
        upper *= 2
    if response(upper).mean(dtype=np.float64) < target - 1e-7:
        raise ValueError("Cannot match enhanced mean using global gain")
    for _ in range(48):
        midpoint = (lower + upper) / 2
        if response(midpoint).mean(dtype=np.float64) < target:
            lower = midpoint
        else:
            upper = midpoint
    return response((lower + upper) / 2), (lower + upper) / 2


def compare_importance(gray):
    baseline, old_R, fine, coarse = make_adaptive_saliency_edge(
        gray, (.12, .16, .20), .5, 1., (.7, 1., 1.6), .5,
    )
    _, clutter = structure_texture_importance(gray, fine, coarse)
    new_R, cues = region_recall_importance(gray, fine, coarse, old_R)
    enhanced, before_floor = fuse_fixed_candidates(fine, coarse, new_R, clutter)
    control, gain = brightness_control(baseline, enhanced)
    return dict(fine=fine, coarse=coarse, clutter=clutter, old_R=old_R, new_R=new_R,
                baseline=baseline, enhanced=enhanced, brightness_control=control,
                before_floor=before_floor, **cues), gain


def response_stats(maps):
    old, new, control = (maps[key] for key in ("baseline", "enhanced", "brightness_control"))
    delta = new - old
    return dict(old_mean=float(old.mean()), enhanced_mean=float(new.mean()),
                control_mean=float(control.mean()), R_increment_mean=float((maps["new_R"] - maps["old_R"]).mean()),
                enhancement_mae=float(np.abs(delta).mean()),
                brighter_fraction_delta_003=float((delta > .03).mean()),
                dimmer_fraction_delta_003=float((delta < -.03).mean()),
                versus_global_gain_mae=float(np.abs(new - control).mean()),
                new_above_control_003_fraction=float((new - control > .03).mean()),
                new_below_control_003_fraction=float((new - control < -.03).mean()))


def render_panel(gray, arrays, path, zoom=1):
    h, w = gray.shape
    cell = max(180, w * zoom)
    sheet = Image.new("RGB", (cell * len(arrays), h * zoom * 2 + 58), "white")
    draw = ImageDraw.Draw(sheet)
    base = np.repeat(gray[..., None], 3, -1)
    for column, (label, values) in enumerate(arrays.items()):
        draw.text((column * cell + 4, 4), label, fill="black")
        top = np.repeat(values[..., None], 3, -1)
        alpha = values[..., None] * .65
        overlay = base if label == "TIR" else base * (1 - alpha) + np.array([1, 0, 0]) * alpha
        for array, y in ((top, 23), (overlay, 23 + h * zoom)):
            image = Image.fromarray(np.uint8(np.clip(array, 0, 1) * 255))
            sheet.paste(image.resize((w * zoom, h * zoom), Image.Resampling.NEAREST),
                        (column * cell + (cell - w * zoom) // 2, y))
    draw.text((4, h * zoom * 2 + 31), "Fixed [0,1]. Red = response, NOT correctness; gain control matches full-image mean.", fill="black")
    sheet.save(path)


def comparison_columns(gray, maps):
    return {"TIR": gray, "fine candidate": maps["fine"], "old C2c": maps["baseline"],
            "global-gain control": maps["brightness_control"], "region-recall C2c": maps["enhanced"]}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "FLIR_datasets/trainB")
    parser.add_argument("--manifest", type=Path, default=ROOT / "FLIR_protocol_v2/manifest.csv")
    parser.add_argument("--coco", type=Path, default=ROOT / "FLIR_ADAS_v2/images_thermal_train/coco.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-audit", type=Path, help="Check old maps and image hashes against a prior importance audit")
    parser.add_argument("--selection-json", type=Path)
    parser.add_argument("--sample-count", type=int, default=24)
    parser.add_argument("--challenge-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--geometry", choices=("full", "center_crop_256"), default="full")
    args = parser.parse_args(argv)
    if args.processed_dir.resolve() != (ROOT / "FLIR_datasets/trainB").resolve():
        parser.error("Development preview requires project trainB; never testB")
    if args.output_dir.exists():
        parser.error("Existing output will not be overwritten; choose a new directory")
    if args.sample_count < 1 or args.challenge_count < 0:
        parser.error("sample-count >= 1 and challenge-count >= 0 required")
    if args.reference_audit and args.selection_json is None:
        args.selection_json = args.reference_audit / "selection.json"
    return args


def main():
    args = parse_args()
    torch.set_num_threads(1)
    selected, selection = select_records(records_from_annotations(args), args)
    if selection["seed"] != args.seed:
        raise ValueError("Reused selection seed differs from --seed")
    reference = None
    if args.reference_audit:
        reference = json.loads((args.reference_audit / "audit_manifest.json").read_text(encoding="utf-8"))
        if reference["summary"]["geometry"] != args.geometry or reference["selection"] != selection:
            raise ValueError("Reference geometry/sample selection differs")
    output = args.output_dir
    for folder in ("frames", "targets", "maps"):
        (output / folder).mkdir(parents=True, exist_ok=False)
    official = ConditionedGenerator(nn.Identity(), "soft_saliency", fusion_mode="direct",
                                    saliency_variant="region_recall")
    frames, targets, cards = [], [], []
    for number, record in enumerate(selected, 1):
        source = args.processed_dir / record["filename"]
        with Image.open(source) as image:
            if image.mode != "L" or image.size != (360, 288):
                raise ValueError(f"Expected processed L-mode 360x288 TIR: {source}")
        _, tir = load_tir(source, args.geometry)
        gray = np.clip((tir[0, 0].numpy() + 1) / 2, 0, 1)
        maps, gain = compare_importance(gray)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        prefix = f"{number:02d}"
        if reference:
            if reference["source_hashes"][record["filename"]] != digest:
                raise ValueError("Reference source image changed")
            with np.load(args.reference_audit / "maps" / f"{prefix}_maps.npz") as archived:
                for key, old_key in (("fine", "fine"), ("coarse", "coarse"), ("clutter", "clutter"),
                                     ("old_R", "R"), ("baseline", "spatial")):
                    np.testing.assert_array_equal(maps[key], archived[old_key])
        actual, reserved = official.make_condition(tir)
        np.testing.assert_array_equal(actual[0, 0].numpy(), maps["enhanced"])
        if reserved.count_nonzero().item():
            raise RuntimeError("Reserved third channel changed")
        if np.any(maps["enhanced"] < maps["baseline"] - 1e-7):
            raise RuntimeError("R-only rescue reduced an existing response")
        np.testing.assert_allclose(maps["brightness_control"].mean(dtype=np.float64),
                                   maps["enhanced"].mean(dtype=np.float64), atol=1e-7, rtol=0)
        comparison = f"frames/{prefix}_comparison.png"
        cues_file = f"frames/{prefix}_cues.png"
        render_panel(gray, comparison_columns(gray, maps), output / comparison)
        render_panel(gray, {"old R": maps["old_R"], "new R": maps["new_R"],
                           "relative detail": maps["relative_detail"],
                           "region neighborhood": maps["neighborhood_support"]}, output / cues_file)
        np.savez_compressed(output / "maps" / f"{prefix}_maps.npz", TIR=gray, **maps)
        frames.append(dict(index=number, filename=record["filename"], video_id=record["video_id"],
                           group=record["group"], sha256=digest, global_gain=gain, **response_stats(maps)))
        crop_cards = []
        for object_number, obj in enumerate(priority_objects(record, args.geometry), 1):
            box = obj["box"]
            margin = max(6, round(max(box[2] - box[0], box[3] - box[1]) * .25))
            crops = [("object", (box[0] - margin, box[1] - margin, box[2] + margin, box[3] + margin))]
            if obj["category"] == "person" and not obj["clipped"]:
                crops.append(("lower_bbox_proxy", (box[0] - margin, box[3] - max(6, (box[3] - box[1]) * .35),
                                                    box[2] + margin, box[3] + margin)))
            for kind, raw in crops:
                x1, y1, x2, y2 = (max(0, math.floor(raw[0])), max(0, math.floor(raw[1])),
                                  min(gray.shape[1], math.ceil(raw[2])), min(gray.shape[0], math.ceil(raw[3])))
                cropped = {key: value[y1:y2, x1:x2] for key, value in maps.items()}
                target_id = f"{prefix}_O{object_number:02d}" + ("_contact" if kind == "lower_bbox_proxy" else "")
                crop_file = f"targets/{target_id}_comparison.png"
                render_panel(gray[y1:y2, x1:x2], comparison_columns(gray[y1:y2, x1:x2], cropped),
                             output / crop_file, zoom=min(6, max(1, math.ceil(160 / (y2 - y1)))))
                targets.append(dict(id=target_id, filename=record["filename"], group=record["group"],
                                    category=obj["category"], annotation_id=obj["annotation_id"], kind=kind,
                                    clipped=obj["clipped"], crop=json.dumps([x1, y1, x2, y2]), **response_stats(cropped)))
                label = f"{target_id} {NAMES[obj['category']]}"
                if kind == "lower_bbox_proxy":
                    label += "：框下部接触区域代理，不是脚部真值"
                crop_cards.append(f'<h4>{html.escape(label)}</h4><img loading="lazy" src="{crop_file}">')
        cards.append(f'<article><h3>{prefix} | {record["group"]} | {html.escape(record["filename"])}</h3>'
                     f'<img loading="lazy" src="{comparison}"><details><summary>重要性与支持信号</summary>'
                     f'<img loading="lazy" src="{cues_file}"></details><details><summary>固定目标与接触区域</summary>'
                     f'{"".join(crop_cards)}</details></article>')
        print(f"Compared {number}/{len(selected)}", flush=True)
    fields = list(response_stats(maps))
    summary = dict(seed=args.seed, geometry=args.geometry, frame_count=len(frames),
                   object_crop_count=sum(row["kind"] == "object" for row in targets),
                   contact_proxy_count=sum(row["kind"] == "lower_bbox_proxy" for row in targets),
                   video_counts=dict(Counter(row["video_id"] for row in frames)),
                   archived_baseline_bitwise_verified=reference is not None,
                   groups={group: {key: float(np.mean([row[key] for row in frames if row["group"] == group]))
                                   for key in fields} for group in sorted({row["group"] for row in frames})},
                   scope="trainB development response audit; no training, checkpoint or testB use",
                   changed="importance R only; fine/coarse/D/fusion/floor fixed",
                   limitations=["Response changes are NOT boundary recall or correctness.",
                                "Boxes and lower-box crops are review locations, not object/boundary labels.",
                                "Global gain is a diagnostic control, not a new training variant.",
                                "Gain is matched on the full image, never recalculated for crops.",
                                "This prototype cannot invent a boundary absent from the candidates."])
    code_paths = [Path(__file__), ROOT / "scripts/audit_c2_importance.py",
                  ROOT / "scripts/review_contour_candidates.py", ROOT / "scripts/audit_c2_conditions.py"]
    code_paths += [ROOT / "pytorch-CycleGAN-and-pix2pix/models" / name
                   for name in ("saliency_edges.py", "conditioned_generator.py")]
    manifest = dict(summary=summary, selection=selection,
                    source_hashes={row["filename"]: row["sha256"] for row in frames},
                    code_hashes={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in code_paths},
                    parameters=dict(highs=[.12, .16, .20], low_ratio=.5, width=1., sigmas=[.7, 1., 1.6],
                                    background_gain=.5, clutter_factor=.4, fine_floor=.25,
                                    relative_gradient_sigmas=[.7, 1.], local_gradient_sigma=3., gradient_floor=.015,
                                    region_sigmas=[2., 4., 8.], contrast_floor=.02, neighborhood_size=5,
                                    rescue_smoothing_sigma=.7, advantage_scale=.3, importance_ceiling=.85, rescue_strength=.65),
                    annotation_use="Select/locate review crops only; never used in edge generation or training")
    for name, value in (("selection.json", selection), ("summary.json", summary), ("audit_manifest.json", manifest)):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(output / "samples.csv", frames)
    write_csv(output / "targets.csv", targets)
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>C2c 弱轮廓保护原型</title>
<style>body{font:16px/1.6 system-ui,sans-serif;background:#f4f6f8;color:#18212c;margin:24px}main{max-width:1800px;margin:auto}
article{background:white;padding:18px;border:1px solid #ccd4de;border-radius:8px;margin:22px 0}img{max-width:100%;height:auto}
summary{cursor:pointer;color:#2459a0;margin:12px 0}.notice{background:#fff2cf;padding:16px}h3{overflow-wrap:anywhere}</style><main>
<h1>C2c：已有弱轮廓保护，不是轮廓补全</h1><p class="notice">seed__SEED__；trainB 开发预览，非独立测试。
只增强 R；细粗候选、D、融合、0.25 细边底线不变，无训练、无置信度、无语义分类。</p>
<p>列顺序：红外、细候选、旧 C2c、整体提亮对照、新 region_recall。统一 [0,1] 显示，不逐图拉伸。
整体提亮对照与新分支的完整图平均响应相同；目标裁剪复用完整图的结果，不重新匹配亮度。</p>
<p>框只定位；框下部不保证是脚，也不是边界标注。看原图是否有可见边界，再检查弱边是否更连续、是否只变亮、杂纹是否也增强。
新分支只保护已有候选，不推断不可见边界；更亮和阈值越过不等于准确率／召回提升。</p>__CARDS__</main></html>'''
    (output / "index.html").write_text(page.replace("__SEED__", str(args.seed)).replace("__CARDS__", "\n".join(cards)), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(output / "index.html", flush=True)


if __name__ == "__main__":
    main()
