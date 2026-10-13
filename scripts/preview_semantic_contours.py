#!/usr/bin/env python3
"""Use frozen 512 semantic scores to preview important soft contours on trainB."""

import argparse
import hashlib
import html
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pytorch-CycleGAN-and-pix2pix"))

from models.conditioned_generator import ConditionedGenerator
from models.semantic_edges import make_semantic_contours, SETTINGS, IMPORTANT_CLASSES
from semantic_preview.checkpoint import sha256_file
from semantic_preview.inference import CLASS_NAMES
from semantic_preview.review import uint8_image, red_overlay, write_csv
from preview_semantics import DEFAULT_REVIEW, load_review, write_json

DEFAULT_SOURCE = ROOT / "runs/edge_audit/segformer_fmb_b3_semantic_preview_seed42"
DEFAULT_OUTPUT = ROOT / "runs/edge_audit/semantic512_contours_seed42_v1"
MODES = ("c2b", "c2c", "semantic512")
LABELS = {"c2b": "C2b", "c2c": "C2c", "semantic512": "Semantic 512"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic-preview-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "FLIR_datasets/trainB")
    parser.add_argument("--review-manifest", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--protocol-manifest", type=Path, default=ROOT / "FLIR_protocol_v2/manifest.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-images", type=int, default=40)
    parser.add_argument("--zoom", type=int, default=6)
    args = parser.parse_args(argv)
    if args.num_images < 1 or not 1 <= args.zoom <= 12:
        parser.error("Positive image count and zoom 1..12 required")
    if args.input_dir.resolve() != (ROOT / "FLIR_datasets/trainB").resolve():
        parser.error("Use the fixed development trainB review; testB must not select this variant")
    if args.output_dir.exists():
        parser.error("Use a new output directory; existing previews and ratings are preserved")
    return args


def read_cache(path, expected_gray):
    with np.load(path, allow_pickle=False) as archive:
        gray = archive["TIR_uint8"].copy()
        names = tuple(archive["class_names"].tolist())
        probabilities = archive["base_probabilities"].copy()
        labels = archive["base_labels"].copy()
    if gray.dtype != np.uint8 or not np.array_equal(gray, expected_gray):
        raise ValueError(f"Cached TIR differs from the fixed original image: {path}")
    if names != CLASS_NAMES or probabilities.dtype != np.float32 or probabilities.shape != (15, *gray.shape):
        raise ValueError(f"Wrong taxonomy, dtype or restored coordinates: {path}")
    if labels.shape != gray.shape or not np.array_equal(labels, probabilities.argmax(0)):
        raise ValueError(f"Cache labels disagree with categorical scores: {path}")
    # The helper also rejects nonfinite, nonunit and unnormalized class scores.
    from models.semantic_edges import semantic_importance
    semantic_importance(probabilities, names)
    return gray, probabilities


def preflight(args):
    review, selected, targets = load_review(args)
    source = args.semantic_preview_dir
    manifest_path = source / "preview_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    status = json.loads((source / "status.json").read_text(encoding="utf-8"))
    if status.get("phase") != "completed" or status.get("completed") != len(manifest["frames"]):
        raise ValueError("Semantic preview cache is incomplete")
    settings = manifest["settings"]
    if settings["short_side"] != 512 or settings.get("baseline_flip_tta") is not False:
        raise ValueError("Use the fixed 512 baseline scores, without baseline flip averaging")
    if manifest["review_manifest_sha256"] != sha256_file(args.review_manifest):
        raise ValueError("Semantic preview belongs to a different fixed review")
    if manifest["protocol_manifest_sha256"] != sha256_file(args.protocol_manifest):
        raise ValueError("Semantic preview belongs to a different data protocol")
    for relative, digest in manifest["code_hashes"].items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or sha256_file(path) != digest:
            raise ValueError(f"Original semantic inference code changed: {relative}")
    frames = {row["filename"]: row for row in manifest["frames"]}
    if len(frames) != len(manifest["frames"]):
        raise ValueError("Duplicate semantic cache source images")
    cache_paths, cache_hashes = {}, {}
    for row in selected:
        name = row["filename"]
        frame = frames.get(name)
        if frame is None or frame["sha256"] != review["source_hashes"][name]:
            raise ValueError(f"Missing or mismatched semantic source: {name}")
        prefix = frame["prefix"]
        if not prefix.isdigit():
            raise ValueError("Semantic cache prefix must be numeric")
        path = source / "maps" / f"{prefix}_responses.npz"
        with Image.open(args.input_dir / name) as image:
            expected_gray = np.asarray(image).copy()
        read_cache(path, expected_gray)
        cache_paths[name] = path
        cache_hashes[name] = sha256_file(path)
    return manifest, selected, targets, cache_paths, cache_hashes


def make_sheet(gray, maps, crop=None, box=None, zoom=1, *, modes=MODES, labels=LABELS,
               importance_key="semantic_importance", importance_caption="R512 / region weight"):
    """Raw contours and overlays at identical scale; box marks only the TIR."""
    if len(modes) != 3:
        raise ValueError("Comparison sheet requires exactly three contour versions")
    if crop is None:
        crop = (0, 0, gray.shape[1], gray.shape[0])
    x0, y0, x1, y1 = crop
    local = gray[y0:y1, x0:x1]
    h, w = local.shape
    panel_width = max(w * zoom, 150)
    panel_height = h * zoom + 28
    sheet = Image.new("RGB", (panel_width * 5, panel_height * 2), "white")
    draw = ImageDraw.Draw(sheet)
    for row in range(2):
        panels = [("TIR / yellow target", Image.fromarray(local).convert("RGB"))]
        for mode in modes:
            values = maps[mode][y0:y1, x0:x1]
            panel = uint8_image(values).convert("RGB") if row == 0 else red_overlay(local, values)
            panels.append((labels[mode] + (" [0,1]" if row == 0 else " overlay"), panel))
        importance = maps[importance_key][y0:y1, x0:x1]
        panels.append((importance_caption, uint8_image(importance).convert("RGB")))
        for col, (caption, panel) in enumerate(panels):
            left = col * panel_width + (panel_width - w * zoom) // 2
            top = row * panel_height + 25
            sheet.paste(panel.resize((w * zoom, h * zoom), Image.Resampling.NEAREST), (left, top))
            draw.text((col * panel_width + 4, row * panel_height + 4), caption, fill="black")
            if col == 0 and box is not None:
                bx0, by0, bx1, by1 = box
                draw.rectangle((left + (bx0-x0)*zoom, top + (by0-y0)*zoom,
                                left + (bx1-x0)*zoom-1, top + (by1-y0)*zoom-1),
                               outline="yellow", width=1)
    return sheet


def write_page(path, frames, targets, settings, pack_id):
    esc = html.escape
    cards = []
    navigation = '<nav>' + ' '.join(f'<a href="#frame-{f["prefix"]}">{f["prefix"]}</a>' for f in frames) + '</nav>'
    for frame in frames:
        prefix = frame["prefix"]
        panels = [(f"frames/{prefix}_TIR.png", "原始红外"),
                  (f"frames/{prefix}_semantic_importance.png", "512语义重要性 R（区域图）")]
        panels += [(f"frames/{prefix}_{mode}.png", f"{LABELS[mode]} 软轮廓") for mode in MODES]
        panels += [(f"frames/{prefix}_semantic512_overlay.png", "新轮廓叠加：红色仅表示强度")]
        grid = ''.join(f'<figure><figcaption>{esc(label)}</figcaption><a href="{src}" target="_blank"><img loading="lazy" src="{src}"></a></figure>' for src,label in panels)
        crops = []
        for target in targets:
            if target["filename"] != frame["filename"]:
                continue
            w,h = target["box_size_pixels"]
            src = f'targets/{target["panel"]}'
            crops.append(f'<section id="{esc(target["id"])}"><h4>{esc(target["id"])} / 当前目标：{esc(target["category"])} / {w:.1f}×{h:.1f}像素</h4>'
                         '<p>黄色框仅在红外面板定位当前目标。第一行看灰度轮廓；第二行看红色叠加。右侧 R 是语义区域权重。</p>'
                         f'<a href="{src}" target="_blank"><img loading="lazy" class="crop" src="{src}"></a></section>')
        cards.append(f'<article id="frame-{prefix}"><h3>{prefix} / {esc(frame["group"])} / {esc(frame["filename"])}</h3><div class="grid">{grid}</div>'
                     f'<details><summary>展开固定目标裁剪（{len(crops)}个）</summary>{"".join(crops)}</details></article>')
    document = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>512语义引导软轮廓：固定样本对照</title><style>
body{font:16px/1.6 system-ui,sans-serif;margin:20px;background:#f4f6f8;color:#18212c}main{max-width:1320px;margin:auto}
article,.notice{background:white;padding:18px;border:1px solid #ccd4de;border-radius:8px;margin:18px 0}.notice{background:#fff4d7}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px}figure{margin:0}figure img{width:100%}
.crop{max-width:100%;height:auto;image-rendering:pixelated}section{border-top:1px solid #ddd;margin-top:18px}h3{overflow-wrap:anywhere}
nav{position:sticky;top:0;background:white;padding:8px;z-index:1}nav a{display:inline-block;padding:2px 5px}summary{cursor:pointer;color:#2459a0}
</style><main><h1>512语义引导软轮廓：C2b / C2c / 新候选</h1>
<div class="notice"><p>固定 trainB 全图推理结果；沿用原有轮廓阈值、尺度、软化宽度、纹理抑制和25%细边保底，只替换细粗融合的重要性权重。</p>
<p>S = 行人 + 车辆组 + 建筑的语义软响应；R = 0.18 + 0.67 S。S、R不是边缘置信度。完整语义区域响应可填满人体；轮廓图只显示已有候选边。</p>
<p>重点看31–40张中的小目标：轮廓是否变强、植被细纹理是否变弱、错误语义是否误压重要边。所有灰度/红色图使用固定[0,1]，未逐图增亮。</p>
<p>这一步不能恢复提边器完全遗漏的轮廓。语义错误与建筑内部纹理也可能被传播；保底不保证轮廓完整。框仅定位审查目标，不用于生成轮廓。</p>
<p>浮点结果在 maps/；描述性统计在 frames.csv 和 targets.csv，不是边缘准确率。既有GAN训练模式和预览评分独立保留。</p></div>'''
    path.write_text(document + navigation + ''.join(cards) + f'<p>Pack: {esc(pack_id)}</p></main></html>', encoding="utf-8")


def weighted_mean(values, weights):
    return float((values * weights).sum(dtype=np.float64) / weights.sum(dtype=np.float64)) if weights.sum() > 0 else None


def run(args):
    torch.set_num_threads(1)
    source, selected, source_targets, cache_paths, cache_hashes = preflight(args)
    code_paths = [Path(__file__), ROOT / "pytorch-CycleGAN-and-pix2pix/models/semantic_edges.py",
                  ROOT / "pytorch-CycleGAN-and-pix2pix/models/saliency_edges.py",
                  ROOT / "pytorch-CycleGAN-and-pix2pix/models/conditioned_generator.py",
                  ROOT / "scripts/semantic_preview/review.py", ROOT / "scripts/preview_semantics.py"]
    code_hashes = {str(p.relative_to(ROOT)): sha256_file(p) for p in code_paths}
    identity = dict(source_pack_id=source["pack_id"],source_manifest_sha256=sha256_file(args.semantic_preview_dir/"preview_manifest.json"),
                    cache_sha256=cache_hashes,settings=SETTINGS,important_classes=IMPORTANT_CLASSES,code_hashes=code_hashes,
                    selected=selected,zoom=args.zoom)
    pack_id = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    output = args.output_dir
    for name in ("frames", "maps", "targets"):
        (output/name).mkdir(parents=True,exist_ok=False)
    baseline_b = ConditionedGenerator(nn.Identity(), "soft_multi", fusion_mode="direct")
    baseline_c = ConditionedGenerator(nn.Identity(), "soft_saliency", fusion_mode="direct")
    frames, targets, frame_rows, target_rows = [], [], [], []
    started = time.perf_counter()
    write_json(output/"status.json",dict(phase="running",completed=0,total=len(selected)))
    print(f"Preflight OK: {len(selected)} fixed trainB frames, {len(source_targets)} targets; cached 512 scores only.",flush=True)
    try:
        for number, row in enumerate(selected, 1):
            name = row["filename"]
            with Image.open(args.input_dir/name) as image:
                expected_gray = np.asarray(image).copy()
            gray, probabilities = read_cache(cache_paths[name],expected_gray)
            # Match the actual conditioned generator's normalize/de-normalize path.
            tir = torch.from_numpy((gray.astype(np.float32)/255.0-0.5)/0.5)[None,None]
            image = np.clip((tir.numpy()[0,0]+1)/2,0,1)
            maps = make_semantic_contours(image,probabilities,CLASS_NAMES)
            for mode, model in (("c2b",baseline_b),("c2c",baseline_c)):
                actual, reserved = model.make_condition(tir)
                np.testing.assert_array_equal(actual[0,0].numpy(),maps[mode])
                if torch.count_nonzero(reserved).item():
                    raise RuntimeError("Reserved confidence channel must remain zero")
            condition = baseline_c(tir,edge_override=torch.from_numpy(maps["semantic512"])[None,None])
            np.testing.assert_array_equal(condition[:,0:1].numpy(),tir.numpy())
            np.testing.assert_array_equal(condition[0,1].numpy(),maps["semantic512"])
            if torch.count_nonzero(condition[:,2:3]).item():
                raise RuntimeError("Direct fusion must pass [TIR, semantic soft contour, zero]")
            if np.any(maps["semantic512"]+1e-7 < SETTINGS["fine_floor"]*maps["fine"]):
                raise RuntimeError("Fine-edge floor violated")
            prefix=f"{number:02d}"
            Image.fromarray(gray).save(output/"frames"/f"{prefix}_TIR.png")
            for key,values in maps.items():
                uint8_image(values).save(output/"frames"/f"{prefix}_{key}.png")
                if key in MODES:
                    red_overlay(gray,values).save(output/"frames"/f"{prefix}_{key}_overlay.png")
            make_sheet(gray,maps).save(output/"frames"/f"{prefix}_comparison.png")
            np.savez_compressed(output/"maps"/f"{prefix}_contours.npz",TIR_uint8=gray,**maps)
            frame=dict(prefix=prefix,filename=name,group=row["group"],video_id=row["video_id"])
            frames.append(frame)
            vegetation=probabilities[CLASS_NAMES.index("vegetation")]
            stats=dict(filename=name,group=row["group"],mean_R512=float(maps["semantic_importance"].mean()),
                       mean_abs_change_c2c=float(np.abs(maps["semantic512"]-maps["c2c"]).mean()))
            for mode in MODES:
                stats[mode+"_mean"]=float(maps[mode].mean())
                stats[mode+"_vegetation_weighted_strength_proxy"]=weighted_mean(maps[mode],vegetation)
                stats[mode+"_important_weighted_strength_proxy"]=weighted_mean(maps[mode],maps["semantic_support"])
            frame_rows.append(stats)
            for item in source_targets:
                if item["filename"]!=name:
                    continue
                panel=f'{item["id"]}_contours.png'
                make_sheet(gray,maps,item["crop"],item["box"],args.zoom).save(output/"targets"/panel)
                x0,y0,x1,y1=item["box"]
                target=dict(item,panel=panel,box_size_pixels=[x1-x0,y1-y0])
                targets.append(target)
                region=np.s_[int(np.floor(y0)):int(np.ceil(y1)),int(np.floor(x0)):int(np.ceil(x1))]
                ts=dict(id=item["id"],filename=name,group=item["group"],category=item["category"],
                        width_pixels=x1-x0,height_pixels=y1-y0,
                        mean_R512_proxy=float(maps["semantic_importance"][region].mean()))
                for mode in MODES:
                    ts[mode+"_box_mean_strength_proxy"]=float(maps[mode][region].mean())
                target_rows.append(ts)
            write_json(output/"status.json",dict(phase="running",completed=number,total=len(selected)))
            print(f"[{number}/{len(selected)}] {name}",flush=True)
        write_csv(output/"frames.csv",frame_rows)
        write_csv(output/"targets.csv",target_rows)
        write_page(output/"index.html",frames,targets,SETTINGS,pack_id)
        # Choose quicklook frames by fixed IDs, never by output quality.
        quick_ids=("01","02","31","34")
        quick=[np.asarray(Image.open(output/"frames"/f"{f['prefix']}_comparison.png")) for f in frames if f["prefix"] in quick_ids]
        if quick:
            Image.fromarray(np.concatenate(quick,axis=0)).save(output/"quicklook.png")
        description=dict(format="semantic_contour_preview_v1",pack_id=pack_id,identity=identity,
                         source_semantic_preview=str(args.semantic_preview_dir),semantic_mode="base",short_side=512,
                         frames=frames,targets=targets,elapsed_seconds=time.perf_counter()-started,
                         environment=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,device="cpu"),
                         controls="C2b and C2c checked pixel-for-pixel against existing ConditionedGenerator",
                         interface="Existing direct edge_override checked as [TIR, edge, zero] on every frame; no training mode registered",
                         limits=["No pixel boundary truth; strength proxies are not accuracy.","Semantic responses are an external supervised FMB prior, not edge confidence.",
                                 "Canny-missing contours cannot be recovered by importance fusion.","Review boxes only choose crops and descriptive statistics.",
                                 "Full-frame development preview; online crop/flip integration and RGB quality still require separate experiments."])
        write_json(output/"preview_manifest.json",description)
        write_json(output/"status.json",dict(phase="completed",completed=len(frames),total=len(selected),targets=len(targets),elapsed_seconds=description["elapsed_seconds"]))
        print(f"Wrote {output/'index.html'}",flush=True)
    except BaseException as error:
        write_json(output/"status.json",dict(phase="failed_or_interrupted",completed=len(frames),total=len(selected),error=type(error).__name__,message=str(error)))
        raise


if __name__ == "__main__":
    run(parse_args())
