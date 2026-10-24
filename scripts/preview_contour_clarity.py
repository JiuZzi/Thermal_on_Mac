#!/usr/bin/env python3
"""Review candidate clarity with fixed raw-TIR semantic/importance weights."""

import argparse
import html
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pytorch-CycleGAN-and-pix2pix"))
from models.clarity_semantic_edges import CLARITY_SETTINGS, clarity_components
from models.semantic_contour import FrozenSemanticPrior, DEFAULT_CHECKPOINT
from preview_semantic_contours import DEFAULT_SOURCE, preflight, read_cache, make_sheet
from preview_semantics import DEFAULT_REVIEW, write_json
from semantic_preview.checkpoint import sha256_file
from semantic_preview.review import uint8_image, red_overlay, write_csv


MODES = ("selective_v1", "narrow_only", "clarity_v1")
LABELS = dict(selective_v1="Previous selective_v1", narrow_only="Width-only control", clarity_v1="Clarity candidate")
PROBES = ("vote_only", "sharpened_canny", "weber_raw", "weber_enhanced")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--semantic-preview-dir", type=Path, default=DEFAULT_SOURCE)
    p.add_argument("--semantic-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--previous-contours-dir", type=Path, default=ROOT / "runs/edge_audit/selective_contours_seed42_floor_v1")
    p.add_argument("--input-dir", type=Path, default=ROOT / "FLIR_datasets/trainB")
    p.add_argument("--review-manifest", type=Path, default=DEFAULT_REVIEW)
    p.add_argument("--protocol-manifest", type=Path, default=ROOT / "FLIR_protocol_v2/manifest.csv")
    p.add_argument("--output-dir", type=Path, default=ROOT / "runs/edge_audit/contour_clarity_seed42_v1")
    p.add_argument("--num-images", type=int, default=40)
    p.add_argument("--cpu-threads", type=int, default=2)
    p.add_argument("--zoom", type=int, default=6)
    args = p.parse_args(argv)
    if args.output_dir.exists(): p.error("Preserve previous reviews; choose a new output directory")
    if args.input_dir.resolve() != (ROOT / "FLIR_datasets/trainB").resolve():
        p.error("Only the fixed development review; no test-set selection")
    if args.num_images < 1 or args.cpu_threads < 1 or not 1 <= args.zoom <= 12:
        p.error("Positive count/threads, zoom 1..12 required")
    return args


def write_page(output, frames, targets):
    intro = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>轮廓清晰度对照</title><style>body{font:16px/1.6 system-ui;background:#f4f6f8;color:#1d2735;margin:20px}main{max-width:1600px;margin:auto}article,.notice{padding:18px;background:white;border:1px solid #ccd3dc;border-radius:8px;margin:18px 0}.notice{background:#fff5dd}img{max-width:100%}h2{overflow-wrap:anywhere}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:10px}figure{margin:0}figure img{width:100%}.crop{image-rendering:pixelated}</style>
<main><h1>轮廓清晰度对照</h1><div class="notice">
<p>主对照从左到右：原始红外、上轮 selective_v1、仅改变脊线/软编码宽度、轻度去噪＋硬票整合候选、同一重要性权重。下排是红色叠加。</p>
<p>所有灰度图按原始 [0,1] 显示，不逐图拉伸；新软宽度0.5像素，旧版1像素。“只改宽度”列用于区分表示变细与检测改善。</p>
<p>当前语义、结构重要性、纹理权重均固定，增强图只供提边，生成器第一通道及语义输入仍用原红外。未新增补边或置信度。</p>
<p>细边保底系数仍为25%，改为原始细边的保留脊线及新宽度；不承诺旧版每个像素的保底。更细不等于边界更准确。</p>
<p>Weber两列只是明确公式的独立探针，不是FWGAN复现；本轮使用无训练权重的NLM，不使用SwinIR。锐化保留为消融，不作为主候选默认。</p>
<p>固定目标框只定位审查区域，不参与算法。线宽、强度、连通域等均为描述性统计，不能替代真实边界准确率或RGB训练收益。</p>
<p><a href="frame_statistics.csv">全图统计</a> · <a href="target_statistics.csv">目标框统计</a> · <a href="preview_manifest.json">配置与来源</a></p></div>'''
    cards = []
    for frame in frames:
        n = frame["prefix"]
        figures = ''.join(f'<figure><figcaption>{html.escape(caption)}</figcaption><a href="frames/{n}_{key}.png"><img loading="lazy" src="frames/{n}_{key}.png"></a></figure>'
            for key, caption in (("TIR", "原始红外"), ("restored", "轻度去噪后（仅提边输入）"),
                ("enhanced", "去噪＋锐化（独立消融）"), ("vote_only", "仅硬票整合"),
                ("sharpened_canny", "额外锐化的Canny"), ("weber_raw", "原图Weber探针"),
                ("weber_enhanced", "增强图Weber探针"), ("uniform_dim", "旧版同均值变暗")))
        crops = ''.join(f'<h4>{html.escape(t["id"])} / {html.escape(t["category"])}</h4><img class="crop" loading="lazy" src="targets/{t["id"]}.png">'
                       for t in targets if t["filename"] == frame["filename"])
        cards.append(f'<article id="f{n}"><h2>{n} / {html.escape(frame["filename"])}</h2><img loading="lazy" src="frames/{n}_comparison.png"><details><summary>查看前处理与独立探针</summary><div class="grid">{figures}</div></details><details><summary>固定目标裁剪</summary>{crops}</details></article>')
    (output / "index.html").write_text(intro + ''.join(cards) + '</main></html>', encoding="utf-8")


def response_stats(values):
    from scipy.ndimage import label
    from skimage.morphology import thin
    hard = values >= .2
    skeleton = thin(hard)
    return dict(mean=float(values.mean(dtype=np.float64)), pixels_ge_02=int(hard.sum()),
                skeleton_pixels_ge_02=int(skeleton.sum()),
                area_per_skeleton_ge_02=float(hard.sum() / max(int(skeleton.sum()), 1)),
                components_ge_02=int(label(hard, structure=np.ones((3, 3), bool))[1]))


def run(args):
    torch.set_num_threads(args.cpu_threads)
    prior = FrozenSemanticPrior(args.semantic_checkpoint)
    header = json.loads((args.semantic_preview_dir / "preview_manifest.json").read_text())
    if prior.info["sha256"] != header["checkpoint"]["sha256"]:
        raise ValueError("Frozen checkpoint differs from cached semantic responses")
    replay_count = 0
    def replay(gray):
        nonlocal replay_count
        p = prior.predict(gray.astype(np.float32) / 255, "cpu")
        replay_count += 1; print(f"Semantic replay [{replay_count}/{args.num_images}]", flush=True)
        return p
    source, selected, targets, caches, cache_hashes = preflight(args, replay_probabilities=replay)
    old = json.loads((args.previous_contours_dir / "preview_manifest.json").read_text())
    status = json.loads((args.previous_contours_dir / "status.json").read_text())
    if status.get("phase") != "completed" or old["identity"]["semantic_pack"] != source["pack_id"]:
        raise ValueError("Previous selective review incomplete or has a different semantic source")
    old_frames = {f["filename"]: f for f in old["frames"]}
    output = args.output_dir
    for name in ("frames", "maps", "targets"):
        (output / name).mkdir(parents=True, exist_ok=False)
    identity = dict(settings=CLARITY_SETTINGS, semantic_pack=source["pack_id"],
        semantic_cache_sha256=cache_hashes, cache_revalidation=source.get("cache_revalidation"),
        previous_manifest_sha256=sha256_file(args.previous_contours_dir / "preview_manifest.json"),
        code_sha256={str(p.relative_to(ROOT)): sha256_file(p) for p in (
            Path(__file__), ROOT / "pytorch-CycleGAN-and-pix2pix/models/clarity_semantic_edges.py",
            ROOT / "pytorch-CycleGAN-and-pix2pix/models/selective_semantic_edges.py",
            ROOT / "pytorch-CycleGAN-and-pix2pix/models/semantic_edges.py")})
    frames, frame_stats, target_stats = [], [], []
    started = time.perf_counter()
    write_json(output / "status.json", dict(phase="running", completed=0, total=len(selected)))
    for index, row in enumerate(selected, 1):
        filename, prefix = row["filename"], f"{index:02d}"
        with Image.open(args.input_dir / filename) as im:
            gray = np.asarray(im).copy()
        _, probabilities = read_cache(caches[filename], gray)
        image = ((gray.astype(np.float32) / 255 - .5) / .5 + 1) / 2
        maps, metadata = clarity_components(image, probabilities)
        previous_path = args.previous_contours_dir / "maps" / f'{old_frames[filename]["prefix"]}_contours.npz'
        with np.load(previous_path, allow_pickle=False) as previous:
            np.testing.assert_array_equal(gray, previous["TIR_uint8"])
            for key in ("joint512", "selective_v1", "joint_importance", "importance_gain"):
                np.testing.assert_array_equal(maps[key], previous[key])
        old_mean, new_mean = (float(maps[k].mean(dtype=np.float64)) for k in ("selective_v1", "clarity_v1"))
        maps["uniform_dim"] = maps["selective_v1"] * (new_mean / old_mean if old_mean else 1.)
        if np.max(maps["uniform_dim"]) > 1:
            raise ValueError("Mean control would clip; use an explicit unclipped comparison")
        Image.fromarray(gray).save(output / "frames" / f"{prefix}_TIR.png")
        for key, values in maps.items():
            uint8_image(values).save(output / "frames" / f"{prefix}_{key}.png")
        red_overlay(gray, maps["clarity_v1"]).save(output / "frames" / f"{prefix}_clarity_v1_overlay.png")
        make_sheet(gray, maps, modes=MODES, labels=LABELS, importance_key="importance_gain",
                   importance_caption="Unchanged importance gain").save(output / "frames" / f"{prefix}_comparison.png")
        np.savez_compressed(output / "maps" / f"{prefix}_contours.npz", TIR_uint8=gray, **maps)
        for key in (*MODES, *PROBES, "uniform_dim"):
            frame_stats.append(dict(filename=filename, prefix=prefix, mode=key, **metadata, **response_stats(maps[key])))
        for target in targets:
            if target["filename"] != filename: continue
            make_sheet(gray, maps, crop=target["crop"], box=target["box"], zoom=args.zoom,
                       modes=MODES, labels=LABELS, importance_key="importance_gain",
                       importance_caption="Unchanged importance gain").save(output / "targets" / f'{target["id"]}.png')
            x0, y0, x1, y1 = target["box"]
            yy, xx = np.indices(gray.shape)
            mask = (xx+.5 >= x0) & (xx+.5 < x1) & (yy+.5 >= y0) & (yy+.5 < y1)
            for key in MODES:
                values = maps[key][mask]
                target_stats.append(dict(id=target["id"], filename=filename, category=target["category"], mode=key,
                    mean=float(values.mean()) if len(values) else None,
                    peak=float(values.max()) if len(values) else None,
                    pixels_ge_02=int((values >= .2).sum())))
        frames.append(dict(prefix=prefix, filename=filename, video_id=row["video_id"], group=row["group"]))
        write_json(output / "status.json", dict(phase="running", completed=index, total=len(selected)))
        print(f"[{index}/{len(selected)}] {prefix}: sigma={metadata['noise_sigma']:.5f}, old mean={old_mean:.4f}, new={new_mean:.4f}", flush=True)
    write_csv(output / "frame_statistics.csv", frame_stats)
    write_csv(output / "target_statistics.csv", target_stats)
    write_json(output / "preview_manifest.json", dict(format="contour_clarity_review_v1", identity=identity,
        frames=frames, targets=targets, seconds=time.perf_counter()-started,
        limits="Fixed development review, no boundary labels, no accuracy/generalization or RGB gain established"))
    write_page(output, frames, targets)
    write_json(output / "status.json", dict(phase="completed", completed=len(frames), targets=len(targets)))
    print(output / "index.html", flush=True)


if __name__ == "__main__": run(parse_args())
