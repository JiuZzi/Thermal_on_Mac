#!/usr/bin/env python3
"""Fixed development review of contour selection and semantic guidance coverage."""

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

from models.selective_semantic_edges import SELECTIVE_SETTINGS, selective_components
from models.learned_color_prior import semantic_reliability, LearnedChromaPrior
from models.guidance_diagnostics import guidance_report
from models.semantic_contour import FrozenSemanticPrior
from preview_semantic_contours import DEFAULT_SOURCE, preflight, read_cache, make_sheet
from preview_semantics import DEFAULT_REVIEW, write_json
from semantic_preview.checkpoint import sha256_file
from semantic_preview.inference import CLASS_NAMES
from semantic_preview.review import uint8_image, red_overlay, write_csv


MODES = ("joint512", "selective_v1", "uniform_dim")
LABELS = dict(joint512="Original joint512", selective_v1="Selective candidate", uniform_dim="Same-mean dimming")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--semantic-preview-dir", type=Path, default=DEFAULT_SOURCE)
    p.add_argument("--previous-contours-dir", type=Path, default=ROOT / "runs/edge_audit/semantic512_structure_joint_seed42_v2")
    p.add_argument("--input-dir", type=Path, default=ROOT / "FLIR_datasets/trainB")
    p.add_argument("--review-manifest", type=Path, default=DEFAULT_REVIEW)
    p.add_argument("--protocol-manifest", type=Path, default=ROOT / "FLIR_protocol_v2/manifest.csv")
    p.add_argument("--output-dir", type=Path, default=ROOT / "runs/edge_audit/selective_contours_seed42_floor_v1")
    p.add_argument("--num-images", type=int, default=40)
    p.add_argument("--zoom", type=int, default=6)
    p.add_argument("--flip-checkpoint", type=Path,
                   help="Same frozen semantic checkpoint: compute actual flip reliability; otherwise report a Q upper bound")
    p.add_argument("--prior", type=Path, help="Optional actual fitted RGB bank; absent means no claim about color-loss coverage")
    p.add_argument("--min-quality", type=float, default=.5)
    p.add_argument("--cpu-threads", type=int, default=2)
    args = p.parse_args(argv)
    if args.output_dir.exists():
        p.error("Use a new output directory; previous reviews are preserved")
    if args.input_dir.resolve() != (ROOT / "FLIR_datasets/trainB").resolve():
        p.error("Only the fixed trainB development review may select this variant")
    if args.num_images < 1 or not 1 <= args.zoom <= 12 or not 0 < args.min_quality <= 1 or args.cpu_threads < 1:
        p.error("Invalid image count, zoom, quality threshold or thread count")
    return args


def write_page(output, frames, targets, diagnostics_mode):
    captions = dict(TIR="原始红外", joint512="旧 joint512", gain_only="仅增加重要性整体增益",
                    ridge_only="仅整合重复响应", selective_v1="新候选：两项组合", uniform_dim="旧版同均值变暗对照",
                    importance_gain="重要性整体增益（不是置信度）", semantic_quality="语义可靠度 Q（不是正确率）",
                    selective_v1_overlay="新候选红色叠加")
    cards = []
    for frame in frames:
        n = frame["prefix"]
        figures = ''.join(f'<figure><figcaption>{html.escape(label)}</figcaption><a href="frames/{n}_{key}.png"><img loading="lazy" src="frames/{n}_{key}.png"></a></figure>'
                          for key, label in captions.items())
        crops = ''.join(f'<h4>{html.escape(t["id"])} / {html.escape(t["category"])}</h4><img class="crop" loading="lazy" src="targets/{t["id"]}.png">'
                        for t in targets if t["filename"] == frame["filename"])
        cards.append(f'<article id="f{n}"><h2>{n} / {html.escape(frame["filename"])}</h2><div class="grid">{figures}</div><details><summary>固定目标裁剪：红外、旧版、新候选、同均值变暗、整体增益</summary>{crops}</details></article>')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>结构选择候选与语义覆盖率诊断</title><style>body{font:16px/1.6 system-ui;background:#f5f6f8;color:#18212c;margin:20px}main{max-width:1400px;margin:auto}article,.notice{background:white;padding:18px;margin:18px 0;border:1px solid #ccd4de;border-radius:8px}.notice{background:#fff4d7}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px}figure{margin:0}figure img{width:100%}.crop{max-width:100%;image-rendering:pixelated}h2{overflow-wrap:anywhere}</style><main><h1>结构选择候选与语义覆盖率诊断</h1>
<div class="notice"><p>同一批固定训练开发图和冻结语义；不训练 GAN、不补边、不预测结构置信度。旧 joint512 保留为训练默认值。</p>
<p>所有轮廓统一按原始 [0,1] 显示，不逐图拉伸对比度。整体变暗对照与新候选具有相同全图平均强度，仅用于诊断选择性。</p>
<p>重要性负责贡献取舍；NMS 作用在跨尺度合并响应上，再按原宽度软化。前处理保持关闭。原始候选仍保存在 NPZ 中。</p>
<p>最终输出保留原始未整合细候选的 25% 强度下限；语义漏检不能再将这个原有保底乘暗。它不保证弱边全部正确或保持旧版的完整强度。</p>
<p>框只定位既有审查对象，不参与算法。框内强度和语义区域统计均不是边界准确率、召回率或着色收益。</p>'''
    links = '<p><a href="guidance_summary.json">语义覆盖率汇总</a> · <a href="guidance_by_class.csv">逐图逐类诊断</a> · <a href="frame_statistics.csv">全图描述统计</a> · <a href="target_statistics.csv">固定框描述统计</a></p>'
    page += f'<p>可靠度诊断模式：{html.escape(diagnostics_mode)}。</p>{links}</div>' + ''.join(cards) + '</main></html>'
    (output / "index.html").write_text(page, encoding="utf-8")


def run(args):
    torch.set_num_threads(args.cpu_threads)
    semantic = FrozenSemanticPrior(args.flip_checkpoint) if args.flip_checkpoint else None
    source_header = json.loads((args.semantic_preview_dir / "preview_manifest.json").read_text())
    if semantic and semantic.info["sha256"] != source_header["checkpoint"]["sha256"]:
        raise ValueError("Flip checkpoint differs from the cached semantic model")
    replay_count = 0
    def replay(gray):
        nonlocal replay_count
        value = semantic.predict(gray.astype(np.float32) / 255, "cpu")
        replay_count += 1
        print(f"Semantic cache replay [{replay_count}/{args.num_images}]", flush=True)
        return value
    source, selected, targets, cache_paths, cache_hashes = preflight(
        args, replay_probabilities=replay if semantic else None)
    old = json.loads((args.previous_contours_dir / "preview_manifest.json").read_text())
    old_frames = {f["filename"]: f for f in old["frames"]}
    if old["identity"]["source_semantic_pack"] != source["pack_id"]:
        raise ValueError("Legacy contours came from different semantic scores")
    bank = LearnedChromaPrior(args.prior, CLASS_NAMES) if args.prior else None
    supported = None if bank is None else [name for name, flag in zip(CLASS_NAMES, bank.supported.tolist()) if flag]
    diagnostic_mode = ("actual original/flip consistency" if semantic else
                       "cached single-view upper bound; online flip reliability can only reduce Q/coverage")
    if bank is None:
        diagnostic_mode += "; no fitted RGB bank supplied, color-loss coverage is not asserted"
    identity = dict(settings=SELECTIVE_SETTINGS, semantic_pack=source["pack_id"], cache_sha256=cache_hashes,
                    cache_revalidation=source.get("cache_revalidation"),
                    flip_checkpoint=None if semantic is None else semantic.info["sha256"],
                    prior_sha256=None if bank is None else bank.metadata["artifact_sha256"],
                    min_quality=args.min_quality,
                    code_sha256={str(p.relative_to(ROOT)): sha256_file(p) for p in [Path(__file__),
                        ROOT/"pytorch-CycleGAN-and-pix2pix/models/selective_semantic_edges.py",
                        ROOT/"pytorch-CycleGAN-and-pix2pix/models/semantic_edges.py",
                        ROOT/"scripts/preview_semantic_contours.py",
                        ROOT/"pytorch-CycleGAN-and-pix2pix/models/guidance_diagnostics.py"]})
    output = args.output_dir
    for name in ("frames", "maps", "targets", "diagnostics"):
        (output / name).mkdir(parents=True, exist_ok=False)
    frames, frame_stats, target_stats, coverage_rows = [], [], [], []
    totals = {name: dict(pixels=0, quality_sum=0., threshold_pixels=0, boundary_pixels=0, boundary_nonzero_pixels=0)
              for name in CLASS_NAMES}
    started = time.perf_counter()
    write_json(output / "status.json", dict(phase="running", completed=0, total=len(selected)))
    for number, row in enumerate(selected, 1):
        name, prefix = row["filename"], f"{number:02d}"
        with Image.open(args.input_dir / name) as im:
            gray = np.asarray(im).copy()
        _, probabilities = read_cache(cache_paths[name], gray)
        # Match legacy online normalization exactly for the pixelwise control.
        image = ((gray.astype(np.float32) / 255 - .5) / .5 + 1) / 2
        maps = selective_components(image, probabilities)
        old_file = args.previous_contours_dir / "maps" / f'{old_frames[name]["prefix"]}_contours.npz'
        with np.load(old_file, allow_pickle=False) as previous:
            np.testing.assert_array_equal(previous["TIR_uint8"], gray)
            np.testing.assert_array_equal(previous["joint512"], maps["joint512"])
        if np.any(maps["selective_v1"] > maps["joint512"] + 2e-7):
            raise RuntimeError("Selection invented/amplified a legacy response")
        before, after = float(maps["joint512"].mean(dtype=np.float64)), float(maps["selective_v1"].mean(dtype=np.float64))
        maps["uniform_dim"] = maps["joint512"] * (after / before if before else 1.)
        p = torch.from_numpy(probabilities)[None]
        second = None
        if semantic:
            if number == 1:
                np.testing.assert_allclose(semantic.predict(image, "cpu"), probabilities, atol=2e-5, rtol=1e-4)
            flipped = semantic.predict(np.fliplr(image).copy(), "cpu")[:, :, ::-1].copy()
            second = torch.from_numpy(flipped)[None]
        quality = semantic_reliability(p, second, boundary_radius=1)
        gate = None if bank is None else bank(p, quality)["strength"]
        report = guidance_report(p, quality, CLASS_NAMES, supported, args.min_quality,
                                 feature_gate=gate, scope=diagnostic_mode)
        report.update(filename=name, prefix=prefix)
        write_json(output / "diagnostics" / f"{prefix}_guidance.json", report)
        maps["semantic_quality"] = quality[0, 0].numpy()
        for category, values in report["classes"].items():
            coverage_rows.append(dict(filename=name, category=category, **values))
            for key in totals[category]:
                totals[category][key] += values[key]
        Image.fromarray(gray).save(output / "frames" / f"{prefix}_TIR.png")
        for key, values in maps.items():
            uint8_image(values).save(output / "frames" / f"{prefix}_{key}.png")
        red_overlay(gray, maps["selective_v1"]).save(output / "frames" / f"{prefix}_selective_v1_overlay.png")
        make_sheet(gray, maps, modes=MODES, labels=LABELS,
                   importance_key="importance_gain", importance_caption="Overall importance gain").save(
                       output / "frames" / f"{prefix}_comparison.png")
        np.savez_compressed(output / "maps" / f"{prefix}_contours.npz", TIR_uint8=gray, **maps)
        frames.append(dict(prefix=prefix, filename=name, video_id=row["video_id"], group=row["group"]))
        frame_stats.append(dict(filename=name, **{key: float(maps[key].mean(dtype=np.float64))
                                                 for key in (*MODES, "gain_only", "ridge_only")}))
        for target in targets:
            if target["filename"] != name:
                continue
            make_sheet(gray, maps, crop=target["crop"], box=target["box"], zoom=args.zoom,
                       modes=MODES, labels=LABELS, importance_key="importance_gain",
                       importance_caption="Overall importance gain").save(output / "targets" / f'{target["id"]}.png')
            x0, y0, x1, y1 = target["box"]
            yy, xx = np.indices(gray.shape)
            mask = (xx + .5 >= x0) & (xx + .5 < x1) & (yy + .5 >= y0) & (yy + .5 < y1)
            target_stats.append(dict(id=target["id"], filename=name, category=target["category"],
                                     **{key: float(maps[key][mask].mean()) if mask.any() else None for key in MODES}))
        write_json(output / "status.json", dict(phase="running", completed=number, total=len(selected)))
        print(f"[{number}/{len(selected)}] {prefix}: old={before:.4f}, candidate={after:.4f}, mean Q={report['mean_quality']:.4f}", flush=True)
    for values in totals.values():
        count = values["pixels"]
        values.update(mean_quality=values["quality_sum"] / max(count, 1),
                      threshold_fraction=values["threshold_pixels"] / max(count, 1),
                      boundary_fraction=values["boundary_pixels"] / max(count, 1))
    summary = dict(mode=diagnostic_mode, images=len(frames), classes=totals,
                   limits="Coverage is not correctness. Target-box edge strength is not boundary recall/accuracy.")
    write_json(output / "guidance_summary.json", summary)
    for filename, rows in (("guidance_by_class.csv", coverage_rows), ("frame_statistics.csv", frame_stats),
                           ("target_statistics.csv", target_stats)):
        write_csv(output / filename, rows)
    write_json(output / "preview_manifest.json", dict(format="selective_contour_review_v1", identity=identity,
               frames=frames, targets=targets, diagnostic_mode=diagnostic_mode, seconds=time.perf_counter()-started))
    write_page(output, frames, targets, diagnostic_mode)
    write_json(output / "status.json", dict(phase="completed", completed=len(frames), targets=len(target_stats)))
    print(output / "index.html", flush=True)


if __name__ == "__main__":
    run(parse_args())
