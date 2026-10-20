#!/usr/bin/env python3
"""Audit semantic/color conditions on the original fixed 40 trainB images.

All colored panels are class labels or prior swatches, not generated photos.
Reuse the verified 512 floating-point cache; no new pretrained model or GAN.
"""

import argparse
import hashlib
import html
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pytorch-CycleGAN-and-pix2pix"))

from models.semantic_color_prior import SemanticColorPrior, color_config
from models.semantic_edges import FMB_CLASS_NAMES, make_structure_semantic_condition
from preview_semantic_contours import DEFAULT_SOURCE, read_cache
from preview_semantics import DEFAULT_REVIEW, load_review, write_json
from semantic_preview.checkpoint import sha256_file
from semantic_preview.inference import PALETTE
from semantic_preview.review import write_csv


DEFAULT_OUTPUT = ROOT / "runs/edge_audit/semantic_color_prior_seed42_v1"
PANELS = (("tir", "TIR"), ("labels", "Semantic classes / display palette"),
          ("vegetation", "P(vegetation)"), ("sky", "P(sky)"),
          ("edge", "Existing joint512"), ("prior", "Color hints / NOT generated RGB"),
          ("margin", "Top1 - Top2 / NOT edge confidence"), ("strength", "Color prior strength"))


def panel_image(array):
    if array.dtype != np.uint8:
        array = np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(array).convert("RGB")


def sheet(panels, names=PANELS, columns=4):
    width, height = panels["tir"].size
    result = Image.new("RGB", (width * columns, (height + 26) * ((len(names) + columns - 1) // columns)), "white")
    draw = ImageDraw.Draw(result)
    for index, (key, caption) in enumerate(names):
        left, top = (index % columns) * width, (index // columns) * (height + 26)
        result.paste(panels[key], (left, top + 26))
        draw.text((left + 5, top + 6), caption, fill="black")
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic-preview-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "FLIR_datasets/trainB")
    parser.add_argument("--review-manifest", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--protocol-manifest", type=Path, default=ROOT / "FLIR_protocol_v2/manifest.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-images", type=int, default=40)
    args = parser.parse_args(argv)
    if args.num_images < 1 or args.input_dir.resolve() != (ROOT / "FLIR_datasets/trainB").resolve():
        parser.error("Use a positive count from the fixed trainB development review")
    if args.output_dir.exists():
        parser.error("Choose a new output directory; existing reviews must be preserved")
    return args


def cached_preflight(args):
    """Validate immutable cached data against the already reviewed joint pack.

    This is not a fresh inference run. Loading/inference source changed since
    the old cache was produced; record that drift instead of pretending to
    reproduce old code. Float cache hashes are anchored in the joint pack.
    """
    review, selected, _ = load_review(args)
    source_path = args.semantic_preview_dir / "preview_manifest.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    status = json.loads((args.semantic_preview_dir / "status.json").read_text(encoding="utf-8"))
    joint_path = ROOT / "runs/edge_audit/semantic512_structure_joint_seed42_v2/preview_manifest.json"
    joint = json.loads(joint_path.read_text(encoding="utf-8"))
    identity = joint["identity"]
    if status.get("phase") != "completed" or status.get("completed") != len(source["frames"]):
        raise ValueError("Incomplete cached semantic preview")
    if source["settings"]["short_side"] != 512 or source["settings"].get("baseline_flip_tta") is not False:
        raise ValueError("Expected the author512 baseline cache")
    if (identity["source_semantic_pack"] != source["pack_id"]
            or identity["source_manifest_sha256"] != sha256_file(source_path)
            or source["review_manifest_sha256"] != sha256_file(args.review_manifest)
            or source["protocol_manifest_sha256"] != sha256_file(args.protocol_manifest)):
        raise ValueError("Cached source/review/protocol identity mismatch")
    weight = ROOT / "runs/pretrained/segformer_fmb_b3/best_model.pth"
    if sha256_file(weight) != source["checkpoint"]["sha256"]:
        raise ValueError("Current frozen semantic weight differs from the cached source")
    source_frames = {row["filename"]: row for row in source["frames"]}
    paths, hashes = {}, {}
    for row in selected:
        name = row["filename"]
        frame = source_frames[name]
        if frame["sha256"] != review["source_hashes"][name] or not frame["prefix"].isdigit():
            raise ValueError("Cached source frame identity mismatch")
        path = args.semantic_preview_dir / "maps" / f'{frame["prefix"]}_responses.npz'
        digest = sha256_file(path)
        if digest != identity["semantic_cache_sha256"].get(name):
            raise ValueError(f"Cached floating-point semantics changed: {name}")
        paths[name], hashes[name] = path, digest
    drift = {}
    for relative, original in source["code_hashes"].items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ValueError("Unsafe or missing original source path")
        current = sha256_file(path)
        if current != original:
            drift[relative] = dict(original=original, current=current)
    provenance = dict(mode="reuse unchanged cached floats; not old-source inference reproduction",
                      joint_manifest_sha256=sha256_file(joint_path),
                      semantic_checkpoint_sha256=source["checkpoint"]["sha256"], source_code_drift=drift)
    return source, selected, paths, hashes, provenance


def run(args):
    torch.set_num_threads(1)
    source, selected, caches, hashes, provenance = cached_preflight(args)
    output = args.output_dir
    (output / "frames").mkdir(parents=True, exist_ok=False)
    (output / "maps").mkdir()
    prior = SemanticColorPrior().eval()
    rows, frames, overview_rows = [], [], []
    write_json(output / "status.json", dict(phase="running", completed=0, total=len(selected)))
    old_contours = ROOT / "runs/edge_audit/semantic512_structure_joint_seed42_v2/maps"
    old_controls_checked = 0
    try:
        with torch.inference_mode():
            for index, row in enumerate(selected, 1):
                name, prefix = row["filename"], f"{index:02d}"
                with Image.open(args.input_dir / name) as image:
                    gray = np.asarray(image).copy()
                _, p = read_cache(caches[name], gray)
                color = prior(torch.from_numpy(p)[None])
                # Match the original normalized-TIR arithmetic exactly.
                image = np.clip(((gray.astype(np.float32) / 255 - .5) / .5 + 1) / 2, 0, 1)
                edge = make_structure_semantic_condition(image, p)
                old = old_contours / f"{prefix}_contours.npz"
                if old.is_file():
                    with np.load(old, allow_pickle=False) as archive:
                        np.testing.assert_array_equal(archive["TIR_uint8"], gray)
                        np.testing.assert_array_equal(archive["joint512"], edge)
                    old_controls_checked += 1
                labels = color["labels"][0].numpy().astype(np.uint8)
                panels = dict(tir=panel_image(gray), labels=panel_image(PALETTE[labels]),
                              vegetation=panel_image(p[6]), sky=panel_image(p[7]), edge=panel_image(edge),
                              prior=panel_image(color["hint_rgb"][0].permute(1, 2, 0).numpy()),
                              margin=panel_image(color["margin"][0, 0].numpy()),
                              strength=panel_image(color["strength"][0, 0].numpy()))
                files = {}
                for key, panel in panels.items():
                    relative = f"frames/{prefix}_{key}.png"
                    panel.save(output / relative)
                    files[key] = relative
                comparison = f"frames/{prefix}_conditions.jpg"
                sheet(panels).save(output / comparison, quality=92)
                if index in (1, 2, 31, 36):
                    overview_rows.append(sheet(panels, (("tir", "TIR"), ("labels", "Semantic labels"),
                                                      ("prior", "Prior / NOT generated RGB"), ("strength", "Prior strength"))))
                np.savez_compressed(output / "maps" / f"{prefix}_prior.npz", TIR_uint8=gray,
                                    labels=labels, joint512=edge, color_features=color["features"][0].numpy(),
                                    margin=color["margin"][0, 0].numpy(), strength=color["strength"][0, 0].numpy())
                stats = dict(prefix=prefix, filename=name,
                             mean_color_strength=float(color["strength"].mean()),
                             mean_semantic_margin=float(color["margin"].mean()))
                stats.update({f"predicted_{category}_fraction": float((labels == c).mean())
                              for c, category in enumerate(FMB_CLASS_NAMES)})
                rows.append(stats)
                frames.append(dict(prefix=prefix, filename=name, files=files, comparison=comparison,
                                   source_cache_sha256=hashes[name]))
                write_json(output / "status.json", dict(phase="running", completed=index, total=len(selected)))
                print(f"{index}/{len(selected)} {name}", flush=True)
        if overview_rows:
            overview = Image.new("RGB", (overview_rows[0].width, sum(row.height for row in overview_rows)), "white")
            top = 0
            for row in overview_rows:
                overview.paste(row, (0, top))
                top += row.height
            overview.save(output / "overview.jpg", quality=92)
        write_csv(output / "descriptive_statistics.csv", rows)
        identity = dict(source_pack_id=source["pack_id"], cache_sha256=hashes, prior=color_config(), provenance=provenance,
                        code_sha256={str(p.relative_to(ROOT)): sha256_file(p) for p in
                                     (Path(__file__), ROOT / "pytorch-CycleGAN-and-pix2pix/models/semantic_color_prior.py")})
        write_json(output / "preview_manifest.json", dict(format="semantic_color_prior_preview_v1",
                   identity=identity, frames=frames, pack_id=hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
                   unchanged_joint512_frames=old_controls_checked,
                   evidence="condition inspection only; no color GAN training or quality claim"))
        cards = []
        for frame in frames:
            cards.append(f'<article><h2>{frame["prefix"]} / {html.escape(frame["filename"])}</h2>'
                         f'<a href="{frame["comparison"]}"><img loading="lazy" src="{frame["comparison"]}"></a></article>')
        page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>语义与颜色先验：开发集条件预览</title><style>body{font:16px/1.6 system-ui;margin:24px;background:#f5f6f8;color:#18212c}main{max-width:1500px;margin:auto}article,.note{background:white;padding:16px;margin:18px 0;border:1px solid #ccd4de;border-radius:8px}img{width:100%;height:auto}h2{font-size:17px;overflow-wrap:anywhere}.note{background:#fff4d7}</style><main>
<h1>语义与颜色先验：开发集条件预览</h1><div class="note"><p>这些是输入条件，不是新模型生成的 RGB。固定 trainB 开发图；复用作者512语义，不训练或微调语义模型。</p>
<p>上行：红外、语义标签（仅显示配色）、植被响应、天空响应。下行：原 joint512、颜色提示、类别分离度、颜色约束强度。</p>
<p>颜色提示来自明确的人为颜色范围，只覆盖天空／植被／道路／人行道，其他类为灰；颜色强度低时逐渐变灰。类标签显示配色与颜色提示不是一回事。</p>
<p>分离度不是边缘置信度，也无法排除高分误分类。请检查树叶间的天空、小目标和语义外溢；当前统计只有描述意义，不能当作准确率。</p></div>'''
        (output / "index.html").write_text(page + "".join(cards) + "</main></html>", encoding="utf-8")
        write_json(output / "status.json", dict(phase="completed", completed=len(frames), total=len(frames),
                                               unchanged_joint512_frames=old_controls_checked))
        print(f"Preview: {output / 'index.html'}", flush=True)
    except Exception as error:
        write_json(output / "status.json", dict(phase="failed", completed=len(frames), total=len(selected), error=str(error)))
        raise


if __name__ == "__main__":
    run(parse_args())
