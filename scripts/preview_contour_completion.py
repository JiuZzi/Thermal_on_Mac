#!/usr/bin/env python3
"""Four-arm completion review on the frozen trainB development images only."""

import argparse
import csv
from dataclasses import asdict
import hashlib
import html
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pytorch-CycleGAN-and-pix2pix"))

from models.clarity_semantic_edges import clarity_components, CLARITY_SETTINGS
from models.contour_completion import completion_comparison, CompletionSettings, MODES
from preview_semantics import DEFAULT_REVIEW, load_review, write_json
from preview_semantic_contours import DEFAULT_SOURCE, read_cache
from semantic_preview.checkpoint import sha256_file
from semantic_preview.review import uint8_image, red_overlay, write_csv


LABELS = dict(observed="0 Observed / no completion", closing="1 Morphological closing",
              geometry="2 Direction + observation", reliable="3 Same paths + reliability")
CHINESE = dict(observed="不补边", closing="简单闭运算", geometry="方向与观测约束补边",
               reliable="相同候选＋可靠性筛选")
DEFAULT_CLARITY = ROOT / "runs/edge_audit/contour_clarity_seed42_v1"
DEFAULT_OUTPUT = ROOT / "runs/edge_audit/contour_completion_seed42_v2"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clarity-dir", type=Path, default=DEFAULT_CLARITY)
    parser.add_argument("--semantic-preview-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "FLIR_datasets/trainB")
    parser.add_argument("--review-manifest", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--protocol-manifest", type=Path, default=ROOT / "FLIR_protocol_v2/manifest.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-images", type=int, default=40)
    parser.add_argument("--zoom", type=int, default=6)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("Preserve existing reviews; use a new output directory")
    if args.input_dir.resolve() != (ROOT / "FLIR_datasets/trainB").resolve():
        parser.error("Development trainB only; never select completion settings on testB")
    if args.num_images < 1 or not 1 <= args.zoom <= 12:
        parser.error("Positive image count and zoom 1..12 required")
    return args


def _image01(gray):
    # Match the original clarity and D conditioning arithmetic exactly.
    return ((gray.astype(np.float32) / 255 - .5) / .5 + 1) / 2


def write_table(path, rows, empty_columns):
    """Even an empty, legitimately rejected review has readable CSV headers."""
    with Path(path).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else empty_columns)
        writer.writeheader(); writer.writerows(rows)


def _cached_maps(path, gray):
    with np.load(path, allow_pickle=False) as archive:
        if not np.array_equal(archive["TIR_uint8"], gray):
            raise ValueError(f"Clarity cache image/coordinates changed: {path}")
        keys = ("clarity_v1", "clarity_v1_fine_ridges", "clarity_v1_coarse_ridges", "raw_fine_ridges")
        values = {key: archive[key].copy() for key in keys}
    for key, value in values.items():
        if value.shape != gray.shape or value.dtype != np.float32 or not np.isfinite(value).all() or (value < 0).any() or (value > 1).any():
            raise ValueError(f"Invalid cached clarity array: {path}/{key}")
        if key.endswith("ridges") and not np.isin(value, (0., 1.)).all():
            raise ValueError(f"Cached centerlines must be binary: {key}")
    ridges = np.logical_or.reduce([values[key] > 0 for key in keys[1:]])
    return values["clarity_v1"], ridges


def preflight(args):
    """Check ALL sources and replay E before creating any review output."""
    _, selected, targets = load_review(args)
    clarity = json.loads((args.clarity_dir / "preview_manifest.json").read_text(encoding="utf-8"))
    semantics = json.loads((args.semantic_preview_dir / "preview_manifest.json").read_text(encoding="utf-8"))
    for folder, manifest in ((args.clarity_dir, clarity), (args.semantic_preview_dir, semantics)):
        status = json.loads((folder / "status.json").read_text(encoding="utf-8"))
        if status.get("phase") != "completed" or status.get("completed") != len(manifest["frames"]):
            raise ValueError(f"Incomplete source review: {folder}")
    if clarity["identity"]["semantic_pack"] != semantics["pack_id"]:
        raise ValueError("Clarity and semantic caches have different provenance")
    if semantics["review_manifest_sha256"] != sha256_file(args.review_manifest) or semantics["protocol_manifest_sha256"] != sha256_file(args.protocol_manifest):
        raise ValueError("Source review/protocol identity differs")
    by_name = {}
    for row in clarity["frames"]:
        if row["filename"] in by_name or not str(row["prefix"]).isdigit():
            raise ValueError("Duplicate image or unsafe clarity prefix")
        by_name[row["filename"]] = row
    semantic_names = {}
    for row in semantics["frames"]:
        if row["filename"] in semantic_names or not str(row["prefix"]).isdigit():
            raise ValueError("Duplicate image or unsafe semantic prefix")
        semantic_names[row["filename"]] = row
    verified = []
    for index, row in enumerate(selected, 1):
        name = row["filename"]
        if name not in by_name or name not in semantic_names:
            raise ValueError(f"Missing fixed frame in cached review: {name}")
        with Image.open(args.input_dir / name) as image:
            gray = np.asarray(image).copy()
        old = by_name[name]
        path = args.clarity_dir / "maps" / f'{old["prefix"]}_contours.npz'
        semantic = args.semantic_preview_dir / "maps" / f'{semantic_names[name]["prefix"]}_responses.npz'
        semantic_digest = sha256_file(semantic)
        if clarity["identity"]["semantic_cache_sha256"].get(name) != semantic_digest:
            raise ValueError(f"Semantic floating cache hash changed: {name}")
        observed, _ = _cached_maps(path, gray)
        _, probabilities = read_cache(semantic, gray)
        replayed, _ = clarity_components(_image01(gray), probabilities, include_probes=False)
        np.testing.assert_array_equal(observed, replayed["clarity_v1"],
                                      err_msg=f"Original observed E no longer reproduces: {name}")
        # Candidate geometry must reproduce too, not just the final weighted E.
        with np.load(path, allow_pickle=False) as archive:
            for key in ("clarity_v1_fine_ridges", "clarity_v1_coarse_ridges", "raw_fine_ridges"):
                np.testing.assert_array_equal(archive[key], replayed[key], err_msg=key)
        verified.append(dict(**row, cache_prefix=old["prefix"], cache_sha256=sha256_file(path), tir_sha256=sha256_file(args.input_dir / name),
                             semantic_cache_sha256=semantic_digest))
        print(f"[CHECK {index:02d}/{len(selected)}] original E and candidate ridges reproduce", flush=True)
    return verified, targets


def comparison_sheet(gray, maps, *, crop=None, box=None, zoom=1, additions=False):
    x0, y0, x1, y1 = (0, 0, gray.shape[1], gray.shape[0]) if crop is None else crop
    local = gray[y0:y1, x0:x1]
    h, w = local.shape
    width, height = max(190, w * zoom), h * zoom + 28
    rows = 1 if additions else 2
    sheet = Image.new("RGB", (5 * width, rows * height), "white")
    draw = ImageDraw.Draw(sheet)
    for row in range(rows):
        panels = [("TIR / diagnostic box only", Image.fromarray(local).convert("RGB"))]
        for mode in MODES:
            if additions:
                mask = np.zeros_like(gray, bool) if mode == "observed" else maps[f"{mode}_mask"]
                base = np.repeat(local[..., None], 3, axis=2).astype(np.float32) / 255
                color = np.asarray((0., 1., 1.) if mode == "reliable" else (1., .7, 0.))
                values = mask[y0:y1, x0:x1, None].astype(np.float32)
                panel = uint8_image(base * (1 - .8 * values) + color * .8 * values)
            else:
                values = maps[mode][y0:y1, x0:x1]
                panel = uint8_image(values).convert("RGB") if row == 0 else red_overlay(local, values)
            panels.append((LABELS[mode], panel))
        for col, (caption, panel) in enumerate(panels):
            left = col * width + (width - w * zoom) // 2
            top = row * height + 25
            sheet.paste(panel.resize((w * zoom, h * zoom), Image.Resampling.NEAREST), (left, top))
            draw.text((col * width + 4, row * height + 5), caption, fill="black")
            if col == 0 and box is not None:
                bx0, by0, bx1, by1 = box
                draw.rectangle((left + (bx0-x0)*zoom, top + (by0-y0)*zoom,
                                left + (bx1-x0)*zoom, top + (by1-y0)*zoom), outline="yellow", width=1)
    return sheet


def write_candidate_svg(path, prefix, shape, metadata):
    h, w = shape
    pieces = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
              f'<image href="../frames/{prefix}_TIR.png" width="{w}" height="{h}"/>']
    for c in metadata["candidates"]:
        color = "#00ffff" if c["reliability_accepted"] else "#ffb000" if c["geometry_selected"] else "#999999"
        points = " ".join(f"{x+.5},{y+.5}" for y, x in c["path_yx"])
        description = html.escape(json.dumps(dict(id=c["id"], selected=c["geometry_selected"],
            accepted=c["reliability_accepted"], score=round(c["reliability_score"], 4),
            reasons=c["reliability_rejections"], geometry_rejection=c["geometry_rejection"],
            features=c["features"]), ensure_ascii=False))
        pieces.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width=".9"><title>{description}</title></polyline>')
    path.write_text("".join(pieces) + "</svg>", encoding="utf-8")


def write_page(output, frames, targets, identity):
    cards = []
    nav = ' '.join(f'<a href="#f{f["prefix"]}">{f["prefix"]}</a>' for f in frames)
    for frame in frames:
        prefix = frame["prefix"]
        figures = ''.join(f'<figure><figcaption>{html.escape(CHINESE[m])}</figcaption><a href="frames/{prefix}_{m}.png"><img loading="lazy" src="frames/{prefix}_{m}.png"></a></figure>' for m in MODES)
        crops = ''.join(f'<h4>{html.escape(t["id"])} / {html.escape(t["category"])}</h4>'
            f'<img loading="lazy" src="targets/{t["id"]}_comparison.png">'
            f'<details><summary>仅新增中心线定位（等亮度诊断）</summary><img loading="lazy" src="targets/{t["id"]}_added.png"></details>'
            for t in targets if t["filename"] == frame["filename"])
        cards.append(f'<article id="f{prefix}"><h2>{prefix} / {html.escape(frame["filename"])}</h2>'
            f'<p>几何选用 {frame["geometry_selected"]} 条；可靠性通过 {frame["reliability_accepted"]} 条。'
            f'中心线像素：闭运算 {frame["closing_pixels"]} / 几何 {frame["geometry_pixels"]} / 筛选 {frame["reliable_pixels"]}。这些是数量，不是准确率。</p>'
            f'<a href="frames/{prefix}_comparison.png"><img loading="lazy" src="frames/{prefix}_comparison.png"></a>'
            f'<details><summary>全图轮廓与新增中心线</summary><div class="grid">{figures}</div>'
            f'<img loading="lazy" src="frames/{prefix}_added.png"></details>'
            f'<details><summary>固定目标裁剪（仅定位，不参与补边）</summary>{crops}</details>'
            f'<details><summary>候选与拒绝原因：青色通过，橙色筛掉，灰色竞争落选；悬停查看</summary>'
            f'<object data="diagnostics/{prefix}_candidates.svg" type="image/svg+xml"></object>'
            f'<p><a href="diagnostics/{prefix}_candidates.json">完整路径与评分</a></p></details></article>')
    intro = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>局部轮廓补全：四组固定对照</title><style>
body{font:16px/1.65 system-ui,sans-serif;background:#f3f5f7;color:#192538;margin:18px}main{max-width:1800px;margin:auto}
article,.notice{background:white;padding:18px;border:1px solid #d3dae3;border-radius:8px;margin:18px 0}.notice{background:#fff3d6}
img{max-width:100%;height:auto}h2{overflow-wrap:anywhere}summary{cursor:pointer;color:#175a91;margin:8px 0}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}
figure{margin:0}figure img{width:100%}object{width:min(100%,1080px);aspect-ratio:360/288}nav{position:sticky;top:0;background:white;padding:8px;z-index:2}nav a{display:inline-block;padding:3px 6px}
</style><main><h1>局部轮廓补全：四组固定对照</h1><div class="notice">
<p>每组使用同一张原始TIR、同一clarity_v1、同一候选脊线。上排为固定[0,1]软轮廓，下排红色仅表示强度；不逐图提亮，软编码σ统一为0.5像素。密集新增中心线仍可能形成宽区域。</p>
<p>四组依次为：不补边、方形闭运算、方向与红外观测约束路径、相同已选路径再经可靠性拒绝与衰减。第四组不重新搜索路径，新增条件逐像素不超过第三组。</p>
<p>展开“新增中心线”可等亮度定位补在哪里；这些诊断颜色不代表实际条件强度。候选评分是固定规则代理，不是经过校准的边界正确概率；语义P/Q不作为硬边界真值。</p>
<p>全部是此前固定trainB开发图。框、类别、图号仅用于展示，不进入算法；包括无人车和复杂纹理场景，不强制每图补线。未接入diffusion，未启用第三通道C，未训练评分网络。</p>
<p>没有真实缺口边界标注，补线数量、闭合率、连通域或更亮的图均不能证明质量改善。先审查误连与漏补，再决定是否接入生成。</p>
<p><a href="frame_statistics.csv">数量与强度统计</a> · <a href="candidate_statistics.csv">所有路径评分</a> · <a href="candidate_labels_template.csv">人工复核模板</a> · <a href="preview_manifest.json">来源/设置</a></p></div>'''
    (output / "index.html").write_text(intro + f'<nav>{nav}</nav>' + ''.join(cards)
        + f'<p>Pack: {identity}</p></main></html>', encoding="utf-8")


def run(args):
    if args.output_dir.exists():
        raise FileExistsError("Preserve previous review output")
    selected, targets = preflight(args)
    settings = CompletionSettings()
    source_files = [Path(__file__), ROOT / "pytorch-CycleGAN-and-pix2pix/models/contour_completion.py",
                    ROOT / "pytorch-CycleGAN-and-pix2pix/models/clarity_semantic_edges.py",
                    ROOT / "pytorch-CycleGAN-and-pix2pix/models/semantic_edges.py",
                    ROOT / "pytorch-CycleGAN-and-pix2pix/models/selective_semantic_edges.py",
                    ROOT / "pytorch-CycleGAN-and-pix2pix/models/saliency_edges.py"]
    identity = dict(settings=asdict(settings), source_clarity_settings=CLARITY_SETTINGS,
        fixed_review_sha256=sha256_file(args.review_manifest), protocol_sha256=sha256_file(args.protocol_manifest),
        clarity_manifest_sha256=sha256_file(args.clarity_dir / "preview_manifest.json"),
        semantic_manifest_sha256=sha256_file(args.semantic_preview_dir / "preview_manifest.json"),
        source_files={str(p.relative_to(ROOT)): sha256_file(p) for p in source_files}, selected=selected)
    pack_id = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    for name in ("frames", "maps", "targets", "diagnostics"):
        (output / name).mkdir()
    started = time.perf_counter()
    frames, frame_rows, candidate_rows, label_rows = [], [], [], []
    completed = 0
    write_json(output / "status.json", dict(phase="running", completed=0, total=len(selected)))
    try:
        for index, row in enumerate(selected, 1):
            tick = time.perf_counter()
            prefix, filename = f"{index:02d}", row["filename"]
            if sha256_file(args.input_dir / filename) != row["tir_sha256"]:
                raise ValueError("Source TIR changed after preflight")
            with Image.open(args.input_dir / filename) as image:
                gray = np.asarray(image).copy()
            source_path = args.clarity_dir / "maps" / f'{row["cache_prefix"]}_contours.npz'
            if sha256_file(source_path) != row["cache_sha256"]:
                raise ValueError("Clarity cache changed after preflight")
            observed, ridges = _cached_maps(source_path, gray)
            maps, metadata = completion_comparison(_image01(gray), observed, trace_ridges=ridges, settings=settings)
            np.testing.assert_array_equal(observed, maps["observed"])
            if np.any(maps["reliable"] > maps["geometry"]):
                raise AssertionError("Reliability arm may only reject/attenuate the same paths")
            Image.fromarray(gray).save(output / "frames" / f"{prefix}_TIR.png")
            for mode in MODES:
                uint8_image(maps[mode]).save(output / "frames" / f"{prefix}_{mode}.png")
            comparison_sheet(gray, maps).save(output / "frames" / f"{prefix}_comparison.png")
            comparison_sheet(gray, maps, additions=True).save(output / "frames" / f"{prefix}_added.png")
            np.savez_compressed(output / "maps" / f"{prefix}_completion.npz", TIR_uint8=gray, **maps)
            write_json(output / "diagnostics" / f"{prefix}_candidates.json", metadata)
            write_candidate_svg(output / "diagnostics" / f"{prefix}_candidates.svg", prefix, gray.shape, metadata)
            for target in targets:
                if target["filename"] != filename: continue
                for added, suffix in ((False, "comparison"), (True, "added")):
                    comparison_sheet(gray, maps, crop=target["crop"], box=target["box"], zoom=args.zoom,
                                     additions=added).save(output / "targets" / f'{target["id"]}_{suffix}.png')
            counts = {mode: int(maps[f"{mode}_mask"].sum()) for mode in MODES[1:]}
            for mode in MODES:
                frame_rows.append(dict(prefix=prefix, filename=filename, mode=mode,
                    mean_strength=float(maps[mode].mean(dtype=np.float64)),
                    added_centerline_pixels=counts.get(mode, 0),
                    raised_pixels=int(np.count_nonzero(maps[mode] > observed)),
                    added_strength=float((maps[mode] - observed).sum(dtype=np.float64))))
            for c in metadata["candidates"]:
                candidate_rows.append(dict(prefix=prefix, filename=filename, candidate_id=c["id"],
                    gap=c["gap"], length=c["length"], cost=c["cost"], strength=c["strength"],
                    geometry_selected=c["geometry_selected"], reliability_accepted=c["reliability_accepted"],
                    reliability_score=c["reliability_score"], geometry_rejection=c["geometry_rejection"],
                    reliability_rejections="|".join(c["reliability_rejections"]), **c["features"]))
                if c["geometry_selected"]:
                    label_rows.append(dict(prefix=prefix, filename=filename, candidate_id=c["id"],
                        boundary_label="", notes=""))
            frame = dict(prefix=prefix, filename=filename, video_id=row["video_id"], group=row["group"],
                geometry_selected=metadata["geometry_selected"], reliability_accepted=metadata["reliability_accepted"],
                closing_pixels=counts["closing"], geometry_pixels=counts["geometry"], reliable_pixels=counts["reliable"],
                eligible_pairs=metadata["eligible_pairs"], truncated_pairs=metadata["truncated_pairs"],
                seconds=time.perf_counter()-tick)
            frames.append(frame); completed = index
            write_json(output / "status.json", dict(phase="running", completed=completed, total=len(selected)))
            print(f"[FRAME {index:02d}/{len(selected)}] {prefix} | geometry={metadata['geometry_selected']} "
                  f"reliable={metadata['reliability_accepted']} | {frame['seconds']:.2f}s", flush=True)
        write_csv(output / "frame_statistics.csv", frame_rows)
        write_table(output / "candidate_statistics.csv", candidate_rows,
                    ("prefix", "filename", "candidate_id", "geometry_selected", "reliability_accepted", "reliability_score"))
        write_table(output / "candidate_labels_template.csv", label_rows,
                    ("prefix", "filename", "candidate_id", "boundary_label", "notes"))
        write_json(output / "preview_manifest.json", dict(format="contour_completion_review_v1", pack_id=pack_id,
            identity=identity, frames=frames, targets=targets, seconds=time.perf_counter()-started,
            annotation_policy="boundary_label: visible_boundary / wrong_connection / uncertain; blank means unreviewed",
            limits=["Development preview only; no true boundary accuracy, generalization or RGB benefit established",
                    "Same selected geometry paths in groups 2/3; reliability is a fixed uncalibrated proxy",
                    "Semantic hard veto, learned scorer, explicit structural C channel and diffusion integration are disabled"]))
        write_page(output, frames, targets, pack_id)
        write_json(output / "status.json", dict(phase="completed", completed=completed, total=len(selected),
                                               targets=len(targets), seconds=time.perf_counter()-started))
    except BaseException as error:
        try:
            write_json(output / "status.json", dict(phase="failed", completed=completed, total=len(selected),
                                                   error=f"{type(error).__name__}: {error}"))
        except OSError:
            pass  # A full disk must not hide the original processing exception.
        raise
    print(output / "index.html", flush=True)


if __name__ == "__main__":
    run(parse_args())
