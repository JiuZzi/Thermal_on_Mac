#!/usr/bin/env python3
"""Audit the existing FMB weight on visible images, without learning colors.

prepare -> annotate fixed points from RGB alone -> infer -> evaluate.
Direct RGB and visible-luminance replication are two out-of-domain probes.
The production thermal preprocessing and training paths are never modified.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import html
import json
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion
import torch
from torch.nn import functional as F

from semantic_preview.checkpoint import load_checkpoint, sha256_file
from semantic_preview.inference import CLASS_NAMES, CLASS_LABELS, PALETTE, MEAN, STD, predict_probabilities


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "runs/semantic_audit/rgb_fmb_seed42_v1"
TARGETS = {"road": 1, "sidewalk": 2, "vegetation": 6, "sky": 7}
GROUPS = (*TARGETS, "other")
MODES = ("rgb", "gray")
POINTS = [(x, y) for y in (60, 144, 240) for x in (60, 180, 300)]
CORE_SETTINGS = dict(min_probability=.9, min_margin=.2, erosion_pixels=2)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def select_records(records, count=32, seed=42):
    """Round-robin scenes, one image per video, using metadata before prediction."""
    eligible = [row for row in records if row["destination_split"] == "trainA"]
    videos = defaultdict(list)
    for row in eligible:
        videos[row["video_id"]].append(row)
    if not 1 <= count <= len(videos):
        raise ValueError("Request 1..number of training RGB videos")
    rng = random.Random(seed)
    scenes = defaultdict(list)
    for video in sorted(videos):
        row = rng.choice(sorted(videos[video], key=lambda r: r["destination_name"]))
        scenes[row["scene"] or "unknown"].append(row)
    for scene in sorted(scenes):
        rng.shuffle(scenes[scene])
    selected = []
    while len(selected) < count:
        for scene in sorted(scenes):
            if scenes[scene] and len(selected) < count:
                selected.append(scenes[scene].pop())
    return selected


def source_image(row):
    name = row["filename"]
    if Path(name).name != name or "/" in name or "\\" in name:
        raise ValueError("Expected a source basename")
    path = ROOT / "FLIR_datasets/trainA" / name
    if sha256_file(path) != row["source_sha256"]:
        raise ValueError(f"Source changed: {name}")
    with Image.open(path) as image:
        if image.mode != "RGB" or image.size != (360, 288):
            raise ValueError("Expected original processed RGB 360x288")
        return np.asarray(image).copy()


def contact_sheet(panels, columns=2):
    width, height = panels[0][1].size
    result = Image.new("RGB", (width * columns, (height + 25) * ((len(panels) + columns - 1) // columns)), "white")
    draw = ImageDraw.Draw(result)
    for index, (caption, panel) in enumerate(panels):
        left, top = index % columns * width, index // columns * (height + 25)
        draw.text((left + 4, top + 5), caption, fill="black")
        result.paste(panel, (left, top + 25))
    return result


def prepare(args):
    output = args.output_dir
    if output.exists():
        raise ValueError("Choose a new audit folder; existing annotations are preserved")
    protocol = ROOT / "FLIR_protocol_v2/manifest.csv"
    with protocol.open(newline="", encoding="utf-8-sig") as handle:
        selected = select_records(list(csv.DictReader(handle)), args.num_images)
    rows = []
    for index, source in enumerate(selected, 1):
        name = source["destination_name"]
        row = dict(prefix=f"{index:02d}", filename=name, video_id=source["video_id"],
                   scene=source["scene"] or "unknown",
                   source_sha256=sha256_file(ROOT / "FLIR_datasets/trainA" / name))
        source_image(row)
        rows.append(row)
    (output / "raw").mkdir(parents=True)
    annotations, panels = [], []
    for row in rows:
        rgb = source_image(row)
        image = Image.fromarray(rgb)
        image.save(output / "raw" / f'{row["prefix"]}_rgb.png')
        marked = image.copy()
        draw = ImageDraw.Draw(marked)
        for index, (x, y) in enumerate(POINTS, 1):
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline="black", width=3)
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), outline="white", width=1)
            draw.text((x + 7, y - 8), str(index), fill="white", stroke_width=1, stroke_fill="black")
            annotations.append(dict(id=f'{row["prefix"]}_P{index}', prefix=row["prefix"], x=x, y=y,
                                    expected=None, note=""))
        marked.save(output / "raw" / f'{row["prefix"]}_points.png')
        panels.append((f'{row["prefix"]} / {row["scene"]}', marked))
    for first in range(0, len(panels), 4):
        contact_sheet(panels[first:first + 4]).save(output / "raw" / f"contact_{first//4 + 1:02d}.jpg", quality=95)
    manifest = dict(format="rgb_semantic_audit_v1", seed=42, split="trainA", frames=rows,
                    selection="scene round-robin, one randomly selected frame per video, before predictions",
                    protocol_sha256=sha256_file(protocol), geometry=[360, 288], modes=list(MODES),
                    short_side=512, flip_tta=False, core_settings=CORE_SETTINGS,
                    points="fixed 3x3 locations; uncertain points excluded; not uniform pixel sampling",
                    prepare_code_sha256=sha256_file(Path(__file__)),
                    limits=["No FLIR semantic pixel truth; manual points are a small development audit.",
                            "No color prior fitting, GAN training or production preprocessing change."])
    write_json(output / "input_manifest.json", manifest)
    write_json(output / "manual_points.json", dict(
        annotation_protocol="RGB-only review before predictions; four color-prior classes plus other/uncertain",
        author="pending", allowed=[*GROUPS, "uncertain"], points=annotations))
    write_json(output / "status.json", dict(phase="prepared", images=len(rows), points=len(annotations)))
    print(f"Prepared {len(rows)} images from {len(rows)} distinct videos; annotate {len(annotations)} fixed points.")


def read_annotations(output, manifest):
    annotation = json.loads((output / "manual_points.json").read_text(encoding="utf-8"))
    expected = {f'{row["prefix"]}_P{i}': (row["prefix"], x, y)
                for row in manifest["frames"] for i, (x, y) in enumerate(POINTS, 1)}
    points = annotation["points"]
    if len(points) != len(expected) or len({p["id"] for p in points}) != len(expected):
        raise ValueError("Manual annotation point set changed")
    for point in points:
        if expected.get(point["id"]) != (point["prefix"], point["x"], point["y"]):
            raise ValueError("Manual point coordinates changed")
        if point["expected"] not in (*GROUPS, "uncertain"):
            raise ValueError("Complete RGB-only point annotation before running inference")
    return annotation


@torch.inference_mode()
def predict_visible(model, rgb, mode, device="cpu", short_side=512):
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1:
        raise ValueError("Expected nonempty uint8 RGB [H,W,3]")
    if mode not in MODES or short_side < 32:
        raise ValueError("Invalid visible input probe")
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError("The semantic audit requires a frozen eval model")
    if mode == "gray":
        gray = np.asarray(Image.fromarray(rgb).convert("L"))
        return predict_probabilities(model, gray, device, short_side=short_side, flip_tta=False)
    h, w = rgb.shape[:2]
    image = rgb
    if min(h, w) < short_side:
        scale = short_side / min(h, w)
        image = np.asarray(Image.fromarray(rgb).resize((int(scale*w), int(scale*h)), Image.Resampling.BILINEAR))
    normalized = (image.astype(np.float32) - MEAN) / STD
    tensor = torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1)))[None].to(device)
    logits = model(tensor)
    if logits.ndim != 4 or logits.shape[:2] != (1, 15) or not torch.isfinite(logits).all():
        raise ValueError("Expected finite 15-class logits")
    p = F.interpolate(logits.float(), size=(h, w), mode="bilinear", align_corners=True).softmax(1)[0].cpu().numpy()
    if not np.isfinite(p).all() or not np.allclose(p.sum(0), 1, atol=2e-6):
        raise ValueError("Invalid visible semantic probabilities")
    return p


def summarize_maps(p):
    labels = p.argmax(0).astype(np.uint8)
    top = np.partition(p, -2, axis=0)[-2:]
    maximum, margin = top[1], top[1] - top[0]
    core = np.full(labels.shape, 255, np.uint8)
    for index in TARGETS.values():
        mask = binary_erosion(labels == index, iterations=CORE_SETTINGS["erosion_pixels"], border_value=0)
        keep = mask & (maximum >= CORE_SETTINGS["min_probability"]) & (margin >= CORE_SETTINGS["min_margin"])
        core[keep] = index
    return labels, maximum, margin, core


def to_image(array):
    if array.dtype != np.uint8:
        array = np.rint(np.clip(array, 0, 1)*255).astype(np.uint8)
    return Image.fromarray(array).convert("RGB")


def infer(args):
    output = args.output_dir
    manifest = json.loads((output / "input_manifest.json").read_text(encoding="utf-8"))
    read_annotations(output, manifest)
    if (output / "predictions_manifest.json").exists() or (output / "maps").exists():
        raise ValueError("Existing predictions are preserved; use a new audit folder")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable")
    torch.set_num_threads(args.cpu_threads)
    model, checkpoint = load_checkpoint(args.checkpoint, torch.device(args.device))
    locked_hash = sha256_file(output / "manual_points.json")
    (output / "maps").mkdir()
    (output / "frames").mkdir()
    started, completed, stats = time.perf_counter(), [], []
    write_json(output / "annotation_lock.json", dict(sha256=locked_hash, locked_before_first_prediction=True))
    try:
        for row in manifest["frames"]:
            rgb = source_image(row)
            panels = [("Original visible RGB", to_image(rgb)),
                      ("Visible luminance (not thermal)", Image.fromarray(rgb).convert("L").convert("RGB"))]
            maps = {}
            predictions = {}
            for mode in MODES:
                p = predict_visible(model, rgb, mode, args.device)
                labels, maximum, margin, core = summarize_maps(p)
                maps[f"{mode}_probabilities"] = p
                maps[f"{mode}_core"] = core
                predictions[mode] = dict(labels=labels, maximum=maximum, margin=margin, core=core)
                to_image(PALETTE[labels]).save(output / "frames" / f'{row["prefix"]}_{mode}_labels.png')
                Image.fromarray(labels).save(output / "maps" / f'{row["prefix"]}_{mode}_labels.png')
                Image.fromarray(core).save(output / "maps" / f'{row["prefix"]}_{mode}_candidate_core.png')
                overlay = to_image(.6 * rgb / 255 + .4 * PALETTE[labels] / 255)
                overlay.save(output / "frames" / f'{row["prefix"]}_{mode}_overlay.png')
                core_colors = np.full_like(rgb, 80)
                for index in TARGETS.values():
                    core_colors[core == index] = PALETTE[index]
                panels.extend(((f"{mode}: classes / display colors", to_image(PALETTE[labels])),
                               (f"{mode}: class overlay", overlay),
                               (f"{mode}: candidate cores / NOT truth", to_image(core_colors))))
                stat = dict(prefix=row["prefix"], mode=mode, filename=row["filename"],
                            mean_max_score=float(maximum.mean()), candidate_core_fraction=float((core != 255).mean()))
                for name, index in TARGETS.items():
                    stat[name + "_predicted_fraction"] = float((labels == index).mean())
                    stat[name + "_core_fraction"] = float((core == index).mean())
                stats.append(stat)
            np.savez_compressed(output / "maps" / f'{row["prefix"]}_responses.npz', **maps)
            contact_sheet(panels, columns=4).save(output / "frames" / f'{row["prefix"]}_comparison.jpg', quality=94)
            completed.append(row["prefix"])
            write_json(output / "status.json", dict(phase="inference", completed=len(completed), total=len(manifest["frames"])))
            print(f'{len(completed)}/{len(manifest["frames"])} {row["scene"]} / {row["filename"]}', flush=True)
        assert sha256_file(output / "manual_points.json") == locked_hash
        write_json(output / "predictions_manifest.json", dict(
            format="rgb_semantic_predictions_v1", input_manifest_sha256=sha256_file(output / "input_manifest.json"),
            annotation_sha256=locked_hash, checkpoint=checkpoint, modes=list(MODES),
            rgb_probe="same geometry/normalization; raw RGB channels, not author thermal preprocessing",
            gray_probe="Pillow luminance then unchanged author grayscale replication",
            environment=dict(python=sys.version, torch=torch.__version__, platform=platform.platform(), device=args.device),
            code_sha256={str(path.relative_to(ROOT)): sha256_file(path) for path in
                         (Path(__file__), ROOT / "scripts/semantic_preview/inference.py", ROOT / "scripts/semantic_preview/checkpoint.py")},
            elapsed_seconds=time.perf_counter()-started, completed=completed))
        with (output / "descriptive_statistics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=stats[0].keys()); writer.writeheader(); writer.writerows(stats)
        write_json(output / "status.json", dict(phase="predictions_complete", completed=len(completed), total=len(completed)))
    except BaseException as error:
        write_json(output / "status.json", dict(phase="failed_or_interrupted", completed=len(completed), error=str(error)))
        raise


def point_metrics(points):
    known = [p for p in points if p["expected"] != "uncertain"]
    confusion = {truth: {pred: 0 for pred in GROUPS} for truth in GROUPS}
    for point in known:
        confusion[point["expected"]][point["predicted"]] += 1
    classes = {}
    for name in GROUPS:
        actual = sum(confusion[name].values())
        predicted = sum(row[name] for row in confusion.values())
        correct = confusion[name][name]
        classes[name] = dict(actual_points=actual, predicted_points=predicted, correct_points=correct,
                             precision=correct/predicted if predicted else None,
                             recall=correct/actual if actual else None)
    accepted = [p for p in known if p["core_group"] is not None]
    return dict(total_points=len(points), uncertain_points=len(points)-len(known), evaluated_points=len(known),
                group_point_agreement=sum(p["expected"]==p["predicted"] for p in known)/len(known) if known else None,
                per_group=classes, confusion=confusion, accepted_candidate_core_points=len(accepted),
                candidate_core_correct_points=sum(p["expected"]==p["core_group"] for p in accepted),
                candidate_core_point_precision=sum(p["expected"]==p["core_group"] for p in accepted)/len(accepted) if accepted else None,
                limits="fixed spatial points and assistant visual labels; not pixel accuracy, IoU or guaranteed mask purity")


def evaluate(args):
    output = args.output_dir
    manifest = json.loads((output / "input_manifest.json").read_text(encoding="utf-8"))
    predictions = json.loads((output / "predictions_manifest.json").read_text(encoding="utf-8"))
    if sha256_file(output / "input_manifest.json") != predictions["input_manifest_sha256"]:
        raise ValueError("Input review changed after inference")
    if sha256_file(output / "manual_points.json") != predictions["annotation_sha256"]:
        raise ValueError("Annotations changed after prediction; this is no longer a blind point audit")
    annotations = read_annotations(output, manifest)
    details = {mode: [] for mode in MODES}
    group_by_id = {index:name for name,index in TARGETS.items()}
    for row in manifest["frames"]:
        with np.load(output / "maps" / f'{row["prefix"]}_responses.npz', allow_pickle=False) as maps:
            for mode in MODES:
                p, core = maps[mode+"_probabilities"], maps[mode+"_core"]
                for point in annotations["points"]:
                    if point["prefix"] != row["prefix"]:
                        continue
                    x, y = point["x"], point["y"]
                    predicted = int(p[:, y, x].argmax())
                    details[mode].append(dict(point, mode=mode, filename=row["filename"],
                        predicted=group_by_id.get(predicted,"other"), predicted_class=CLASS_NAMES[predicted],
                        predicted_score=float(p[predicted,y,x]), core_group=group_by_id.get(int(core[y,x]))))
    metrics = {mode:point_metrics(points) for mode,points in details.items()}
    write_json(output / "point_audit.json", dict(annotation_author=annotations["author"],
               annotation_sha256=predictions["annotation_sha256"], modes=metrics, points=details,
               evidence="development point audit; no full semantic truth, learned colors or TIR quality evidence"))
    legend = " ".join(f'<span style="border-left:14px solid rgb{tuple(int(v) for v in PALETTE[i])};padding-left:4px">{html.escape(label)}</span>'
                      for i,label in enumerate(CLASS_LABELS))
    cards = []
    for row in manifest["frames"]:
        cards.append(f'<article id="frame-{row["prefix"]}"><h2>{row["prefix"]} / {html.escape(row["scene"])} / {html.escape(row["filename"])}</h2>'
                     f'<a href="frames/{row["prefix"]}_comparison.jpg"><img loading="lazy" src="frames/{row["prefix"]}_comparison.jpg"></a>'
                     f'<details><summary>查看原图检查点</summary><img loading="lazy" src="raw/{row["prefix"]}_points.png"></details></article>')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RGB语义区域审查：现有FMB权重</title><style>body{font:16px/1.6 system-ui;margin:24px;background:#f4f6f8;color:#172434}main{max-width:1500px;margin:auto}article,.note{background:white;border:1px solid #ccd6e0;padding:16px;margin:16px 0;border-radius:8px}img{width:100%;height:auto}h2{font-size:17px;overflow-wrap:anywhere}.note{background:#fff4d5}.legend{display:flex;gap:14px;flex-wrap:wrap}</style><main>
<h1>RGB语义区域审查：现有冻结FMB权重</h1><div class="note">固定trainA开发图，每张来自不同视频；不训练、不拟合颜色先验。标签显示色不代表颜色学习结果。<br>
每图上行：原RGB、可见光灰度、RGB输入类别图、RGB输入叠加。下行：RGB候选核心区、灰度输入类别图、灰度叠加、灰度候选核心区。<br>
候选核心区使用预设0.9响应、0.2类别差和2像素腐蚀筛选；它不是可靠性保证。点审查在预测前只看原RGB标记，不能替代像素真值或全图准确率。</div>'''
    (output / "index.html").write_text(page+f'<div class="legend">{legend}</div>'+"".join(cards)+"</main></html>",encoding="utf-8")
    write_json(output / "status.json", dict(phase="completed", images=len(manifest["frames"]), manual_points=len(annotations["points"])))
    print(json.dumps(metrics,ensure_ascii=False,indent=2),flush=True)
    print(f"Audit: {output / 'index.html'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare","infer","evaluate"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-images", type=int, default=32, help="prepare only; one image per trainA video")
    parser.add_argument("--device", choices=("cpu","cuda"), default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--checkpoint", type=Path, default=ROOT/"runs/pretrained/segformer_fmb_b3/best_model.pth")
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error("cpu-threads must be positive")
    {"prepare":prepare,"infer":infer,"evaluate":evaluate}[args.stage](args)


if __name__ == "__main__":
    main()
