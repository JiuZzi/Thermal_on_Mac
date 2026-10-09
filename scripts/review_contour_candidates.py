#!/usr/bin/env python3
"""Prepare a fixed trainB object review; boxes locate objects, not boundaries.

This is a human development audit, not an accuracy benchmark or training input.
Use --ratings-json to summarize the JSON exported by the offline review page.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = {1: "person", 2: "bike", 3: "car", 4: "motor", 6: "bus",
              7: "train", 8: "truck", 74: "rider", 79: "other_vehicle"}
NAMES = {"person": "行人", "bike": "自行车", "car": "汽车", "motor": "摩托车",
         "bus": "公交车", "train": "列车", "truck": "卡车", "rider": "骑行者",
         "other_vehicle": "其他车辆", "building_scene": "建筑主结构（全图审查）"}
METHODS = ("C1", "C2b", "C2c")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "FLIR_datasets/trainB")
    parser.add_argument("--manifest", type=Path, default=ROOT / "FLIR_protocol_v2/manifest.csv")
    parser.add_argument("--coco", type=Path, default=ROOT / "FLIR_ADAS_v2/images_thermal_train/coco.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/edge_audit/important_contour_review_v1")
    parser.add_argument("--sample-count", type=int, default=30)
    parser.add_argument("--challenge-count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--geometry", choices=("full", "center_crop_256"), default="full")
    parser.add_argument("--selection-json", type=Path, help="Reuse an existing selection.json for the same frames")
    parser.add_argument("--ratings-json", type=Path, help="Summarize downloaded ratings; do not regenerate images")
    args = parser.parse_args()
    if args.sample_count < 1 or args.challenge_count < 0:
        parser.error("sample-count >= 1 and challenge-count >= 0 are required")
    return args


def transformed_box(annotation, width, height, geometry):
    """Raw COCO -> resize 500x400 -> crop 360x288 -> optional crop 256."""
    x, y, w, h = annotation["bbox"]
    offset_x, offset_y = (70, 56) if geometry == "full" else (122, 72)
    limit_x, limit_y = (360, 288) if geometry == "full" else (256, 256)
    raw = (x * 500 / width - offset_x, y * 400 / height - offset_y,
           (x + w) * 500 / width - offset_x, (y + h) * 400 / height - offset_y)
    box = (max(0., min(limit_x, raw[0])), max(0., min(limit_y, raw[1])),
           max(0., min(limit_x, raw[2])), max(0., min(limit_y, raw[3])))
    return box, any(abs(a - b) > 1e-6 for a, b in zip(raw, box))


def records_from_annotations(args):
    if args.processed_dir.name != "trainB":
        raise ValueError("Development audit requires trainB, never testB")
    data = json.loads(args.coco.read_text(encoding="utf-8"))
    images = {Path(row["file_name"]).name: row for row in data["images"]}
    annotations = defaultdict(list)
    for row in data["annotations"]:
        if row["category_id"] in CATEGORIES:
            annotations[row["image_id"]].append(row)
    with args.manifest.open(newline="", encoding="utf-8-sig") as handle:
        rows = [r for r in csv.DictReader(handle) if r["destination_split"] == "trainB"]
    records = []
    for row in rows:
        image = images[row["destination_name"]]
        record = {"filename": row["destination_name"], "video_id": row["video_id"],
                  "source_width": image["width"], "source_height": image["height"],
                  "annotations": annotations[image["id"]]}
        # The challenge pool is selected ONLY from annotation size, not edge output.
        record["small_object_proxy"] = any(
            2 <= box[2] - box[0] <= 30 and 2 <= box[3] - box[1] <= 20 and not clipped
            for ann in record["annotations"]
            for box, clipped in [transformed_box(ann, image["width"], image["height"], "full")]
        )
        records.append(record)
    return sorted(records, key=lambda r: r["filename"])


def balanced_sample(records, count, rng):
    """Random round-robin over videos. This is video-balanced, not frame-uniform."""
    by_video = defaultdict(list)
    for record in records:
        by_video[record["video_id"]].append(record)
    for values in by_video.values():
        rng.shuffle(values)
    chosen = []
    while len(chosen) < count:
        videos = sorted(key for key, values in by_video.items() if values)
        if not videos:
            break
        rng.shuffle(videos)
        for video in videos:
            chosen.append(by_video[video].pop())
            if len(chosen) == count:
                break
    return chosen


def select_records(records, args):
    lookup = {r["filename"]: r for r in records}
    if args.selection_json:
        selection = json.loads(args.selection_json.read_text(encoding="utf-8"))
        selected = []
        for item in selection["samples"]:
            row = dict(lookup[item["filename"]])
            if row["video_id"] != item["video_id"]:
                raise ValueError("Selection video metadata changed")
            row["group"] = item["group"]
            selected.append(row)
        return selected, selection
    rng = random.Random(args.seed)
    regular = balanced_sample(records, args.sample_count, rng)
    used = {r["filename"] for r in regular}
    pool = [r for r in records if r["filename"] not in used and r["small_object_proxy"]]
    challenge = balanced_sample(pool, args.challenge_count, rng)
    if len(regular) != args.sample_count or len(challenge) != args.challenge_count:
        raise ValueError("Not enough images; reduce requested sample counts explicitly")
    selected = [dict(row, group="regular") for row in regular]
    selected += [dict(row, group="small_object_challenge") for row in challenge]
    selection = {"seed": args.seed, "regular_sampling": "video-balanced random round-robin",
                 "challenge_sampling": "same sampling within annotated small-object pool",
                 "small_object_proxy": "full processed bbox width 2..30, height 2..20, not clipped",
                 "samples": [{k: r[k] for k in ("filename", "video_id", "group")} for r in selected]}
    return selected, selection


def summarize(args):
    pack = json.loads((args.output_dir / "review_manifest.json").read_text(encoding="utf-8"))
    exported = json.loads(args.ratings_json.read_text(encoding="utf-8"))
    if exported.get("pack_id") != pack["pack_id"]:
        raise ValueError("Ratings belong to a different sample/geometry/code package")
    ratings = exported["ratings"]
    aggregates = {}
    flags = []
    eligible = completed = 0
    for item in pack["items"]:
        rating = ratings.get(item["id"], {})
        if any(rating.get(k, "") not in ("", "0", "1", "2", "u")
               for k in ("candidate", "A", "B", "C")):
            raise ValueError(f"Invalid grade in {item['id']}")
        visible = rating.get("visibility") in ("clear", "weak")
        complete = visible and all(rating.get(k) in ("0", "1", "2")
                                   for k in ("candidate", "A", "B", "C"))
        eligible += int(visible)
        completed += int(complete)
        key = (item["group"], item["category"])
        label = " / ".join(key)
        result = aggregates.setdefault(label, {"total": 0, "eligible": 0, "complete": 0,
                                               "grades": {m: Counter() for m in ("candidate", *METHODS)}})
        result["total"] += 1
        result["eligible"] += int(visible)
        result["complete"] += int(complete)
        if not complete:
            continue
        result["grades"]["candidate"][rating["candidate"]] += 1
        actual = {}
        for anonymous, method in pack["method_mapping"].items():
            actual[method] = rating[anonymous]
            result["grades"][method][rating[anonymous]] += 1
        reasons = []
        if rating["candidate"] == "2":
            reasons.append("候选并集大部分缺失：优先复核提边阶段")
        if actual["C2b"] == "0" and actual["C2c"] == "2":
            reasons.append("C2b基本保留但C2c大部分缺失：复核加权抑制或显示强度")
        if reasons:
            flags.append({"item_id": item["id"], "reasons": reasons})
    report = {"pack_id": pack["pack_id"], "item_count": len(pack["items"]),
              "visible_item_count": eligible, "complete_comparable_item_count": completed,
              "grade_meaning": {"0": "基本保留", "1": "局部缺失", "2": "大部分缺失"},
              "groups_and_categories": aggregates, "review_flags": flags,
              "limitation": "Human ordinal review, not pixel accuracy. Separate scene/object and sampling groups."}
    (args.output_dir / "review_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Complete comparable items: {completed}/{len(pack['items'])}")
    print(args.output_dir / "review_summary.json")


PAGE = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>重要轮廓逐目标审查</title><style>
body{font:16px/1.6 system-ui,sans-serif;background:#f4f6f8;color:#18212c;margin:0}main{max-width:1400px;margin:auto;padding:24px}
header{position:sticky;top:0;background:#fff;padding:12px 24px;border-bottom:1px solid #ddd;z-index:2}
button,select,input{font:inherit;padding:6px;margin:4px;border:1px solid #aab3bd;border-radius:5px}button{cursor:pointer;background:#eef4ff}
article{background:white;border:1px solid #ccd4de;border-radius:8px;padding:18px;margin:20px 0}img{max-width:100%;height:auto}
.raw{max-height:350px;image-rendering:pixelated}.notice{background:#fff2cf;padding:12px}.muted{color:#566577;font-size:14px}
details{margin:14px 0;border-top:1px solid #ddd;padding-top:10px}summary{cursor:pointer;color:#2459a0}
.fields{display:flex;flex-wrap:wrap;gap:8px}label{display:block}a{color:#2459a0}textarea{width:95%;min-height:60px;font:inherit}
</style><header><b>重要轮廓审查</b> <span id="progress"></span>
<button onclick="download()">导出审查结果 JSON</button><button onclick="document.getElementById('import').click()">导入结果 JSON</button>
<input id="import" type="file" accept=".json" hidden><button onclick="reveal()">完成后查看方法对应关系</button>
<select id="scope" onchange="filter()"><option value="pilot">先查前5张的重点目标</option><option value="priority">全部固定重点目标</option>
<option value="scene">建筑全图审查</option><option value="all">完整目标清单</option></select></header><main>
<h1>先判断原图，再检查轮廓</h1><p class="notice">只评原图可辨认的可见边界。框是FLIR定位参考，不是边界真值；框外、遮挡处不算漏检。
暗不等于无边：对照提供统一 ×1 和 ×4 显示。×4 仅供观察，训练数值不变。候选并集不是准确率上限。</p>
<p>0 基本保留：可辨认的主要外边界大致可追踪；1 局部缺失：部分明显边界有缺口；2 大部分缺失：可辨认的主体边界多数找不到；无法判断则选 u。
候选并集是3尺度×3阈值的硬Canny并集。建筑只能作全图人工审查；无建筑或边界不可辨认请排除。常规组与小目标挑战组分开汇总。</p>
<p class="muted">本页是trainB开发审查，不是独立测试；主观等级不是边界准确率。请定期导出JSON，浏览器本地缓存不替代文件备份。</p>
<p>默认精简入口每图最多3个固定目标：最小行人/骑行者、最小车辆、最大剩余目标。只按框尺寸选取，未看轮廓效果。
精简入口未覆盖的目标不能算作已通过，完整清单可从顶部切换。</p>
<div id="mapping"></div>__CARDS__</main><script>
const PACK=__PACK__, key='contour-review-'+PACK.pack_id;let ratings={};
try{ratings=JSON.parse(localStorage.getItem(key)||'{}')}catch(e){}
const fields=['visibility','candidate','A','B','C','clutter','notes'];
function restore(){document.querySelectorAll('[data-id]').forEach(card=>{const r=ratings[card.dataset.id]||{};fields.forEach(f=>{const el=card.querySelector('[name="'+f+'"]');if(el)el.value=r[f]||''})});progress()}
function progress(){let n=0;PACK.items.forEach(i=>{const r=ratings[i.id]||{};if(['clear','weak'].includes(r.visibility)&&['candidate','A','B','C'].every(k=>['0','1','2'].includes(r[k])))n++});let shown=document.querySelectorAll('article:not([hidden])').length;document.getElementById('progress').textContent='已完成可比较项 '+n+'；当前显示 '+shown+' 项'}
function filter(){const scope=document.getElementById('scope').value;document.querySelectorAll('article').forEach(card=>{const scene=card.dataset.category==='building_scene',priority=card.dataset.priority==='1';card.hidden=!(scope==='all'||(scope==='scene'&&scene)||(scope==='priority'&&priority&&!scene)||(scope==='pilot'&&Number(card.dataset.frame)<=5&&priority&&!scene))});progress()}
document.addEventListener('change',e=>{const card=e.target.closest('[data-id]');if(!card)return;const id=card.dataset.id;ratings[id]=ratings[id]||{};ratings[id][e.target.name]=e.target.value;try{localStorage.setItem(key,JSON.stringify(ratings))}catch(e){alert('本地缓存不可用，请导出JSON保存')}progress()});
function download(){const blob=new Blob([JSON.stringify({pack_id:PACK.pack_id,ratings:ratings},null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='contour_review_ratings.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}
document.getElementById('import').addEventListener('change',async e=>{try{const obj=JSON.parse(await e.target.files[0].text());if(obj.pack_id!==PACK.pack_id)throw Error('文件与当前审查包不对应');ratings=obj.ratings;localStorage.setItem(key,JSON.stringify(ratings));restore()}catch(err){alert(err.message)}});
function reveal(){document.getElementById('mapping').textContent='方法对应关系：'+Object.entries(PACK.method_mapping).map(([k,v])=>k+' = '+v).join('；')}
restore();filter();</script></html>'''


def select_field(name, title, choices):
    options = '<option value="">未审查</option>' + ''.join(
        f'<option value="{value}">{html.escape(label)}</option>' for value, label in choices)
    return f'<label>{html.escape(title)}<select name="{name}">{options}</select></label>'


def create_page(pack, cards):
    grade = [("0", "0 基本保留"), ("1", "1 局部缺失"), ("2", "2 大部分缺失"), ("u", "无法判断")]
    rendered = []
    for item in pack["items"]:
        card = cards[item["id"]]
        escaped_id = html.escape(item["id"])
        fields = ''.join(select_field(m, f"轮廓 {m}", grade) for m in ("A", "B", "C"))
        rendered.append(f'''<article data-id="{escaped_id}" data-category="{item['category']}" data-priority="{int(item['priority'])}" data-frame="{item['frame_number']}"><h2>{escaped_id} · {NAMES[item['category']]}</h2>
<p class="muted">{html.escape(item['filename'])} · {item['group']} · {card['note']}</p>
<p>第一步：仅看红外原图，判断边界是否可辨认。<a href="{card['frame']}" target="_blank">查看全图与目标编号</a></p>
<img class="raw" loading="lazy" src="{card['raw']}" alt="原始红外目标区域">
{select_field('visibility', '原图可见性', [('clear','边界可辨认'),('weak','较弱但部分可辨认'),('uncertain','无法辨认'),('absent','无对应目标 / 无建筑')])}
<details><summary>第二步：展开候选和匿名轮廓对照</summary><p>面板顺序：TIR、候选并集、A、B、C。
同一目标使用相同裁剪和最近邻放大；红色叠加表示强度，不表示正确性。</p>
<p>统一显示 ×1</p><img loading="lazy" src="{card['panel']}" alt="原始强度对照">
<p>统一显示 ×4（仅显示增益，不能当作实际强度）</p><img loading="lazy" src="{card['gain']}" alt="统一增益对照">
<div class="fields">{select_field('candidate','候选并集',grade)}{fields}</div></details>
{select_field('clutter','背景非关键纹理（可选，全图对比）',[('none','较少'),('some','部分'),('many','较多'),('u','无法判断')])}
<label>备注：缺口位置、错位、疑似误边、未被框定位的目标或建筑区域<textarea name="notes"></textarea></label></article>''')
    data = {"pack_id": pack["pack_id"], "items": [{"id": i["id"]} for i in pack["items"]],
            "method_mapping": pack["method_mapping"]}
    return PAGE.replace('__CARDS__', '\n'.join(rendered)).replace('__PACK__', json.dumps(data, ensure_ascii=True))


def prepare(args):
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    from skimage.feature import canny
    from torch import nn
    sys.path.insert(0, str(ROOT / "pytorch-CycleGAN-and-pix2pix"))
    from models.conditioned_generator import ConditionedGenerator

    if args.output_dir.exists():
        raise FileExistsError("Use a new output directory to preserve an existing review and its ratings")
    records = records_from_annotations(args)
    selected, selection = select_records(records, args)
    aliases = list(METHODS)
    random.Random(args.seed + 991).shuffle(aliases)
    mapping = dict(zip(("A", "B", "C"), aliases))
    generators = {m: ConditionedGenerator(nn.Identity(), mode, fusion_mode="direct")
                  for m, mode in zip(METHODS, ("canny", "soft_multi", "soft_saliency"))}
    output = args.output_dir
    (output / "frames").mkdir(parents=True)
    (output / "targets").mkdir()
    items, cards, hashes = [], {}, {}

    def overlay(gray, values, gain):
        base = np.repeat(gray[..., None], 3, axis=2)
        alpha = np.clip(values * gain, 0, 1)[..., None] * .8
        rgb = base * (1 - alpha) + np.array([255, 55, 45]) * alpha
        return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))

    def panel(gray, maps, crop, gain):
        source = Image.fromarray(gray, "L").convert("RGB").crop(crop)
        views = [source] + [overlay(gray, maps[m], gain).crop(crop) for m in ("candidate", *mapping.values())]
        scale = min(6, max(1, math.ceil(200 / source.height)))
        size = (source.width * scale, source.height * scale)
        sheet = Image.new("RGB", (size[0] * 5, size[1] + 24), "white")
        draw = ImageDraw.Draw(sheet)
        for index, (image, label) in enumerate(zip(views, ("TIR", "candidate union", "A", "B", "C"))):
            sheet.paste(image.resize(size, Image.Resampling.NEAREST), (index * size[0], 0))
            draw.text((index * size[0] + 4, size[1] + 3), label, fill="black")
        return sheet

    for frame_number, record in enumerate(selected, 1):
        prefix = f"{frame_number:02d}"
        path = args.processed_dir / record["filename"]
        hashes[record["filename"]] = hashlib.sha256(path.read_bytes()).hexdigest()
        with Image.open(path) as opened:
            original = opened.convert("RGB").convert("L")
            if original.size != (360, 288):
                raise ValueError(f"Expected processed 360x288 image: {path}")
            if args.geometry == "center_crop_256":
                original = original.crop((52, 16, 308, 272))
            gray = np.asarray(original, dtype=np.uint8).copy()
        tir = torch.from_numpy(gray.astype(np.float32) / 127.5 - 1)[None, None]
        maps = {}
        for method, generator in generators.items():
            edge, reserved = generator.make_condition(tir)
            assert torch.count_nonzero(reserved).item() == 0
            maps[method] = edge[0, 0].numpy()
        normalized = np.clip((tir[0, 0].numpy() + 1) / 2, 0, 1)
        maps["candidate"] = np.logical_or.reduce([
            canny(normalized, sigma=sigma, low_threshold=high * .5, high_threshold=high)
            for sigma in (.7, 1., 1.6) for high in (.12, .16, .20)]).astype(np.float32)
        np.savez_compressed(output / "frames" / f"{prefix}_maps.npz", **maps)
        frame = original.convert("RGB")
        draw = ImageDraw.Draw(frame)
        objects = []
        for annotation in sorted(record["annotations"], key=lambda ann: ann["id"]):
            box, clipped = transformed_box(annotation, record["source_width"], record["source_height"], args.geometry)
            if box[2] - box[0] < 2 or box[3] - box[1] < 2:
                continue
            number = len(objects) + 1
            target_id = f"{prefix}_O{number:02d}"
            category = CATEGORIES[annotation["category_id"]]
            draw.rectangle(box, outline=(0, 255, 255), width=1)
            draw.text((box[0], max(0, box[1] - 12)), f"O{number:02d}", fill=(0, 255, 255))
            objects.append((target_id, category, box, clipped, annotation["id"]))
        frame_file = f"frames/{prefix}_locations.png"
        frame.save(output / frame_file)
        area = lambda obj: (obj[2][2] - obj[2][0]) * (obj[2][3] - obj[2][1])
        priority_ids = set()
        for pool in ([obj for obj in objects if obj[1] in ("person", "rider")],
                     [obj for obj in objects if obj[1] not in ("person", "rider")]):
            if pool:
                priority_ids.add(min(pool, key=lambda obj: (area(obj), obj[0]))[0])
        remaining = [obj for obj in objects if obj[0] not in priority_ids]
        if remaining:
            priority_ids.add(max(remaining, key=lambda obj: (area(obj), obj[0]))[0])
        # Add one explicit manual scene item because COCO contains no building boxes.
        objects.append((f"{prefix}_scene", "building_scene", (0, 0, original.width, original.height), False, None))
        for target_id, category, box, clipped, annotation_id in objects:
            margin = max(6, round(max(box[2] - box[0], box[3] - box[1]) * .25))
            crop = (max(0, math.floor(box[0] - margin)), max(0, math.floor(box[1] - margin)),
                    min(original.width, math.ceil(box[2] + margin)), min(original.height, math.ceil(box[3] + margin)))
            raw_file = f"targets/{target_id}_tir.png"
            panel_file = f"targets/{target_id}_compare.png"
            gain_file = f"targets/{target_id}_gain4.png"
            raw = original.crop(crop)
            scale = min(6, max(1, math.ceil(200 / raw.height)))
            raw.resize((raw.width * scale, raw.height * scale), Image.Resampling.NEAREST).save(output / raw_file)
            panel(gray, maps, crop, 1).save(output / panel_file)
            panel(gray, maps, crop, 4).save(output / gain_file)
            item = {"id": target_id, "filename": record["filename"], "video_id": record["video_id"],
                    "group": record["group"], "category": category, "annotation_id": annotation_id,
                    "box": list(box), "crop": list(crop), "truncated_by_preprocessing": clipped,
                    "frame_number": frame_number, "priority": target_id in priority_ids}
            items.append(item)
            cards[target_id] = {"raw": raw_file, "panel": panel_file, "gain": gain_file, "frame": frame_file,
                                "note": "裁剪截断：只评可见边界" if clipped else "框仅作定位参考"}
        print(f"Prepared frame {frame_number}/{len(selected)}: {len(objects)-1} objects", flush=True)
    code_hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (
        "scripts/review_contour_candidates.py", "pytorch-CycleGAN-and-pix2pix/models/conditioned_generator.py",
        "pytorch-CycleGAN-and-pix2pix/models/saliency_edges.py")}
    pack = {"selection": selection, "geometry": args.geometry, "method_mapping": mapping,
            "items": items, "source_hashes": hashes, "code_hashes": code_hashes,
            "parameters": {"highs": [.12, .16, .20], "low_ratio": .5, "sigmas": [.7, 1., 1.6],
                           "soft_width": 1., "background_gain": .5},
            "priority_sampling": "per-frame smallest person/rider, smallest vehicle, largest remaining; bbox only",
            "scope": "trainB development audit; object boxes are reference locations, never contour labels"}
    pack["pack_id"] = hashlib.sha256(json.dumps(pack, sort_keys=True).encode()).hexdigest()[:20]
    (output / "selection.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "review_manifest.json").write_text(json.dumps(pack, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "index.html").write_text(create_page(pack, cards), encoding="utf-8")
    with (output / "review_template.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        fields = ("id", "filename", "group", "category", "visibility", "candidate", "A", "B", "C", "clutter", "notes")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in items:
            writer.writerow({field: item.get(field, "") for field in fields})
    print(f"Review package: {output / 'index.html'}")
    print(f"Frames: {len(selected)}, object items: {sum(i['category'] != 'building_scene' for i in items)}, scene items: {len(selected)}")


if __name__ == "__main__":
    args = parse_args()
    if args.ratings_json:
        summarize(args)
    else:
        prepare(args)
