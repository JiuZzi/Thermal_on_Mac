#!/usr/bin/env python3
"""Prepare and export visually reviewed interior seed regions, not dense truth."""

import argparse
from collections import Counter
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt

from audit_rgb_semantics import DEFAULT_OUTPUT, TARGETS, source_image, write_json
from semantic_preview.checkpoint import sha256_file


def prepare(audit, excluded_frames=()):
    output = audit / "seed_regions"
    if output.exists():
        raise ValueError("Existing seed review is preserved")
    manifest = json.loads((audit / "input_manifest.json").read_text())
    notes = json.loads((audit / "manual_points.json").read_text()).get("frame_notes", {})
    excluded = set(excluded_frames)
    known_prefixes = {frame["prefix"] for frame in manifest["frames"]}
    if not excluded <= known_prefixes:
        raise ValueError("An excluded frame is absent from this audit")
    if any(not notes.get(prefix) for prefix in excluded):
        raise ValueError("Record an RGB-only frame note before excluding a frame")
    (output / "candidates").mkdir(parents=True)
    regions, cards = [], []
    for frame in manifest["frames"]:
        prefix = frame["prefix"]
        if prefix in excluded:
            continue
        rgb = source_image(frame)
        with np.load(audit / "maps" / f"{prefix}_responses.npz", allow_pickle=False) as maps:
            core = maps["rgb_core"]
        for name, index in TARGETS.items():
            mask = core == index
            distance = distance_transform_edt(mask)
            y, x = np.unravel_index(distance.argmax(), distance.shape)
            radius = min(12, int(max(0, distance[y, x]-1)/np.sqrt(2)))
            while radius >= 4 and not mask[y-radius:y+radius+1, x-radius:x+radius+1].all():
                radius -= 1
            if radius < 4:
                continue
            box = [int(x-radius), int(y-radius), int(x+radius+1), int(y+radius+1)]
            region_id = f"{prefix}_{name}"
            roi = Image.fromarray(rgb).crop(tuple(box))
            roi.save(output / "candidates" / f"{region_id}_roi.png")
            source = Image.fromarray(rgb)
            draw = ImageDraw.Draw(source)
            draw.rectangle(tuple(box), outline="red", width=2)
            source.save(output / "candidates" / f"{region_id}_context.png")
            card = Image.new("RGB", (240, 320), "white")
            ImageDraw.Draw(card).text((4, 4), region_id, fill="black")
            card.paste(source.resize((240, 192), Image.Resampling.BILINEAR), (0, 24))
            card.paste(roi.resize((96, 96), Image.Resampling.NEAREST), (0, 222))
            ImageDraw.Draw(card).text((104, 238), "ROI: nearest", fill="black")
            cards.append(card)
            regions.append(dict(id=region_id, prefix=prefix, filename=frame["filename"], video_id=frame["video_id"],
                                source_sha256=frame["source_sha256"], class_name=name, class_id=index,
                                box=box, status="pending", note=""))
    for start in range(0, len(cards), 6):
        sheet = Image.new("RGB", (720, 640), "white")
        for index, card in enumerate(cards[start:start+6]):
            sheet.paste(card, (index%3*240, index//3*320))
        sheet.save(output / f"review_{start//6+1:02d}.jpg", quality=95)
    write_json(output / "review.json", dict(
        selection="largest inscribed interior square per predicted class/image; direct-RGB core candidates",
        source_predictions_sha256=sha256_file(audit / "predictions_manifest.json"),
        review_protocol="post-prediction assistant visual review; separate from locked blind point audit",
        excluded_frames={prefix:notes[prefix] for prefix in sorted(excluded)},
        limits="Sparse interior seeds only, easy-region and semantic-score selection bias; not full masks or independent test truth.",
        regions=regions))
    print(f"Prepared {len(regions)} region candidates for visual review in {output}")


def export(audit):
    output = audit / "seed_regions"
    review = json.loads((output / "review.json").read_text())
    if sha256_file(audit / "predictions_manifest.json") != review["source_predictions_sha256"]:
        raise ValueError("Source predictions changed")
    if any(row["status"] not in ("accepted", "rejected", "uncertain") for row in review["regions"]):
        raise ValueError("Complete seed visual review first")
    if (output / "reviewed_masks").exists():
        raise ValueError("Existing exports are preserved")
    (output / "reviewed_masks").mkdir()
    masks = {}
    for row in review["regions"]:
        if row["status"] != "accepted":
            continue
        source_image(row)
        mask = masks.setdefault(row["prefix"], np.full((288,360), 255, np.uint8))
        x0,y0,x1,y1 = row["box"]
        if not (0<=x0<x1<=360 and 0<=y0<y1<=288) or (mask[y0:y1,x0:x1] != 255).any():
            raise ValueError("Invalid or overlapping reviewed seed regions")
        mask[y0:y1,x0:x1] = row["class_id"]
    for prefix, mask in masks.items():
        Image.fromarray(mask).save(output / "reviewed_masks" / f"{prefix}_seed_labels.png")
    accepted = [r for r in review["regions"] if r["status"] == "accepted"]
    summary = dict(candidate_regions=len(review["regions"]), accepted_regions=len(accepted),
                   status_counts=dict(Counter(r["status"] for r in review["regions"])),
                   accepted_class_counts=dict(Counter(r["class_name"] for r in accepted)),
                   accepted_image_count=len(masks), accepted_video_count=len({r["video_id"] for r in accepted}),
                   labeled_pixels=sum(int((m != 255).sum()) for m in masks.values()),
                   ignored_value=255, class_ids=TARGETS, review_sha256=sha256_file(output / "review.json"),
                   supervision="assistant visually reviewed sparse RGB interiors, not expert dense ground truth",
                   limits="Not representative full-class color distributions; not independent accuracy evidence; no color model fitted.")
    write_json(output / "summary.json", summary)
    cards=[]
    for row in review["regions"]:
        region_id=row["id"]
        cards.append(f'<article><h2>{html.escape(region_id)} / {row["status"]}</h2><p>{html.escape(row["note"])}</p>'
                     f'<img src="candidates/{region_id}_context.png"><img class="roi" src="candidates/{region_id}_roi.png"></article>')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>RGB内部种子区域复核</title>
<style>body{font:16px/1.6 system-ui;background:#f4f6f8;margin:24px;color:#172434}main{max-width:1100px;margin:auto}article{background:white;padding:16px;border:1px solid #ccd6e0;margin:12px 0}img{width:360px;max-width:70%;height:auto}.roi{width:120px;max-width:25%;image-rendering:pixelated;margin-left:20px}h2{font-size:18px}.note{background:#fff2ce;padding:16px}</style><main>
<h1>RGB内部种子区域：逐块视觉复核</h1><p class="note">红框是候选内部小区域，右侧为原像素放大。仅accepted区域写入稀疏标签图，其他像素为255。复核由Codex视觉完成，不是专家标注或全图语义真值。候选偏向容易的内部区域，不能直接代表完整颜色分布。当前未学习颜色先验。</p>'''
    (output / "index.html").write_text(page+"".join(cards)+"</main></html>",encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("prepare","export"))
    parser.add_argument("--audit-dir",type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument("--exclude-frames", nargs="*", default=[],
                        help="prepare only; explicit frame prefixes with existing RGB-only notes")
    args=parser.parse_args()
    if args.stage == "prepare":
        prepare(args.audit_dir, args.exclude_frames)
    else:
        if args.exclude_frames:
            parser.error("--exclude-frames is only valid for prepare")
        export(args.audit_dir)
