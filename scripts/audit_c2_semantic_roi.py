#!/usr/bin/env python3
"""Preview detector-guided C2c and long-line structure retention on trainB.

Uses detector predictions only for the candidate edge map. FLIR labels are
used solely to choose video-held-out trainB images, never to create the ROI.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn

from audit_c2_conditions import ROOT, load_tir, map_image
from train_flir_roi_detector import DEFAULT_VAL_VIDEOS, load_records
from models.conditioned_generator import ConditionedGenerator
from models.saliency_edges import long_line_importance, object_line_contour
from models.thermal_roi_detector import ThermalRoiDetector, boxes_to_importance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "FLIR_datasets/trainB")
    parser.add_argument("--manifest", type=Path, default=ROOT / "FLIR_protocol_v2/manifest.csv")
    parser.add_argument("--coco", type=Path, default=ROOT / "FLIR_ADAS_v2/images_thermal_train/coco.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/edge_audit/c2c_semantic_roi")
    parser.add_argument("--val-videos", nargs="+", default=DEFAULT_VAL_VIDEOS)
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--score-threshold", type=float, default=0.10)
    parser.add_argument("--background-gain", type=float, default=0.5)
    args = parser.parse_args()
    if args.sample_count < 1 or not 0 < args.score_threshold < 1 or not 0 < args.background_gain <= 1:
        parser.error("Invalid sample count, score threshold, or background gain")
    if args.input_dir.resolve() != (ROOT / "FLIR_datasets/trainB").resolve():
        parser.error("Preview is restricted to FLIR trainB; testB must not tune the ROI")
    return args


def show_map(values: np.ndarray) -> Image.Image:
    return map_image(torch.from_numpy(values.astype(np.float32))[None, None])


def sheet(rows: list[list[Image.Image]], names: list[str]) -> Image.Image:
    labels = (
        "TIR + predicted boxes", "object ROI", "long lines", "C2b",
        "position C2c", "detector only", "detector + lines", "abs diff",
    )
    width, height = rows[0][0].size
    label_height = 26
    result = Image.new("RGB", (len(labels) * width, len(rows) * (height + label_height)), "white")
    draw = ImageDraw.Draw(result)
    for row, (panels, name) in enumerate(zip(rows, names)):
        top = row * (height + label_height)
        for column, (panel, label) in enumerate(zip(panels, labels)):
            left = column * width
            result.paste(panel, (left, top))
            draw.text((left + 4, top + height + 4),
                      f"{label} {name[:14]}" if column == 0 else label, fill="black")
    return result


def main() -> None:
    args = parse_args()
    detector = ThermalRoiDetector(args.checkpoint)
    records = load_records(args.manifest, args.coco)
    held_out = [row for row in records if row["video_id"] in set(args.val_videos)]
    selected = random.Random(args.seed).sample(held_out, min(args.sample_count, len(held_out)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "individual").mkdir(exist_ok=True)
    baseline = ConditionedGenerator(nn.Identity(), "soft_multi", fusion_mode="direct")
    position = ConditionedGenerator(nn.Identity(), "soft_saliency_position", fusion_mode="direct")
    train_path = ConditionedGenerator(
        nn.Identity(), "soft_object_line", fusion_mode="direct",
        detector_checkpoint=str(args.checkpoint),
        detector_score_threshold=args.score_threshold,
        saliency_background_gain=args.background_gain,
    )
    rows = []
    statistics = []
    for index, record in enumerate(selected, start=1):
        path = args.input_dir / record["filename"]
        original, tir = load_tir(path, "full")
        gray = np.clip((tir[0, 0].numpy() + 1.0) / 2.0, 0.0, 1.0)
        detections = detector.predict(gray, args.score_threshold)
        object_roi = boxes_to_importance(gray.shape, detections)
        line_roi, line_count = long_line_importance(gray)
        current = baseline.make_condition(tir)[0][0, 0].numpy()
        positional = position.make_condition(tir)[0][0, 0].numpy()
        detector_only = object_line_contour(
            gray, object_roi, np.zeros_like(line_roi), args.background_gain,
        )
        candidate = object_line_contour(gray, object_roi, line_roi, args.background_gain)
        actual, reserved = train_path.make_condition(tir)
        np.testing.assert_allclose(actual[0, 0].numpy(), candidate, atol=1e-7, rtol=0)
        if torch.count_nonzero(reserved).item() != 0:
            raise RuntimeError("The confidence channel must stay zero")
        marked = original.convert("RGB")
        draw = ImageDraw.Draw(marked)
        for box, score, _ in detections:
            draw.rectangle(box, outline="yellow", width=2)
            draw.text((box[0], box[1]), f"{score:.2f}", fill="yellow")
        panels = [marked, show_map(object_roi), show_map(line_roi), show_map(current),
                  show_map(positional), show_map(detector_only), show_map(candidate),
                  show_map(np.abs(candidate - current))]
        rows.append(panels)
        sheet([panels], [path.name]).save(args.output_dir / "individual" / f"{index:02d}_{path.stem}.png")
        statistics.append({
            "filename": path.name,
            "predicted_boxes": len(detections),
            "long_lines": line_count,
            "object_roi_mean": float(object_roi.mean()),
            "line_roi_mean": float(line_roi.mean()),
            "c2b_mean": float(current.mean()),
            "position_c2c_mean": float(positional.mean()),
            "detector_only_mean": float(detector_only.mean()),
            "candidate_c2c_mean": float(candidate.mean()),
            "candidate_vs_c2b_mean_absolute_difference": float(np.abs(candidate - current).mean()),
            "lines_added_mean_absolute_difference": float(np.abs(candidate - detector_only).mean()),
        })
    sheet(rows[:4], [row["filename"] for row in statistics[:4]]).save(args.output_dir / "quicklook.png")
    sheet(rows, [row["filename"] for row in statistics]).save(args.output_dir / "overview.png")
    with (args.output_dir / "samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=statistics[0].keys())
        writer.writeheader()
        writer.writerows(statistics)
    summary = {
        "detector_checkpoint": str(args.checkpoint),
        "val_video_ids": sorted(args.val_videos),
        "score_threshold": args.score_threshold,
        "background_gain": args.background_gain,
        "sample_count": len(statistics),
        "mean_predicted_boxes": float(np.mean([row["predicted_boxes"] for row in statistics])),
        "mean_c2b_difference": float(np.mean([row["candidate_vs_c2b_mean_absolute_difference"] for row in statistics])),
        "interpretation": "Predictions, not GT boxes, make the ROI. Long lines are not building labels. Map differences are not edge accuracy.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.output_dir / 'quicklook.png'}", flush=True)


if __name__ == "__main__":
    main()
