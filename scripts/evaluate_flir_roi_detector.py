#!/usr/bin/env python3
"""Audit a FLIR ROI detector on video-held-out trainB images at fixed thresholds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from train_flir_roi_detector import (
    DEFAULT_VAL_VIDEOS, FlirRoiDataset, box_iou, load_records, select_records,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pytorch-CycleGAN-and-pix2pix"))
from models.thermal_roi_detector import TARGET_LABELS, ThermalRoiDetector, boxes_to_importance


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "FLIR_datasets/trainB")
    parser.add_argument("--manifest", type=Path, default=ROOT / "FLIR_protocol_v2/manifest.csv")
    parser.add_argument("--coco", type=Path, default=ROOT / "FLIR_ADAS_v2/images_thermal_train/coco.json")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/detectors/flir_roi_ssdlite/threshold_audit.json")
    parser.add_argument("--val-videos", nargs="+", default=DEFAULT_VAL_VIDEOS)
    parser.add_argument("--max-val-images", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


@torch.inference_mode()
def collect_predictions(detector: ThermalRoiDetector, dataset: FlirRoiDataset):
    output = []
    for index in range(len(dataset)):
        image, target = dataset[index]
        prediction = detector.model([image])[0]
        output.append((prediction, target))
    return output


def scores_at_threshold(prediction, target, threshold: float, minimum_height: float = 0.0):
    true_positive = false_positive = false_negative = 0
    valid_gt = [index for index, box in enumerate(target["boxes"])
                if float(box[3] - box[1]) >= minimum_height]
    ignored_gt = set(range(len(target["boxes"]))) - set(valid_gt)
    used_gt = set()
    for index, score in enumerate(prediction["scores"]):
        if float(score) < threshold or int(prediction["labels"][index]) not in TARGET_LABELS:
            continue
        label = int(prediction["labels"][index])
        matches = [
            (box_iou(prediction["boxes"][index], target["boxes"][gt]), gt)
            for gt in valid_gt
            if gt not in used_gt and label == int(target["labels"][gt])
        ]
        if matches:
            best, gt = max(matches)
            if best >= 0.5:
                true_positive += 1
                used_gt.add(gt)
                continue
        # Predictions that only cover an ignored tiny target do not count
        # against large-target precision.
        if any(
            label == int(target["labels"][gt])
            and box_iou(prediction["boxes"][index], target["boxes"][gt]) >= 0.5
            for gt in ignored_gt
        ):
            continue
        false_positive += 1
    false_negative = len(valid_gt) - len(used_gt)
    return true_positive, false_positive, false_negative


def summarize(pairs, threshold: float, minimum_height: float):
    counts = np.sum(
        np.asarray([scores_at_threshold(prediction, target, threshold, minimum_height)
                    for prediction, target in pairs], dtype=np.int64),
        axis=0,
    )
    tp, fp, fn = (int(value) for value in counts)
    return {
        "minimum_target_height_px": minimum_height,
        "score_threshold": threshold,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision_iou_0_5": tp / max(tp + fp, 1),
        "recall_iou_0_5": tp / max(tp + fn, 1),
    }


def summarize_roi_coverage(pairs, threshold: float, minimum_height: float):
    covered = total = 0
    area_fractions = []
    for prediction, target in pairs:
        detections = [
            (tuple(float(value) for value in box), float(score), int(label))
            for box, score, label in zip(
                prediction["boxes"], prediction["scores"], prediction["labels"],
            )
            if float(score) >= threshold and int(label) in TARGET_LABELS
        ]
        roi = boxes_to_importance((288, 360), detections)
        area_fractions.append(float((roi >= 0.5).mean()))
        for box in target["boxes"]:
            if float(box[3] - box[1]) < minimum_height:
                continue
            x = min(359, max(0, int((float(box[0]) + float(box[2])) / 2)))
            y = min(287, max(0, int((float(box[1]) + float(box[3])) / 2)))
            total += 1
            covered += int(roi[y, x] >= 0.5)
    return {
        "minimum_target_height_px": minimum_height,
        "score_threshold": threshold,
        "gt_center_roi_coverage": covered / max(total, 1),
        "mean_image_roi_area_fraction": float(np.mean(area_fractions)),
    }


def main():
    args = parse_args()
    detector = ThermalRoiDetector(args.checkpoint)
    records = load_records(args.manifest, args.coco)
    _, val_records = select_records(
        records, set(args.val_videos), args.seed, None, args.max_val_images,
    )
    dataset = FlirRoiDataset(args.processed_dir, val_records)
    pairs = collect_predictions(detector, dataset)
    thresholds = (0.05, 0.10, 0.20, 0.30, 0.50)
    results = [summarize(pairs, threshold, minimum_height)
               for minimum_height in (0.0, 20.0) for threshold in thresholds]
    coverage = [summarize_roi_coverage(pairs, threshold, minimum_height)
                for minimum_height in (0.0, 20.0) for threshold in thresholds]
    report = {
        "checkpoint": str(args.checkpoint),
        "validation_images": len(dataset),
        "held_out_video_ids": sorted(args.val_videos),
        "testB_used": False,
        "results": results,
        "roi_coverage": coverage,
        "interpretation": "Box precision/recall evaluates ROI localization, not pixel contour quality.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")
    for result in results:
        print(f"height>={result['minimum_target_height_px']:g}, score>={result['score_threshold']:.2f}: "
              f"precision={result['precision_iou_0_5']:.3f}, recall={result['recall_iou_0_5']:.3f}")
    for result in coverage:
        print(f"height>={result['minimum_target_height_px']:g}, score>={result['score_threshold']:.2f}: "
              f"ROI center coverage={result['gt_center_roi_coverage']:.3f}, "
              f"image area={result['mean_image_roi_area_fraction']:.3f}")


if __name__ == "__main__":
    main()
