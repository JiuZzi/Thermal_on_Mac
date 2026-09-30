#!/usr/bin/env python3
"""Preview C2c's importance-guided contour map against the unchanged C2b.

Without --saliency-dir, importance is a smooth middle-third position prior,
not semantic object saliency. An external directory may contain predicted
per-image importance PNGs for preview, at the original TIR image resolution.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt
from torch import nn

from audit_c2_conditions import ROOT, load_tir, map_image
from models.conditioned_generator import ConditionedGenerator
from models.saliency_edges import center_importance, make_saliency_edge


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "FLIR_datasets/trainB")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/edge_audit/c2c_saliency")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--saliency-dir", type=Path,
                        help="Optional predicted importance PNGs, named <TIR stem>.png; preview only")
    source.add_argument("--oracle-coco", type=Path,
                        help="TRAINING-SET GT boxes for a diagnostic preview only; never for model training/test")
    parser.add_argument("--geometry", choices=("center_crop_256", "full"), default="center_crop_256")
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--saliency-sigmas", default="0.7,1.0,1.6")
    parser.add_argument("--saliency-inner-weight", type=float, default=0.8)
    parser.add_argument("--saliency-outer-weight", type=float, default=0.2)
    parser.add_argument("--saliency-transition-fraction", type=float, default=24.0 / 256.0)
    parser.add_argument("--saliency-background-gain", type=float, default=0.5)
    parser.add_argument("--edge-low-threshold", type=float, default=0.08)
    parser.add_argument("--edge-high-threshold", type=float, default=0.16)
    parser.add_argument("--edge-soft-width", type=float, default=1.0)
    parser.add_argument("--soft-high-thresholds", default="0.12,0.16,0.20")
    args = parser.parse_args()
    try:
        args.saliency_sigmas = tuple(float(x.strip()) for x in args.saliency_sigmas.split(","))
        args.soft_high_thresholds = tuple(float(x.strip()) for x in args.soft_high_thresholds.split(","))
        ConditionedGenerator(
            nn.Identity(), "soft_saliency", fusion_mode="direct",
            edge_low_threshold=args.edge_low_threshold,
            edge_high_threshold=args.edge_high_threshold,
            edge_soft_width=args.edge_soft_width,
            soft_high_thresholds=args.soft_high_thresholds,
            saliency_sigmas=args.saliency_sigmas,
            saliency_inner_weight=args.saliency_inner_weight,
            saliency_outer_weight=args.saliency_outer_weight,
            saliency_transition_fraction=args.saliency_transition_fraction,
            saliency_background_gain=args.saliency_background_gain,
        )
    except ValueError as error:
        parser.error(str(error))
    if args.sample_count < 1:
        parser.error("--sample-count must be positive")
    return args


def load_importance(path: Path, tir_path: Path, geometry: str) -> np.ndarray:
    with Image.open(tir_path) as image:
        original_size = image.size
    with Image.open(path) as image:
        if image.size != original_size:
            raise ValueError(f"Importance map must have the original TIR size: {path}")
        image = image.convert("L")
        if geometry == "center_crop_256":
            left = (image.width - 256) // 2
            top = (image.height - 256) // 2
            image = image.crop((left, top, left + 256, top + 256))
        return np.asarray(image, dtype=np.float32).copy() / 255.0


def load_oracle_index(path: Path) -> dict[str, tuple[dict, list[dict]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    target_names = {"person", "rider", "car", "bike", "motor", "bus", "truck", "other vehicle"}
    target_ids = {item["id"] for item in data["categories"] if item["name"] in target_names}
    annotations = defaultdict(list)
    for item in data["annotations"]:
        if item["category_id"] in target_ids:
            annotations[item["image_id"]].append(item)
    return {
        Path(image["file_name"]).name: (image, annotations[image["id"]])
        for image in data["images"]
    }


def oracle_importance(image: dict, annotations: list[dict], geometry: str) -> np.ndarray:
    """Rasterize source COCO boxes through the fixed FLIR offline geometry."""
    mask = Image.new("L", (360, 288), 0)
    draw = ImageDraw.Draw(mask)
    for item in annotations:
        x, y, width, height = item["bbox"]
        x1 = x * 500 / image["width"] - 70
        y1 = y * 400 / image["height"] - 56
        x2 = (x + width) * 500 / image["width"] - 70
        y2 = (y + height) * 400 / image["height"] - 56
        draw.rectangle((round(x1), round(y1), round(x2), round(y2)), fill=255)
    if geometry == "center_crop_256":
        mask = mask.crop((52, 16, 308, 272))
    inside = np.asarray(mask, dtype=np.uint8) > 0
    if not inside.any():
        return np.full(inside.shape, 0.1, dtype=np.float32)
    # Full weight inside each box, with a soft shoulder outside to retain
    # likely object boundaries at the box edge.
    shoulder = np.exp(-0.5 * (distance_transform_edt(~inside) / 5.0) ** 2)
    return (0.1 + 0.9 * shoulder).astype(np.float32)


def contact_sheet(rows: list[list[Image.Image]], names: list[str]) -> Image.Image:
    labels = ("TIR", "importance S", "fine", "coarse", "C2b", "C2c", "abs difference")
    tile_width, tile_height = rows[0][0].size
    label_height = 28
    sheet = Image.new("RGB", (tile_width * len(labels), (tile_height + label_height) * len(rows)), "white")
    draw = ImageDraw.Draw(sheet)
    for row_index, (panels, name) in enumerate(zip(rows, names)):
        top = row_index * (tile_height + label_height)
        for column, (panel, label) in enumerate(zip(panels, labels)):
            left = column * tile_width
            sheet.paste(panel, (left, top))
            draw.text((left + 4, top + tile_height + 4),
                      f"{label} | {name[:14]}" if column == 0 else label, fill="black")
    return sheet


def main() -> None:
    args = parse_args()
    oracle_index = None
    if args.oracle_coco is not None:
        if args.input_dir.resolve() != (ROOT / "FLIR_datasets/trainB").resolve():
            raise ValueError("Oracle boxes are restricted to FLIR_datasets/trainB previews")
        oracle_index = load_oracle_index(args.oracle_coco)
    files = sorted(path for path in args.input_dir.iterdir()
                   if path.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff"))
    if not files:
        raise ValueError(f"No images in {args.input_dir}")
    selected = random.Random(args.seed).sample(files, min(args.sample_count, len(files)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    individual_dir = args.output_dir / "individual"
    individual_dir.mkdir(exist_ok=True)
    c2b = ConditionedGenerator(
        nn.Identity(), "soft_multi", fusion_mode="direct",
        edge_low_threshold=args.edge_low_threshold,
        edge_high_threshold=args.edge_high_threshold,
        edge_soft_width=args.edge_soft_width,
        soft_high_thresholds=args.soft_high_thresholds,
    )
    c2c = ConditionedGenerator(
        nn.Identity(), "soft_saliency", fusion_mode="direct",
        edge_low_threshold=args.edge_low_threshold,
        edge_high_threshold=args.edge_high_threshold,
        edge_soft_width=args.edge_soft_width,
        soft_high_thresholds=args.soft_high_thresholds,
        saliency_sigmas=args.saliency_sigmas,
        saliency_inner_weight=args.saliency_inner_weight,
        saliency_outer_weight=args.saliency_outer_weight,
        saliency_transition_fraction=args.saliency_transition_fraction,
        saliency_background_gain=args.saliency_background_gain,
    )
    rows: list[list[Image.Image]] = []
    statistics: list[dict[str, float | str]] = []
    for index, path in enumerate(selected, start=1):
        original, tir = load_tir(path, args.geometry)
        image = np.clip((tir[0, 0].numpy() + 1.0) / 2.0, 0.0, 1.0)
        if oracle_index is not None:
            if path.name not in oracle_index:
                raise KeyError(f"No COCO image entry for {path.name}")
            metadata, annotations = oracle_index[path.name]
            importance = oracle_importance(metadata, annotations, args.geometry)
        elif args.saliency_dir is None:
            importance = center_importance(
                *image.shape, args.saliency_inner_weight,
                args.saliency_outer_weight, args.saliency_transition_fraction,
            )
        else:
            importance = load_importance(args.saliency_dir / f"{path.stem}.png", path, args.geometry)
        proposed, fine, coarse = make_saliency_edge(
            image, args.soft_high_thresholds,
            args.edge_low_threshold / args.edge_high_threshold,
            args.edge_soft_width, args.saliency_sigmas, importance,
            args.saliency_background_gain,
        )
        actual, reserved = c2c.make_condition(
            tir, saliency_override=torch.from_numpy(importance)[None, None],
        )
        np.testing.assert_allclose(actual[0, 0].numpy(), proposed, atol=1e-7, rtol=0)
        if torch.count_nonzero(reserved).item() != 0:
            raise RuntimeError("The reserved confidence channel must stay zero")
        baseline = c2b.make_condition(tir)[0][0, 0].numpy()
        diff = np.abs(proposed - baseline)
        panels = [original.convert("RGB")]
        panels += [map_image(torch.from_numpy(values)[None, None])
                   for values in (importance, fine, coarse, baseline, proposed, diff)]
        rows.append(panels)
        contact_sheet([panels], [path.name]).save(individual_dir / f"{index:02d}_{path.stem}.png")
        statistics.append({
            "filename": path.name,
            "importance_mean": float(importance.mean()),
            "fine_mean": float(fine.mean()),
            "coarse_mean": float(coarse.mean()),
            "c2b_mean": float(baseline.mean()),
            "c2c_mean": float(proposed.mean()),
            "c2b_c2c_mean_absolute_difference": float(diff.mean()),
        })
    contact_sheet(rows, [path.name for path in selected]).save(args.output_dir / "overview.png")
    contact_sheet(rows[:4], [path.name for path in selected[:4]]).save(args.output_dir / "quicklook.png")
    with (args.output_dir / "samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=statistics[0].keys())
        writer.writeheader()
        writer.writerows(statistics)
    summary = {
        "input_dir": str(args.input_dir), "sample_count": len(selected), "geometry": args.geometry,
        "saliency_source": (
            "training_box_oracle_preview_only" if oracle_index is not None else
            "position_prior" if args.saliency_dir is None else "external_map_preview_only"
        ),
        "saliency_dir": str(args.saliency_dir) if args.saliency_dir else None,
        "oracle_coco": str(args.oracle_coco) if args.oracle_coco else None,
        "saliency_sigmas": args.saliency_sigmas,
        "saliency_inner_weight": args.saliency_inner_weight,
        "saliency_outer_weight": args.saliency_outer_weight,
        "saliency_transition_fraction": args.saliency_transition_fraction,
        "saliency_background_gain": args.saliency_background_gain,
        "edge_low_threshold": args.edge_low_threshold,
        "edge_high_threshold": args.edge_high_threshold,
        "edge_soft_width": args.edge_soft_width,
        "soft_high_thresholds": args.soft_high_thresholds,
        "mean_c2b_c2c_absolute_difference": float(np.mean([
            item["c2b_c2c_mean_absolute_difference"] for item in statistics
        ])),
        "interpretation": (
            "Map differences are descriptive, not edge accuracy. Training GT boxes cannot be used at inference."
            if oracle_index is not None else
            "Map differences are descriptive, not edge accuracy. The center prior has no object semantics."
            if args.saliency_dir is None else
            "Map differences are descriptive, not edge accuracy. External map quality requires separate validation."
        ),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.output_dir / 'quicklook.png'}")
    print(f"Wrote {args.output_dir / 'overview.png'}")


if __name__ == "__main__":
    main()
