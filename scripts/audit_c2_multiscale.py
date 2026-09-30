#!/usr/bin/env python3
"""Preview how Canny smoothing scale changes the existing C2b soft edges.

Each scale keeps C2b's three thresholds and distance-softening width. Only
``edge_sigma`` varies. The last two panels compare equal and fixed weighted
means; this script does not change training or claim either is more accurate.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torch import nn

from audit_c2_conditions import ROOT, load_tir, map_image
from models.conditioned_generator import ConditionedGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "FLIR_datasets/trainB")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/edge_audit/c2_multiscale")
    parser.add_argument("--geometry", choices=("center_crop_256", "full"), default="center_crop_256")
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sigmas", default="0.7,1.0,1.6")
    parser.add_argument("--weights", default="0.25,0.50,0.25",
                        help="Exploratory fixed weights for the three sigma maps; normalized to sum to one")
    parser.add_argument("--edge-low-threshold", type=float, default=0.08)
    parser.add_argument("--edge-high-threshold", type=float, default=0.16)
    parser.add_argument("--soft-high-thresholds", default="0.12,0.16,0.20")
    parser.add_argument("--edge-soft-width", type=float, default=1.0)
    args = parser.parse_args()
    try:
        args.sigmas = tuple(float(value.strip()) for value in args.sigmas.split(","))
        args.weights = tuple(float(value.strip()) for value in args.weights.split(","))
        args.soft_high_thresholds = tuple(float(value.strip()) for value in args.soft_high_thresholds.split(","))
    except ValueError as error:
        parser.error(f"Expected comma-separated numbers: {error}")
    if len(args.sigmas) != 3 or any(value <= 0 for value in args.sigmas):
        parser.error("--sigmas must contain three positive smoothing scales")
    if len(args.weights) != 3 or any(value < 0 for value in args.weights) or sum(args.weights) <= 0:
        parser.error("--weights must contain three nonnegative values with a positive sum")
    args.weights = tuple(value / sum(args.weights) for value in args.weights)
    if args.sample_count < 1:
        parser.error("--sample-count must be positive")
    return args


def make_sheet(rows: list[list[Image.Image]], labels: list[str], filenames: list[str]) -> Image.Image:
    width, height = rows[0][0].size
    label_height = 29
    sheet = Image.new("RGB", (width * len(labels), (height + label_height) * len(rows)), "white")
    draw = ImageDraw.Draw(sheet)
    for row_index, (images, filename) in enumerate(zip(rows, filenames)):
        y = row_index * (height + label_height)
        for column_index, (label, panel) in enumerate(zip(labels, images)):
            x = column_index * width
            sheet.paste(panel, (x, y))
            caption = f"{label} | {filename[:21]}" if column_index == 0 else label
            draw.text((x + 4, y + height + 4), caption, fill="black")
    return sheet


def main() -> None:
    args = parse_args()
    files = sorted(
        path for path in args.input_dir.iterdir()
        if path.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff")
    )
    if not files:
        raise ValueError(f"No images in {args.input_dir}")
    selected = random.Random(args.seed).sample(files, min(args.sample_count, len(files)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    individual_dir = args.output_dir / "individual"
    individual_dir.mkdir(exist_ok=True)

    extractors = [
        ConditionedGenerator(
            nn.Identity(), "soft_multi", edge_sigma=sigma,
            edge_low_threshold=args.edge_low_threshold,
            edge_high_threshold=args.edge_high_threshold,
            edge_soft_width=args.edge_soft_width,
            soft_high_thresholds=args.soft_high_thresholds,
            fusion_mode="direct",
        )
        for sigma in args.sigmas
    ]
    labels = ["TIR"] + [f"sigma {sigma:g}" + (" (C2b)" if sigma == 1.0 else "") for sigma in args.sigmas]
    labels += ["equal mean", "weighted " + "/".join(f"{weight:.2f}" for weight in args.weights)]
    rows: list[list[Image.Image]] = []
    statistics: list[dict[str, object]] = []
    for index, path in enumerate(selected, start=1):
        original, tir = load_tir(path, args.geometry)
        edges = [extractor.make_condition(tir)[0] for extractor in extractors]
        average = torch.stack(edges).mean(dim=0)
        weighted = sum(weight * edge for weight, edge in zip(args.weights, edges))
        panels = [original.convert("RGB")] + [map_image(edge) for edge in edges]
        panels += [map_image(average), map_image(weighted)]
        rows.append(panels)
        make_sheet([panels], labels, [path.name]).save(individual_dir / f"{index:02d}_{path.stem}.png")
        statistics.append({
            "filename": path.name,
            **{f"sigma_{sigma:g}_mean": float(edge.mean()) for sigma, edge in zip(args.sigmas, edges)},
            "three_scale_mean": float(average.mean()),
            "weighted_mean": float(weighted.mean()),
            "small_vs_large_mean_abs_difference": float((edges[0] - edges[2]).abs().mean()),
            "c2b_vs_three_scale_mean_abs_difference": float((edges[1] - average).abs().mean()),
            "equal_vs_weighted_mean_abs_difference": float((average - weighted).abs().mean()),
        })

    make_sheet(rows, labels, [path.name for path in selected]).save(args.output_dir / "overview.png")
    make_sheet(rows[:4], labels, [path.name for path in selected[:4]]).save(args.output_dir / "quicklook.png")
    with (args.output_dir / "samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=statistics[0].keys())
        writer.writeheader()
        writer.writerows(statistics)
    summary = {
        "input_dir": str(args.input_dir),
        "sample_count": len(selected),
        "geometry": args.geometry,
        "sigmas": args.sigmas,
        "weights": args.weights,
        "edge_low_threshold": args.edge_low_threshold,
        "edge_high_threshold": args.edge_high_threshold,
        "soft_high_thresholds": args.soft_high_thresholds,
        "edge_soft_width": args.edge_soft_width,
        "interpretation": "Pixel means and differences describe maps; they do not measure contour accuracy.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.output_dir / 'quicklook.png'}")
    print(f"Wrote {args.output_dir / 'overview.png'}")
    print(f"Wrote {individual_dir}")


if __name__ == "__main__":
    main()
