#!/usr/bin/env python3
"""Preview position-guided multi-scale Canny fusion without changing training.

The optional texture exception uses local edge density, not a tree detector.
All panels use the existing C2b thresholds and edge softening; only the Canny
smoothing scales and their spatial fusion differ.
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
from scipy.ndimage import uniform_filter
from torch import nn

from audit_c2_conditions import ROOT, load_tir, map_image
from models.conditioned_generator import ConditionedGenerator


def comma_floats(value: str, name: str, parser: argparse.ArgumentParser) -> tuple[float, float, float]:
    try:
        result = tuple(float(part.strip()) for part in value.split(","))
    except ValueError as error:
        parser.error(f"{name}: expected three comma-separated numbers: {error}")
    if len(result) != 3 or any(part < 0 for part in result) or sum(result) <= 0:
        parser.error(f"{name}: expected three nonnegative numbers with a positive sum")
    total = sum(result)
    return tuple(part / total for part in result)


def positive_sigmas(value: str, parser: argparse.ArgumentParser) -> tuple[float, float, float]:
    try:
        result = tuple(float(part.strip()) for part in value.split(","))
    except ValueError as error:
        parser.error(f"--sigmas: expected three comma-separated numbers: {error}")
    if len(result) != 3 or any(part <= 0 for part in result):
        parser.error("--sigmas must contain three positive smoothing scales")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "FLIR_datasets/trainB")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/edge_audit/c2_roi_preview")
    parser.add_argument("--geometry", choices=("center_crop_256", "full"), default="center_crop_256")
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sigmas", default="0.7,1.0,1.6")
    parser.add_argument("--middle-weights", default="0.45,0.40,0.15")
    parser.add_argument("--outer-weights", default="0.15,0.45,0.40")
    parser.add_argument("--transition-px", type=float, default=24.0)
    parser.add_argument("--texture-window", type=int, default=15)
    parser.add_argument("--texture-low", type=float, default=0.35)
    parser.add_argument("--texture-high", type=float, default=0.55)
    parser.add_argument("--edge-low-threshold", type=float, default=0.08)
    parser.add_argument("--edge-high-threshold", type=float, default=0.16)
    parser.add_argument("--soft-high-thresholds", default="0.12,0.16,0.20")
    parser.add_argument("--edge-soft-width", type=float, default=1.0)
    args = parser.parse_args()
    args.sigmas = positive_sigmas(args.sigmas, parser)
    args.middle_weights = comma_floats(args.middle_weights, "--middle-weights", parser)
    args.outer_weights = comma_floats(args.outer_weights, "--outer-weights", parser)
    try:
        args.soft_high_thresholds = tuple(float(part.strip()) for part in args.soft_high_thresholds.split(","))
    except ValueError as error:
        parser.error(f"--soft-high-thresholds: {error}")
    if args.sample_count < 1 or args.transition_px <= 0 or args.texture_window < 1:
        parser.error("sample count, transition width, and texture window must be positive")
    if args.texture_window % 2 == 0 or not (0 <= args.texture_low < args.texture_high <= 1):
        parser.error("texture window must be odd and texture thresholds must satisfy 0 <= low < high <= 1")
    return args


def smoothstep(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0.0, 1.0)
    return values * values * (3.0 - 2.0 * values)


def center_band(height: int, transition_px: float) -> np.ndarray:
    y = np.arange(height, dtype=np.float32) + 0.5
    entering = smoothstep((y - (height / 3 - transition_px / 2)) / transition_px)
    leaving = smoothstep((y - (2 * height / 3 - transition_px / 2)) / transition_px)
    return entering * (1.0 - leaving)


def map_from_array(values: np.ndarray) -> Image.Image:
    return map_image(torch.from_numpy(values.astype(np.float32))[None, None])


def sheet(rows: list[list[Image.Image]], names: list[str]) -> Image.Image:
    width, height = rows[0][0].size
    labels = ("TIR", "current C2b", "equal mean", "position ROI", "ROI + texture", "texture exception")
    label_height = 28
    output = Image.new("RGB", (width * len(labels), (height + label_height) * len(rows)), "white")
    draw = ImageDraw.Draw(output)
    for row_idx, (panels, name) in enumerate(zip(rows, names)):
        top = row_idx * (height + label_height)
        for col_idx, (panel, label) in enumerate(zip(panels, labels)):
            left = col_idx * width
            output.paste(panel, (left, top))
            caption = f"{label} | {name[:18]}" if col_idx == 0 else label
            draw.text((left + 4, top + height + 4), caption, fill="black")
    return output


def main() -> None:
    args = parse_args()
    files = sorted(path for path in args.input_dir.iterdir() if path.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff"))
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
    middle = np.asarray(args.middle_weights, dtype=np.float32)[:, None, None]
    outer = np.asarray(args.outer_weights, dtype=np.float32)[:, None, None]
    rows: list[list[Image.Image]] = []
    statistics: list[dict[str, object]] = []
    for index, path in enumerate(selected, start=1):
        original, tir = load_tir(path, args.geometry)
        maps = np.stack([extractor.make_condition(tir)[0][0, 0].numpy() for extractor in extractors])
        height, width = maps.shape[1:]
        band = center_band(height, args.transition_px)[None, :, None]
        weights = outer * (1.0 - band) + middle * band
        equal = maps.mean(axis=0)
        position = np.sum(maps * weights, axis=0)

        density = uniform_filter((maps[1] > 0.5).astype(np.float32), size=args.texture_window)
        texture = smoothstep((density - args.texture_low) / (args.texture_high - args.texture_low))
        exception = texture * (1.0 - band[0])
        exception_weights = weights * (1.0 - exception[None]) + middle * exception[None]
        position_texture = np.sum(maps * exception_weights, axis=0)
        if not np.allclose(weights.sum(axis=0), 1.0, atol=1e-6) or not np.allclose(exception_weights.sum(axis=0), 1.0, atol=1e-6):
            raise RuntimeError("Spatial weights must sum to one at every pixel")

        panels = [
            original.convert("RGB"), map_from_array(maps[1]), map_from_array(equal),
            map_from_array(position), map_from_array(position_texture), map_from_array(exception),
        ]
        rows.append(panels)
        sheet([panels], [path.name]).save(individual_dir / f"{index:02d}_{path.stem}.png")
        statistics.append({
            "filename": path.name,
            "current_c2b_mean": float(maps[1].mean()),
            "equal_mean": float(equal.mean()),
            "position_mean": float(position.mean()),
            "position_texture_mean": float(position_texture.mean()),
            "texture_exception_mean": float(exception.mean()),
            "position_vs_c2b_mean_abs_difference": float(np.abs(position - maps[1]).mean()),
            "position_texture_vs_c2b_mean_abs_difference": float(np.abs(position_texture - maps[1]).mean()),
        })

    names = [path.name for path in selected]
    sheet(rows, names).save(args.output_dir / "overview.png")
    sheet(rows[:4], names[:4]).save(args.output_dir / "quicklook.png")
    with (args.output_dir / "samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=statistics[0].keys())
        writer.writeheader()
        writer.writerows(statistics)
    summary = {
        "input_dir": str(args.input_dir), "sample_count": len(selected), "geometry": args.geometry,
        "sigmas": args.sigmas, "middle_weights": args.middle_weights,
        "outer_weights": args.outer_weights, "transition_px": args.transition_px,
        "texture_window": args.texture_window, "texture_low": args.texture_low,
        "texture_high": args.texture_high,
        "edge_low_threshold": args.edge_low_threshold, "edge_high_threshold": args.edge_high_threshold,
        "soft_high_thresholds": args.soft_high_thresholds, "edge_soft_width": args.edge_soft_width,
        "interpretation": "The texture exception is not a tree segmentation. Map differences are not accuracy measures.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.output_dir / 'quicklook.png'}")
    print(f"Wrote {args.output_dir / 'overview.png'}")


if __name__ == "__main__":
    main()
