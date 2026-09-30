#!/usr/bin/env python3
"""Compare position C2c with content-adaptive C2c candidates on trainB."""

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
from models.conditioned_generator import ConditionedGenerator
from models.saliency_edges import make_adaptive_saliency_edge


LABELS = (
    "TIR", "C2b", "position C2c", "importance", "structure+texture",
    "preserving importance", "edge-preserving", "candidate difference",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "FLIR_datasets/trainB")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/edge_audit/c2c_adaptive")
    parser.add_argument("--geometry", choices=("center_crop_256", "full"), default="center_crop_256")
    parser.add_argument("--sample-count", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def make_sheet(rows: list[list[Image.Image]], names: list[str]) -> Image.Image:
    width, height = rows[0][0].size
    label_height = 25
    output = Image.new("RGB", (len(LABELS) * width, len(rows) * (height + label_height)), "white")
    draw = ImageDraw.Draw(output)
    for row_index, (panels, name) in enumerate(zip(rows, names)):
        top = row_index * (height + label_height)
        for column, (panel, label) in enumerate(zip(panels, LABELS)):
            left = column * width
            output.paste(panel, (left, top))
            draw.text((left + 3, top + height + 3),
                      f"{label} | {name[:12]}" if column == 0 else label, fill="black")
    return output


def main():
    args = parse_args()
    if args.input_dir.resolve() != (ROOT / "FLIR_datasets/trainB").resolve():
        raise ValueError("Tune candidates on trainB only, not on testB")
    if args.sample_count < 1:
        raise ValueError("--sample-count must be positive")
    files = sorted(path for path in args.input_dir.iterdir()
                   if path.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff"))
    selected = random.Random(args.seed).sample(files, min(args.sample_count, len(files)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "individual").mkdir(exist_ok=True)
    c2b = ConditionedGenerator(nn.Identity(), "soft_multi", fusion_mode="direct")
    position = ConditionedGenerator(nn.Identity(), "soft_saliency_position", fusion_mode="direct")
    adaptive = ConditionedGenerator(nn.Identity(), "soft_saliency", fusion_mode="direct")
    rows = []
    stats = []
    for index, path in enumerate(selected, start=1):
        original, tir = load_tir(path, args.geometry)
        gray = np.clip((tir[0, 0].numpy() + 1.0) / 2.0, 0.0, 1.0)
        baseline = c2b.make_condition(tir)[0][0, 0].numpy()
        positional = position.make_condition(tir)[0][0, 0].numpy()
        plain, importance, _, _ = make_adaptive_saliency_edge(
            gray, (0.12, 0.16, 0.20), 0.5, 1.0, (0.7, 1.0, 1.6), 0.5,
        )
        filtered, filtered_importance, _, _ = make_adaptive_saliency_edge(
            gray, (0.12, 0.16, 0.20), 0.5, 1.0, (0.7, 1.0, 1.6), 0.5,
            variant="edge_preserving",
        )
        actual, reserved = adaptive.make_condition(tir)
        np.testing.assert_allclose(actual[0, 0].numpy(), plain, atol=1e-7, rtol=0)
        if torch.count_nonzero(reserved).item() != 0:
            raise RuntimeError("C2c confidence channel must remain zero")
        maps = [baseline, positional, importance, plain, filtered_importance,
                filtered, np.abs(plain - filtered)]
        panels = [original.convert("RGB")]
        panels += [map_image(torch.from_numpy(values)[None, None]) for values in maps]
        rows.append(panels)
        make_sheet([panels], [path.name]).save(args.output_dir / "individual" / f"{index:02d}_{path.stem}.png")
        stats.append({
            "filename": path.name,
            "importance_mean": float(importance.mean()),
            "preserving_importance_mean": float(filtered_importance.mean()),
            "c2b_edge_mean": float(baseline.mean()),
            "position_edge_mean": float(positional.mean()),
            "adaptive_edge_mean": float(plain.mean()),
            "preserving_edge_mean": float(filtered.mean()),
            "adaptive_vs_c2b_mae": float(np.abs(plain - baseline).mean()),
            "preserving_vs_c2b_mae": float(np.abs(filtered - baseline).mean()),
        })
    make_sheet(rows[:4], [item["filename"] for item in stats[:4]]).save(args.output_dir / "quicklook.png")
    make_sheet(rows, [item["filename"] for item in stats]).save(args.output_dir / "overview.png")
    with (args.output_dir / "samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=stats[0].keys())
        writer.writeheader()
        writer.writerows(stats)
    summary = {
        "sample_count": len(stats), "geometry": args.geometry, "seed": args.seed,
        "mean_adaptive_vs_c2b_mae": float(np.mean([item["adaptive_vs_c2b_mae"] for item in stats])),
        "mean_preserving_vs_c2b_mae": float(np.mean([item["preserving_vs_c2b_mae"] for item in stats])),
        "interpretation": "Map differences are not contour accuracy; compare important details and foliage visually.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output_dir / "quicklook.png", flush=True)


if __name__ == "__main__":
    main()
