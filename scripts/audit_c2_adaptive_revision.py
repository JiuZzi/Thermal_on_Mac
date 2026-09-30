#!/usr/bin/env python3
"""Compare the two isolated C2c fixes with the previous adaptive C2c."""

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
    "TIR", "C2b", "previous adaptive", "soft density only",
    "revised C2c", "importance", "brighter than previous", "dimmer than previous",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "FLIR_datasets/trainB")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/edge_audit/c2c_revision")
    parser.add_argument("--geometry", choices=("center_crop_256", "full"), default="center_crop_256")
    parser.add_argument("--sample-count", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def contact_sheet(rows: list[list[Image.Image]], names: list[str]) -> Image.Image:
    width, height = rows[0][0].size
    label_height = 25
    sheet = Image.new("RGB", (len(LABELS) * width, len(rows) * (height + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    for row_index, (panels, name) in enumerate(zip(rows, names)):
        top = row_index * (height + label_height)
        for column, (panel, label) in enumerate(zip(panels, LABELS)):
            left = column * width
            sheet.paste(panel, (left, top))
            draw.text((left + 3, top + height + 3),
                      f"{label} | {name[:12]}" if column == 0 else label, fill="black")
    return sheet


def generate(path: Path, geometry: str, c2b: ConditionedGenerator,
             c2c: ConditionedGenerator):
    original, tir = load_tir(path, geometry)
    gray = np.clip((tir[0, 0].numpy() + 1.0) / 2.0, 0.0, 1.0)
    options = ((0.12, 0.16, 0.20), 0.5, 1.0, (0.7, 1.0, 1.6), 0.5)
    old, _, _, _ = make_adaptive_saliency_edge(
        gray, *options, variant="structure_texture_legacy",
    )
    density_only, _, _, _ = make_adaptive_saliency_edge(
        gray, *options, variant="structure_texture_density_only",
    )
    new, importance, fine, coarse = make_adaptive_saliency_edge(gray, *options)
    actual, reserved = c2c.make_condition(tir)
    np.testing.assert_allclose(actual[0, 0].numpy(), new, rtol=0, atol=1e-7)
    if torch.count_nonzero(reserved).item() != 0:
        raise RuntimeError("C2c confidence channel must remain zero")
    baseline = c2b.make_condition(tir)[0][0, 0].numpy()
    brighter = np.maximum(new - old, 0.0)
    dimmer = np.maximum(old - new, 0.0)
    maps = (baseline, old, density_only, new, importance, brighter, dimmer)
    panels = [original.convert("RGB")]
    panels += [map_image(torch.from_numpy(values)[None, None]) for values in maps]
    fine_only = (fine > 0.4) & (coarse < 0.1)
    return panels, {
        "filename": path.name,
        "old_mean": float(old.mean()),
        "density_only_mean": float(density_only.mean()),
        "new_mean": float(new.mean()),
        "mean_abs_change": float(np.abs(new - old).mean()),
        "brighter_pixel_fraction_delta_0_05": float((brighter > 0.05).mean()),
        "dimmer_pixel_fraction_delta_0_05": float((dimmer > 0.05).mean()),
        "fine_only_pixels": int(fine_only.sum()),
        "fine_only_old_to_fine_mean": float(np.mean(old[fine_only] / fine[fine_only])) if fine_only.any() else None,
        "fine_only_new_to_fine_mean": float(np.mean(new[fine_only] / fine[fine_only])) if fine_only.any() else None,
    }


def main():
    args = parse_args()
    if args.input_dir.resolve() != (ROOT / "FLIR_datasets/trainB").resolve():
        raise ValueError("Tune C2c on trainB, never on testB")
    if args.sample_count < 1:
        raise ValueError("--sample-count must be positive")
    files = sorted(path for path in args.input_dir.iterdir()
                   if path.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff"))
    if not files:
        raise ValueError(f"No input images in {args.input_dir}")
    selected = random.Random(args.seed).sample(files, min(args.sample_count, len(files)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "individual").mkdir(exist_ok=True)
    c2b = ConditionedGenerator(nn.Identity(), "soft_multi", fusion_mode="direct")
    c2c = ConditionedGenerator(nn.Identity(), "soft_saliency", fusion_mode="direct")
    rows, statistics = [], []
    for index, path in enumerate(selected, start=1):
        panels, item = generate(path, args.geometry, c2b, c2c)
        rows.append(panels)
        statistics.append(item)
        contact_sheet([panels], [path.name]).save(
            args.output_dir / "individual" / f"{index:02d}_{path.stem}.png",
        )
    names = [item["filename"] for item in statistics]
    contact_sheet(rows[:4], names[:4]).save(args.output_dir / "quicklook.png")
    contact_sheet(rows, names).save(args.output_dir / "overview.png")
    with (args.output_dir / "samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=statistics[0].keys())
        writer.writeheader()
        writer.writerows(statistics)
    summary = {
        "sample_count": len(statistics), "seed": args.seed, "geometry": args.geometry,
        "mean_abs_change": float(np.mean([item["mean_abs_change"] for item in statistics])),
        "mean_brighter_fraction": float(np.mean([item["brighter_pixel_fraction_delta_0_05"] for item in statistics])),
        "mean_dimmer_fraction": float(np.mean([item["dimmer_pixel_fraction_delta_0_05"] for item in statistics])),
        "interpretation": "Changes and fine-only ratios describe edge strength, not contour correctness or RGB quality.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output_dir / "quicklook.png", flush=True)


if __name__ == "__main__":
    main()
