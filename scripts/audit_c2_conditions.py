#!/usr/bin/env python3
"""Preview C1/C2 conditions and optionally measure normal-vs-zero edge use.

Run without --checkpoint-run-dir before training to inspect the same fixed
processed TIR images. After a short C2 pilot, pass its checkpoint directory to
compare one generator's RGB outputs with normal and zeroed edge conditions.
An output difference measures reliance, not whether the edge improved quality.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pytorch-CycleGAN-and-pix2pix"))

from models import networks  # noqa: E402
from models.conditioned_generator import ConditionedGenerator  # noqa: E402


def args_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "FLIR_datasets/trainB")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/edge_audit/c2_trainB")
    parser.add_argument("--geometry", choices=("center_crop_256", "full"), default="center_crop_256")
    parser.add_argument("--sample-count", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--edge-sigma", type=float, default=1.0)
    parser.add_argument("--edge-low-threshold", type=float, default=0.08)
    parser.add_argument("--edge-high-threshold", type=float, default=0.16)
    parser.add_argument("--edge-soft-width", type=float, default=1.0)
    parser.add_argument("--soft-high-thresholds", type=str, default="0.12,0.16,0.20")
    parser.add_argument("--checkpoint-run-dir", type=Path, help="C2a/C2b run directory with train_opt.txt and weights")
    parser.add_argument("--epoch", default="latest")
    args = parser.parse_args()
    if args.sample_count < 1:
        parser.error("--sample-count must be positive")
    try:
        args.soft_high_thresholds = tuple(float(part.strip()) for part in args.soft_high_thresholds.split(","))
        ConditionedGenerator(
            nn.Identity(), "soft_multi", edge_sigma=args.edge_sigma,
            edge_low_threshold=args.edge_low_threshold,
            edge_high_threshold=args.edge_high_threshold,
            edge_soft_width=args.edge_soft_width,
            soft_high_thresholds=args.soft_high_thresholds,
            fusion_mode="direct",
        )
    except ValueError as error:
        parser.error(str(error))
    return args


def training_options(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    options = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" in line and not line.startswith("-"):
            key, value = line.split(":", 1)
            options[key.strip()] = value.split("\t", 1)[0].strip()
    return options


def load_generator(run_dir: Path, epoch: str) -> tuple[ConditionedGenerator, dict[str, str], torch.device]:
    options = training_options(run_dir / "train_opt.txt")
    mode = options.get("condition_mode")
    if mode not in ("soft_single", "soft_multi") or options.get("fusion_mode") != "direct":
        raise ValueError("Checkpoint must be a direct-fusion C2a/C2b run")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    backbone = networks.define_G(
        3, 3, int(options["ngf"]), options["netG"], options["norm"],
        options["no_dropout"] == "False", options["init_type"], float(options["init_gain"]),
    )
    generator = ConditionedGenerator(
        backbone, mode,
        edge_sigma=float(options["edge_sigma"]),
        edge_low_threshold=float(options["edge_low_threshold"]),
        edge_high_threshold=float(options["edge_high_threshold"]),
        edge_soft_width=float(options["edge_soft_width"]),
        soft_high_thresholds=ast.literal_eval(options["soft_high_thresholds"]),
        fusion_mode="direct",
    )
    checkpoint = run_dir / f"{epoch}_net_G_A.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    try:
        weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        weights = torch.load(checkpoint, map_location="cpu")
    generator.load_state_dict(weights, strict=True)
    return generator.to(device).eval(), options, device


def load_tir(path: Path, geometry: str) -> tuple[Image.Image, torch.Tensor]:
    with Image.open(path) as image:
        image = image.convert("RGB").convert("L")
        if geometry == "center_crop_256":
            if min(image.size) < 256:
                raise ValueError(f"Image is smaller than the training crop: {path}")
            left = (image.width - 256) // 2
            top = (image.height - 256) // 2
            image = image.crop((left, top, left + 256, top + 256))
        image = image.copy()
    values = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return image, torch.from_numpy(values[None, None])


def to_rgb(tensor: torch.Tensor) -> Image.Image:
    values = tensor.detach().float().cpu().numpy()[0].transpose(1, 2, 0)
    return Image.fromarray(np.clip((values + 1) * 127.5, 0, 255).astype(np.uint8), "RGB")


def map_image(edge: torch.Tensor) -> Image.Image:
    values = edge.detach().float().cpu().numpy()[0, 0]
    return Image.fromarray(np.clip(values * 255, 0, 255).astype(np.uint8), "L").convert("RGB")


def main() -> None:
    args = args_parser()
    files = sorted(path for path in args.input_dir.iterdir() if path.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff"))
    if not files:
        raise ValueError(f"No images in {args.input_dir}")
    selected = random.Random(args.seed).sample(files, min(args.sample_count, len(files)))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    common = dict(
        edge_sigma=args.edge_sigma, edge_low_threshold=args.edge_low_threshold,
        edge_high_threshold=args.edge_high_threshold, edge_soft_width=args.edge_soft_width,
        soft_high_thresholds=args.soft_high_thresholds, fusion_mode="direct",
    )
    extractors = {
        mode: ConditionedGenerator(nn.Identity(), mode, **common)
        for mode in ("canny", "soft_single", "soft_multi")
    }
    generator = options = device = None
    if args.checkpoint_run_dir is not None:
        generator, options, device = load_generator(args.checkpoint_run_dir, args.epoch)
        for key in ("edge_sigma", "edge_low_threshold", "edge_high_threshold", "edge_soft_width"):
            if not np.isclose(float(options[key]), getattr(args, key)):
                raise ValueError(f"Preview setting {key} differs from the checkpoint's training setting")
        if ast.literal_eval(options["soft_high_thresholds"]) != args.soft_high_thresholds:
            raise ValueError("Preview multi-threshold values differ from the checkpoint's training setting")

    rows = []
    previews = []
    for path in selected:
        original, tir = load_tir(path, args.geometry)
        hard = extractors["canny"].make_condition(tir)[0]
        single = extractors["soft_single"].make_condition(tir)[0]
        multi = extractors["soft_multi"].make_condition(tir)[0]
        row = {
            "filename": path.name,
            "hard_edge_pixel_fraction": float(hard.mean()),
            "single_soft_mean": float(single.mean()),
            "multi_soft_mean": float(multi.mean()),
            "single_multi_mean_abs_difference": float((single - multi).abs().mean()),
        }
        images = [original.convert("RGB"), map_image(hard), map_image(single), map_image(multi)]
        if generator is not None:
            with torch.no_grad():
                actual = tir.to(device)
                normal = generator(actual)
                zeroed = generator(actual, edge_override=torch.zeros_like(actual))
            row["normal_zero_rgb_mean_abs_difference"] = float((normal - zeroed).abs().mean() / 2)
            difference = (normal - zeroed).abs().mean(dim=1, keepdim=True) / 2
            images.extend((to_rgb(normal), to_rgb(zeroed), map_image(difference)))
        rows.append(row)
        previews.append(images)

    tile_w, tile_h = previews[0][0].size
    label_h = 24
    columns = ("TIR", "C1 hard", "C2a single", "C2b multi", "RGB normal", "RGB zero edge", "RGB abs diff")[:len(previews[0])]
    sheet = Image.new("RGB", (tile_w * len(columns), (tile_h + label_h) * len(previews)), "white")
    draw = ImageDraw.Draw(sheet)
    for row_index, (path, images) in enumerate(zip(selected, previews)):
        y = row_index * (tile_h + label_h)
        for column, image in enumerate(images):
            sheet.paste(image, (column * tile_w, y))
            draw.text((column * tile_w + 4, y + tile_h + 3), columns[column], fill="black")
    sheet.save(args.output_dir / "conditions.png")

    with (args.output_dir / "samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "input_dir": str(args.input_dir), "sample_count": len(selected), "geometry": args.geometry,
        "edge_sigma": args.edge_sigma, "edge_low_threshold": args.edge_low_threshold,
        "edge_high_threshold": args.edge_high_threshold, "edge_soft_width": args.edge_soft_width,
        "soft_high_thresholds": args.soft_high_thresholds,
        "checkpoint_run_dir": str(args.checkpoint_run_dir) if args.checkpoint_run_dir else None,
        "mean_hard_edge_pixel_fraction": float(np.mean([row["hard_edge_pixel_fraction"] for row in rows])),
        "mean_single_soft_value": float(np.mean([row["single_soft_mean"] for row in rows])),
        "mean_multi_soft_value": float(np.mean([row["multi_soft_mean"] for row in rows])),
        "mean_single_multi_difference": float(np.mean([row["single_multi_mean_abs_difference"] for row in rows])),
        "interpretation": "RGB normal-vs-zero difference measures condition use, not improvement or edge correctness.",
    }
    if generator is not None:
        summary["mean_normal_zero_rgb_difference"] = float(np.mean([
            row["normal_zero_rgb_mean_abs_difference"] for row in rows
        ]))
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {args.output_dir / 'conditions.png'}, samples.csv, and summary.json")


if __name__ == "__main__":
    main()
