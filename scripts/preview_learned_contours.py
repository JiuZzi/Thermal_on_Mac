"""Export float contour maps and uniformly rendered two-variant comparisons."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from contour_learning.data import file_hash
from contour_learning.model import FrozenContourExtractor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pytorch-CycleGAN-and-pix2pix"))
from models.conditioned_generator import ConditionedGenerator


def verify_comparison(a, b):
    if a["training_config"]["variant"] != "supervised" or b["training_config"]["variant"] != "contrastive":
        raise ValueError("checkpoint-a must be supervised; checkpoint-b must be contrastive")
    for key in ("dataset_sha256", "protocol_sha256", "initial_model_sha256", "purpose"):
        if a[key] != b[key]:
            raise ValueError(f"Comparison mismatch: {key}")
    for key, value in a["training_config"].items():
        if key != "variant" and b["training_config"].get(key) != value:
            raise ValueError(f"Comparison training config mismatch: {key}")


def render_panel(gray, maps, path, zoom=1, synthetic=False):
    names = ["TIR", "C1", "C2b", "C2c", "Boundary", "Boundary + local CL"]
    arrays = [gray] + list(maps.values())
    h, w = gray.shape
    cell_width = max(160, w * zoom)
    panel = Image.new("RGB", (cell_width * len(arrays), h * zoom * 2 + 40), "white")
    draw = ImageDraw.Draw(panel)
    base = np.repeat(gray[..., None], 3, -1)
    for i, (name, array) in enumerate(zip(names, arrays)):
        draw.text((i * cell_width + 3, 3), name, fill="black")
        top = Image.fromarray(np.uint8(np.clip(array, 0, 1) * 255)).convert("RGB")
        overlay = base if i == 0 else base * (1 - 0.65 * array[..., None]) + np.array([1, 0, 0]) * (0.65 * array[..., None])
        bottom = Image.fromarray(np.uint8(np.clip(overlay, 0, 1) * 255))
        for image, y in ((top, 20), (bottom, 20 + h * zoom)):
            panel.paste(image.resize((w * zoom, h * zoom), Image.Resampling.NEAREST),
                        (i * cell_width + (cell_width - w * zoom) // 2, y))
    footer = "SYNTHETIC SMOKE ONLY" if synthetic else "Red = strength"
    draw.text((3, h * zoom * 2 + 23), footer, fill="black")
    panel.save(path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-a", type=Path, required=True)
    parser.add_argument("--checkpoint-b", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--review-manifest", type=Path,
                        help="Optional existing review_manifest.json: same frames and priority crops")
    parser.add_argument("--max-images", type=int, default=40)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args(argv)


def run(args):
    if args.max_images < 1:
        raise ValueError("max-images must be positive")
    extractors = [FrozenContourExtractor(path, args.device) for path in (args.checkpoint_a, args.checkpoint_b)]
    verify_comparison(*(extractor.metadata for extractor in extractors))
    synthetic = extractors[0].metadata["purpose"] == "synthetic_implementation_check"
    labeled_images = {row["image_sha256"]: row["split"] for row in extractors[0].metadata["annotations"]}
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Preview output is nonempty; use a new directory")
    review = json.loads(args.review_manifest.read_text(encoding="utf-8")) if args.review_manifest else None
    if review:
        files = [args.image_dir / sample["filename"] for sample in review["selection"]["samples"]][:args.max_images]
    else:
        files = sorted(p for p in args.image_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))[:args.max_images]
    if not files:
        raise ValueError("No preview images")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    references = [ConditionedGenerator(torch.nn.Identity(), condition_mode=mode, fusion_mode="direct")
                  for mode in ("canny", "soft_multi", "soft_saliency")]
    entries = []
    for i, path in enumerate(files, 1):
        image_hash = file_hash(path)
        if review and review.get("source_hashes", {}).get(path.name) != image_hash:
            raise ValueError(f"Input differs from the fixed review image: {path.name}")
        with Image.open(path) as image:
            if image.mode != "L":
                raise ValueError(f"Expected processed 8-bit grayscale TIR: {path}")
            gray = np.array(image, dtype=np.float32) / 255
        tir = torch.from_numpy(gray[None, None] * 2 - 1)
        maps = {name: generator.make_condition(tir)[0][0, 0].numpy()
                for name, generator in zip(("C1", "C2b", "C2c"), references)}
        for name, extractor in zip(("supervised", "contrastive"), extractors):
            maps[name] = extractor(tir)[0, 0].numpy()
        np.savez_compressed(args.output_dir / f"{i:02d}_maps.npz", TIR=gray, **maps)
        for name in ("supervised", "contrastive"):
            Image.fromarray(np.uint8(maps[name] * 255)).save(args.output_dir / f"{i:02d}_{name}.png")
        render_panel(gray, maps, args.output_dir / f"{i:02d}_comparison.png",
                     zoom=4 if synthetic else 1, synthetic=synthetic)
        crops = []
        if review:
            for item in review["items"]:
                if item["filename"] != path.name or not item.get("priority"):
                    continue
                x1, y1, x2, y2 = map(int, item["crop"])
                if not (0 <= x1 < x2 <= gray.shape[1] and 0 <= y1 < y2 <= gray.shape[0]):
                    raise ValueError("Review crop does not match input geometry")
                crop_name = f"{i:02d}_{item['id']}_crop.png"
                render_panel(gray[y1:y2, x1:x2], {k: v[y1:y2, x1:x2] for k, v in maps.items()},
                             args.output_dir / crop_name, zoom=4, synthetic=synthetic)
                crops.append({"item_id": item["id"], "crop": [x1, y1, x2, y2], "panel": crop_name})
        entries.append({"filename": path.name, "image_sha256": image_hash, "crops": crops,
                        "annotation_split": labeled_images.get(image_hash, "not_annotated")})
    report = {"purpose": extractors[0].metadata["purpose"],
              "warning": "Synthetic checkpoints demonstrate interfaces only; not FLIR quality evidence."
                         if extractors[0].metadata["purpose"] == "synthetic_implementation_check" else None,
              "checkpoint_sha256": [file_hash(args.checkpoint_a), file_hash(args.checkpoint_b)],
              "metadata": [extractor.metadata for extractor in extractors], "images": entries,
              "display": "All maps use [0,1], no per-image normalization. Red means strength, not correctness."}
    (args.output_dir / "preview_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {len(entries)} full-frame comparisons to {args.output_dir}; purpose={report['purpose']}", flush=True)


if __name__ == "__main__":
    run(parse_args())
