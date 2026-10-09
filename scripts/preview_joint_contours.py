"""Compare the two single-stage, label-free contour/GAN checkpoints."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pytorch-CycleGAN-and-pix2pix"))

from models import networks
from models.conditioned_generator import ConditionedGenerator
from models.joint_contour import JointContourGenerator
from models.joint_contour_cycle_gan_model import sha256_file


def load_generator(checkpoint, device):
    suffix = "_net_G_A.pth"
    if not checkpoint.name.endswith(suffix):
        raise ValueError("Use the joint experiment's *_net_G_A.pth checkpoint")
    metadata_path = checkpoint.with_name(checkpoint.name[:-len(suffix)] + "_joint_metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format") != "joint_contour_v1" or sha256_file(checkpoint) != metadata["checksums"]["G_A"]:
        raise ValueError("Checkpoint metadata/checksum mismatch")
    cfg = metadata["joint_config"]
    backbone = networks.define_G(3, 3, cfg["ngf"], cfg["netG"], cfg["norm"], not cfg["no_dropout"])
    source_keys = ("edge_sigma", "edge_low_threshold", "edge_high_threshold", "edge_soft_width",
                   "soft_high_thresholds", "saliency_sigmas", "saliency_inner_weight",
                   "saliency_outer_weight", "saliency_transition_fraction", "saliency_background_gain")
    generator = JointContourGenerator(
        backbone, cfg["condition_mode"], cfg["contour_width"], cfg["contour_embedding_dim"],
        cfg["contour_epsilon"], cfg["contour_residual_limit"], **{key: cfg[key] for key in source_keys})
    signature = list(hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).digest())
    generator.register_buffer("joint_signature", torch.tensor(signature, dtype=torch.uint8))
    generator.register_buffer("joint_updates", torch.zeros((), dtype=torch.int64))
    generator.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    if generator.joint_signature.tolist() != signature:
        raise ValueError("Checkpoint configuration signature mismatch")
    generator.to(device).eval().requires_grad_(False)
    return generator, metadata


def check_pair(metadata):
    a, b = metadata
    if a["joint_config"]["joint_variant"] != "plain" or b["joint_config"]["joint_variant"] != "contrastive":
        raise ValueError("Supply plain first, contrastive second")
    for key in ("training_data_sha256", "initial_parameters_sha256", "training_config", "scope", "updates"):
        if a[key] != b[key]:
            raise ValueError(f"Unmatched comparison: {key}")
    for key, value in a["joint_config"].items():
        if key != "joint_variant" and value != b["joint_config"][key]:
            raise ValueError(f"Unmatched joint configuration: {key}")


def render(gray, maps, rgbs, path, zoom=1, scope="quality_unverified"):
    arrays = {"TIR": gray, **maps, **rgbs}
    h, w = gray.shape
    cell_width = max(160, w * zoom)
    panel = Image.new("RGB", (cell_width * len(arrays), h * zoom * 2 + 45), "white")
    draw = ImageDraw.Draw(panel)
    base = np.repeat(gray[..., None], 3, -1)
    for i, (name, array) in enumerate(arrays.items()):
        draw.text((i * cell_width + 4, 3), name, fill="black")
        if array.ndim == 3:
            top_array = bottom_array = array
        else:
            top_array = np.repeat(array[..., None], 3, -1)
            bottom_array = base if name == "TIR" else base * (1 - array[..., None] * 0.65) + np.array([1, 0, 0]) * array[..., None] * 0.65
        for array, y in ((top_array, 20), (bottom_array, 20 + h * zoom)):
            image = Image.fromarray(np.uint8(np.clip(array, 0, 1) * 255))
            panel.paste(image.resize((w * zoom, h * zoom), Image.Resampling.NEAREST),
                        (i * cell_width + (cell_width - w * zoom) // 2, y))
    footer = "IMPLEMENTATION SMOKE ONLY: not quality evidence" if scope == "implementation_smoke" else "Red = score strength, not correctness; inspect unseen videos."
    draw.text((3, h * zoom * 2 + 25), footer, fill="black")
    panel.save(path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plain-checkpoint", type=Path, required=True)
    parser.add_argument("--contrastive-checkpoint", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--review-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-images", type=int, default=5)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--probes", action="store_true", help="Also export zero/shift/prior contour interventions")
    return parser.parse_args(argv)


@torch.no_grad()
def run(args):
    if args.num_images < 1:
        raise ValueError("num-images must be positive")
    models = [load_generator(path, args.device) for path in (args.plain_checkpoint, args.contrastive_checkpoint)]
    metadata = [item[1] for item in models]
    check_pair(metadata)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Use a new preview output directory")
    review = json.loads(args.review_manifest.read_text(encoding="utf-8")) if args.review_manifest else None
    if review:
        files = [args.image_dir / row["filename"] for row in review["selection"]["samples"]][:args.num_images]
    else:
        files = sorted(p for p in args.image_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))[:args.num_images]
    if not files:
        raise ValueError("No input images")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    references = [ConditionedGenerator(torch.nn.Identity(), mode, fusion_mode="direct")
                  for mode in ("soft_multi", "soft_saliency")]
    entries = []
    for number, path in enumerate(files, 1):
        digest = sha256_file(path)
        if review and review.get("source_hashes", {}).get(path.name) != digest:
            raise ValueError("Fixed review image content differs")
        with Image.open(path) as image:
            if image.mode != "L":
                raise ValueError("Use processed 8-bit grayscale TIR")
            gray = np.array(image, dtype=np.float32) / 255
        cpu_tir = torch.from_numpy(gray[None, None] * 2 - 1)
        tir = cpu_tir.to(args.device)
        maps = {name: generator.make_condition(cpu_tir)[0][0, 0].numpy()
                for name, generator in zip(("C2b", "C2c"), references)}
        rgbs, interventions = {}, {}
        for tag, (generator, _) in zip(("JR0", "JR1"), models):
            details = generator(tir, return_details=True)
            maps.setdefault("S0", details["prior"][0, 0].cpu().numpy())
            maps[tag] = details["edge"][0, 0].cpu().numpy()
            rgbs[f"RGB {tag}"] = (details["rgb"][0].permute(1, 2, 0).cpu().numpy() + 1) / 2
            if args.probes:
                interventions[tag] = {}
                for intervention in ("zero", "shift", "prior"):
                    prediction = generator(tir, prior_override=details["prior"], condition_override=intervention)
                    array = (prediction[0].permute(1, 2, 0).cpu().numpy() + 1) / 2
                    interventions[tag][intervention] = float(np.abs(array - rgbs[f"RGB {tag}"]).mean())
                    Image.fromarray(np.uint8(np.clip(array, 0, 1) * 255)).save(args.output_dir / f"{number:02d}_{tag}_{intervention}_rgb.png")
        np.savez_compressed(args.output_dir / f"{number:02d}_maps.npz", TIR=gray, **maps, **rgbs)
        render(gray, maps, rgbs, args.output_dir / f"{number:02d}_comparison.png", scope=metadata[0]["scope"])
        for tag in ("JR0", "JR1"):
            Image.fromarray(np.uint8(maps[tag] * 255)).save(args.output_dir / f"{number:02d}_{tag}_soft.png")
        crops = []
        if review:
            for item in review["items"]:
                if item["filename"] != path.name or not item.get("priority"):
                    continue
                x1, y1, x2, y2 = map(int, item["crop"])
                if not (0 <= x1 < x2 <= gray.shape[1] and 0 <= y1 < y2 <= gray.shape[0]):
                    raise ValueError("Review crop geometry mismatch")
                crop_name = f"{item['id']}_comparison.png"
                render(gray[y1:y2, x1:x2], {k: a[y1:y2, x1:x2] for k, a in maps.items()},
                       {k: a[y1:y2, x1:x2] for k, a in rgbs.items()}, args.output_dir / crop_name,
                       zoom=4, scope=metadata[0]["scope"])
                crops.append(crop_name)
        entries.append({"filename": path.name, "sha256": digest, "crops": crops,
                        "condition_intervention_rgb_mean_absolute_change": interventions})
    manifest = {"scope": metadata[0]["scope"], "metadata": metadata, "images": entries,
                "meaning": "JR0=joint refinement; JR1=same + local map contrast. Scores are not confidence.",
                "limits": "Intervention sensitivity is not proof of contour correctness. Old trainB review frames may be training images."}
    (args.output_dir / "preview_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Exported {len(entries)} comparisons: {args.output_dir}", flush=True)


if __name__ == "__main__":
    run(parse_args())
