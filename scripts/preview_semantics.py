#!/usr/bin/env python3
"""Frozen FMB SegFormer-B3 on fixed FLIR trainB frames; no GAN training."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
from PIL import Image
import torch

from semantic_preview.checkpoint import load_checkpoint, sha256_file
from semantic_preview.inference import predict, CLASS_NAMES, CLASS_LABELS, PALETTE
from semantic_preview.review import save_frame, render_crop, response_stats, write_csv, write_page

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REVIEW = ROOT / "scripts/semantic_preview/fixed_review_seed42.json"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs/pretrained/segformer_fmb_b3/best_model.pth")
    parser.add_argument("--input-dir", type=Path, default=ROOT / "FLIR_datasets/trainB")
    parser.add_argument("--review-manifest", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--protocol-manifest", type=Path, default=ROOT / "FLIR_protocol_v2/manifest.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/edge_audit/segformer_fmb_b3_semantic_preview_seed42")
    parser.add_argument("--num-images", type=int, default=40)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--short-side", type=int, default=512)
    parser.add_argument("--detail-short-side", type=int, default=1024, help="Separate same-weight size probe; 0 disables")
    parser.add_argument("--flip-tta", action="store_true", help="Optional flip averaging on the detail probe only; baseline unchanged")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--zoom", type=int, default=4)
    args = parser.parse_args(argv)
    if args.num_images < 1 or args.cpu_threads < 1 or not 1 <= args.zoom <= 12:
        parser.error("Positive image/thread counts and zoom 1..12 required")
    if args.short_side != 512:
        parser.error("Author baseline is fixed at short-side 512; use detail-short-side for a separate size probe")
    if args.detail_short_side and not 512 < args.detail_short_side <= 1536:
        parser.error("detail-short-side must be 0, or >512 and <=1536")
    if args.flip_tta and not args.detail_short_side:
        parser.error("flip-tta requires a separate detail probe")
    if args.input_dir.resolve() != (ROOT / "FLIR_datasets/trainB").resolve():
        parser.error("This development preview uses trainB only; testB must not select semantic settings")
    if args.output_dir.exists():
        parser.error("Use a new output directory; earlier review files and ratings will not be overwritten")
    return args


def choose_device(name):
    mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if mps else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable in this interpreter; select cpu or fix the CUDA environment")
    if name == "mps" and not mps:
        raise ValueError("MPS unavailable in this process; use cpu or an MPS-enabled terminal")
    return torch.device(name)


def load_review(args):
    if not args.review_manifest.is_file():
        raise FileNotFoundError(f"Missing fixed review manifest: {args.review_manifest}")
    review = json.loads(args.review_manifest.read_text(encoding="utf-8"))
    if review.get("geometry") != "full" or review.get("selection", {}).get("seed") != 42:
        raise ValueError("Use a seed42 full-frame review with source hashes and original-coordinate crops")
    samples = review["selection"]["samples"]
    if len({row["filename"] for row in samples}) != len(samples):
        raise ValueError("Duplicate fixed sample names")
    if args.num_images > len(samples):
        raise ValueError(f"Fixed review contains {len(samples)} images; request no more than that")
    with args.protocol_manifest.open(newline="", encoding="utf-8-sig") as handle:
        records = {row["destination_name"]: row for row in csv.DictReader(handle) if row["destination_split"] == "trainB"}
    selected = samples[:args.num_images]
    selected_names = {row["filename"] for row in selected}
    # Preflight every file BEFORE writing any preview. Crop coordinates never enter prediction.
    for row in selected:
        name = row["filename"]
        if Path(name).name != name or "/" in name or "\\" in name:
            raise ValueError("Review filenames must be basenames")
        if name not in records or records[name]["video_id"] != row["video_id"]:
            raise ValueError(f"Sample is not in the frozen trainB protocol: {name}")
        source = args.input_dir / name
        if not source.is_file() or sha256_file(source) != review.get("source_hashes", {}).get(name):
            raise ValueError(f"Fixed image missing or changed: {name}. Use the same processed FLIR data; do not silently substitute frames.")
        with Image.open(source) as image:
            if image.mode != "L" or image.size != (360, 288):
                raise ValueError(f"Expected processed 8-bit L-mode 360x288 TIR: {name}")
    targets = [item for item in review["items"] if item["filename"] in selected_names and item.get("priority")]
    if len({item["id"] for item in targets}) != len(targets):
        raise ValueError("Duplicate target IDs")
    for item in targets:
        if not all(char.isalnum() or char in "_-" for char in item["id"]):
            raise ValueError("Invalid target ID")
        x0, y0, x1, y1 = item["crop"]
        if not all(isinstance(v, int) for v in item["crop"]) or not (0 <= x0 < x1 <= 360 and 0 <= y0 < y1 <= 288):
            raise ValueError(f"Invalid crop geometry: {item['id']}")
        box = item["box"]
        if len(box) != 4 or not np.isfinite(box).all() or not (0 <= box[0] < box[2] <= 360 and 0 <= box[1] < box[3] <= 288):
            raise ValueError(f"Invalid diagnostic box: {item['id']}")
    return review, selected, targets


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def run(args):
    torch.set_num_threads(args.cpu_threads)
    device = choose_device(args.device)
    review, selected, source_targets = load_review(args)
    print(f"Preflight OK: {len(selected)} fixed trainB frames; {len(source_targets)} review crops. Loading frozen B3 on {device}.", flush=True)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    settings = {"short_side": args.short_side, "detail_short_side": args.detail_short_side,
                "detail_flip_tta": args.flip_tta, "baseline_flip_tta": False, "zoom": args.zoom,
                "normalization_mean": [123.675, 116.28, 103.53], "normalization_std": [58.395, 57.12, 57.375],
                "grayscale": "uint8 repeated to 3 channels; no contrast/gamma/CLAHE",
                "resize": "PIL bilinear, enlarge only; dimensions truncated as in author code",
                "restore": "bilinear logits, align_corners=True, then softmax on original coordinates",
                "fusion": "none; baseline and detail probe exported separately"}
    package_dir = ROOT / "scripts/semantic_preview"
    code_paths = [Path(__file__), *sorted(package_dir.rglob("*.py"))]
    code_hashes = {str(p.relative_to(ROOT)): sha256_file(p) for p in code_paths}
    identity = {"checkpoint_sha256": checkpoint["sha256"], "review_sha256": sha256_file(args.review_manifest),
                "settings": settings, "images": selected, "code_hashes": code_hashes}
    pack_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    modes = ["base"] + (["detail"] if args.detail_short_side else [])
    output = args.output_dir
    for name in ("frames", "maps", "targets"):
        (output / name).mkdir(parents=True, exist_ok=False)
    frames, targets, target_rows, frame_rows = [], [], [], []
    write_json(output / "status.json", {"phase": "running", "completed": 0, "total": len(selected)})
    started = time.perf_counter()
    try:
        for number, row in enumerate(selected, 1):
            with Image.open(args.input_dir / row["filename"]) as image:
                gray = np.asarray(image).copy()
            predictions, durations = {}, {}
            for mode in modes:
                t0 = time.perf_counter()
                predictions[mode] = predict(model, gray, device,
                    short_side=args.short_side if mode == "base" else args.detail_short_side,
                    flip_tta=False if mode == "base" else args.flip_tta)
                durations[mode] = time.perf_counter() - t0
            prefix = f"{number:02d}"
            files = save_frame(gray, predictions, output / "frames", prefix)
            arrays = {"TIR_uint8": gray, "class_names": np.asarray(CLASS_NAMES), "palette": PALETTE}
            for mode, prediction in predictions.items():
                arrays.update({f"{mode}_{key}": value for key, value in prediction.items() if isinstance(value, np.ndarray)})
            np.savez_compressed(output / "maps" / f"{prefix}_responses.npz", **arrays)
            frame = {**row, "prefix": prefix, "sha256": review["source_hashes"][row["filename"]],
                     "files": files, "timing_seconds": durations,
                     "geometry": {mode: {key: prediction[key] for key in ("input_size", "logit_size")} for mode, prediction in predictions.items()}}
            frames.append(frame)
            for mode, prediction in predictions.items():
                frame_rows.append(dict(frame=number, filename=row["filename"], group=row["group"], mode=mode,
                    inference_seconds=durations[mode], mean_entropy=float(prediction["normalized_entropy"].mean()),
                    mean_max_score=float(prediction["max_score"].mean()),
                    **{f"fraction_{name}": float((prediction["labels"] == i).mean()) for i, name in enumerate(CLASS_NAMES)}))
            for item in source_targets:
                if item["filename"] != row["filename"]:
                    continue
                target = {**item, "panel": f"{item['id']}_semantics.png"}
                render_crop(gray, predictions, item["crop"], output / "targets" / target["panel"], args.zoom)
                for mode, prediction in predictions.items():
                    stats = response_stats(prediction["probabilities"], item)
                    target[mode] = stats
                    target_rows.append(dict(id=item["id"], filename=item["filename"], group=item["group"],
                        category=item["category"], mode=mode, truncated=item.get("truncated_by_preprocessing", False),
                        box=json.dumps(item["box"]), crop=json.dumps(item["crop"]), **stats))
                targets.append(target)
            write_page(output / "index.html", frames, targets, modes, pack_id, settings)
            write_json(output / "status.json", {"phase": "running", "completed": number, "total": len(selected)})
            print(f"[{number}/{len(selected)}] {row['group']} {row['filename']} | "
                  + " ".join(f"{mode}={durations[mode]:.2f}s" for mode in modes), flush=True)
        provenance = json.loads((package_dir / "upstream.json").read_text())
        manifest = {"format": "semantic_preview_v1", "pack_id": pack_id,
            "scope": "Frozen external FMB semantic prior; fixed trainB development review, not FLIR pixel accuracy",
            "checkpoint": checkpoint, "upstream": provenance, "settings": settings,
            "device": str(device), "environment": {"python": sys.version, "torch": torch.__version__, "platform": platform.platform()},
            "review_manifest_sha256": sha256_file(args.review_manifest),
            "protocol_manifest_sha256": sha256_file(args.protocol_manifest),
            "code_hashes": code_hashes,
            "classes": [{"id": i, "name": name, "label": CLASS_LABELS[i], "color": PALETTE[i].tolist()} for i,name in enumerate(CLASS_NAMES)],
            "frames": frames, "targets": targets, "elapsed_seconds": time.perf_counter() - started,
            "limits": ["No FLIR semantic pixel truth; response proxies are not accuracy or boundary recall.",
                       "Boxes/crops only localize review; none enter model inference.",
                       "The optional larger size uses interpolated input, not additional real detail.",
                       "Semantic region scores, entropy and maximum class score are not calibrated edge confidence.",
                       "Inference overlays are semantic visualization, not generated RGB.",
                       "The checkpoint's best_miou belongs to its original validation, not this FLIR preview.",
                       "Checkpoint tensor layout verified; original training split/provenance supplied by upstream, not inferred from weights."]}
        write_json(output / "preview_manifest.json", manifest)
        write_csv(output / "frames.csv", frame_rows)
        write_csv(output / "targets.csv", target_rows)
        write_json(output / "status.json", {"phase": "completed", "completed": len(frames), "total": len(selected),
                                           "targets": len(targets), "elapsed_seconds": manifest["elapsed_seconds"]})
        print(f"Completed: {output / 'index.html'}", flush=True)
        return manifest
    except BaseException as error:
        write_json(output / "status.json", {"phase": "failed_or_interrupted", "completed": len(frames),
                                            "total": len(selected), "error": str(error)})
        raise


if __name__ == "__main__":
    run(parse_args())
