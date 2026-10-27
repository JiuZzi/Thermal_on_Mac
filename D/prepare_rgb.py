"""RGB self-pairs with the existing clarity fusion and a frozen RGB teacher."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image

from .common import ROOT, DEFAULT_CACHE, EDGE_SOURCES, read_json, write_json, select_sources, sha256

DEFAULT_RGB = ROOT / "runs/D/rgb_selfpairs/train256_val12_seed42"
TEACHER = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
REVISION = "d633b2072669ca68d8f8e309de9b52bfdbf6bf72"
NAMES = ("background", "road", "sidewalk", "building", "lamp", "sign", "vegetation", "sky",
         "person", "car", "truck", "bus", "motorcycle", "bicycle", "pole")
CITY_TO_FMB = {"road":"road", "sidewalk":"sidewalk", "building":"building", "wall":"background",
               "fence":"background", "pole":"pole", "traffic light":"lamp", "traffic sign":"sign",
               "vegetation":"vegetation", "terrain":"background", "sky":"sky", "person":"person",
               "rider":"person", "car":"car", "truck":"truck", "bus":"bus", "train":"background",
               "motorcycle":"motorcycle", "bicycle":"bicycle"}
PROBABILITY_POLICY = {"algorithm":"float64_group_sum_then_unit_mass_float32",
                      "roundoff_tolerance":3e-6, "reject_nonfinite":True}


def map_probabilities(probabilities, names):
    probabilities = np.asarray(probabilities)
    if (probabilities.ndim != 3 or not all(probabilities.shape[1:]) or
        set(names) != set(CITY_TO_FMB) or len(names) != 19 or probabilities.shape[0] != len(names)):
        raise ValueError("Expected the pinned 19-class Cityscapes teacher")
    tolerance = PROBABILITY_POLICY["roundoff_tolerance"]
    values = probabilities.astype(np.float64)
    if (not np.isfinite(values).all() or np.any(values < -tolerance) or np.any(values > 1 + tolerance)
        or not np.allclose(values.sum(0), 1., rtol=0, atol=tolerance)):
        raise ValueError("Invalid source categorical probabilities; expected finite unit mass in [0,1]")
    # Float32 sums can produce e.g. background=1.000000119. Correct only
    # rounding-sized errors; the contour module's strict validation stays intact.
    values = np.clip(values, 0., 1.)
    result = np.zeros((15, *values.shape[1:]), np.float64)
    for index, name in enumerate(names):
        result[NAMES.index(CITY_TO_FMB[name])] += values[index]
    result /= result.sum(0, keepdims=True)
    return result.astype(np.float32)


def resume_cache(output, expected, rows, source_base):
    """Verify the saved prefix before skipping it; never silently skip bad data."""
    saved = read_json(output / "manifest.json")
    status = read_json(output / "status.json")
    expected_base = json.loads(json.dumps({k:v for k,v in expected.items() if k != "frames"}))
    saved_base = {k:v for k,v in saved.items() if k not in ("frames", "selection", "probability_fix")}
    if saved_base != expected_base or status.get("total") != len(rows):
        raise ValueError("Resume requires identical source manifest, teacher, contour settings/code and sample count")
    plan = [dict(source=r["image"], role=r["split"], group=str(r["group"])) for r in rows]
    if "selection" in saved and saved["selection"] != plan:
        raise ValueError("Resume sample selection differs")
    if "probability_fix" in saved and saved["probability_fix"]["policy"] != PROBABILITY_POLICY:
        raise ValueError("Resume probability policy differs")
    frames = saved["frames"]
    if len(frames) > len(rows) or status.get("phase") not in ("preparing", "completed"):
        raise ValueError("Invalid cache progress")
    seen = {}
    for index, frame in enumerate(frames):
        row = rows[index]
        path = (source_base / row["image"]).resolve()
        expected_relative = f"maps/{row['split']}_{path.stem}.npz"
        if (frame["source"] != row["image"] or frame["role"] != row["split"] or
            frame["group"] != str(row["group"]) or frame["filename"] != path.name or
            frame["stem"] != path.stem or frame["condition"] != expected_relative or
            sha256(path) != frame["source_sha256"] or
            sha256(output / expected_relative) != frame["condition_sha256"]):
            raise ValueError(f"Saved source/cache changed: {path.name}")
        with np.load(output / expected_relative, allow_pickle=False) as data:
            rgb, edge = data["RGB_uint8"], data["E"]
            if (rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8 or edge.shape != rgb.shape[:2] or
                frame["native_size"] != [rgb.shape[1], rgb.shape[0]] or
                not np.isfinite(edge).all() or np.any((edge < 0) | (edge > 1))):
                raise ValueError(f"Invalid saved RGB/E: {path.name}")
            identity = hashlib.sha256(rgb.tobytes()).hexdigest()
        if identity in seen and seen[identity] != row["split"]:
            raise ValueError("Decoded RGB duplicate leaks into validation")
        seen[identity] = row["split"]
    saved["selection"] = plan
    saved.setdefault("probability_fix", dict(policy=PROBABILITY_POLICY, first_frame=len(frames) + 1))
    return saved, seen


def balanced(rows, limit):
    buckets = defaultdict(list)
    for row in sorted(rows, key=lambda r:r["image"]):
        buckets[str(row["group"])].append(row)
    rng = random.Random(42); keys = sorted(buckets); rng.shuffle(keys)
    for key in keys:
        rng.shuffle(buckets[key])
    result = []
    while len(result) < min(limit, len(rows)):
        for key in keys:
            if buckets[key] and len(result) < limit:
                result.append(buckets[key].pop())
    return sorted(result, key=lambda r:r["image"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-manifest", type=Path, default=ROOT / "runs/learned_color/data_seed42_v1/cycle_manifest.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RGB)
    parser.add_argument("--train-limit", type=int, default=256)
    parser.add_argument("--val-limit", type=int, default=12)
    parser.add_argument("--teacher-dir", type=Path)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Verify and continue an existing RGB cache")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if min(args.train_limit, args.val_limit) < 1:
        parser.error("Use positive limits")
    if args.output_dir.exists() and not args.resume:
        parser.error("Preserve existing cache; use matching --resume or a new output-dir")
    if args.resume and not args.output_dir.is_dir():
        parser.error("No existing cache to resume")
    # Reuse D's validation of every path and cross-modality video partition.
    select_sources(args.data_manifest, limit=1)
    data = read_json(args.data_manifest)
    rows = []
    for role, limit in (("train", args.train_limit), ("val", args.val_limit)):
        chosen = balanced([r for r in data["domains"]["A"] if r["split"] == role], limit)
        if not chosen:
            parser.error(f"Missing RGB {role}")
        rows.extend(chosen)
    import torch
    import torch.nn.functional as F
    from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
    torch.set_num_threads(4)
    if args.teacher_dir:
        snapshot = args.teacher_dir
    else:
        from huggingface_hub import snapshot_download
        snapshot = Path(snapshot_download(TEACHER, revision=REVISION, cache_dir=str(DEFAULT_CACHE),
                         local_files_only=not args.allow_download,
                         allow_patterns=["config.json", "preprocessor_config.json", "pytorch_model.bin", "README.md"]))
    processor = AutoImageProcessor.from_pretrained(snapshot, local_files_only=True,
                                                   size={"height":512, "width":512})
    with torch.random.fork_rng(devices=[]):
        teacher = AutoModelForSemanticSegmentation.from_pretrained(snapshot, local_files_only=True,
                                     trust_remote_code=False, weights_only=True).eval().requires_grad_(False).to(args.device)
    names = [str(teacher.config.id2label[i]).lower() for i in range(teacher.config.num_labels)]
    sys.path.insert(0, str(ROOT / "pytorch-CycleGAN-and-pix2pix"))
    from models.clarity_semantic_edges import clarity_components, CLARITY_SETTINGS
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=args.resume); (output / "maps").mkdir(exist_ok=args.resume)
    teacher_files = {p.name:sha256(p) for p in Path(snapshot).iterdir() if p.name in ("config.json", "preprocessor_config.json", "pytorch_model.bin")}
    result = dict(format="thermal_D_rgb_selfpairs_v1", seed=42, manifest_sha256=sha256(args.data_manifest),
                  teacher=dict(model=TEACHER, revision=REVISION, files=teacher_files, short_side="processor 512x512",
                               flip_tta=False, mapping=CITY_TO_FMB), contour_variant="clarity_v1", settings=CLARITY_SETTINGS,
                  code_sha256={p:sha256(ROOT / p) for p in EDGE_SOURCES}, frames=[],
                  limits="RGB grayscale is a proxy, not real TIR; no paired TIR/RGB or TIR-domain training")
    if args.resume:
        result, seen = resume_cache(output, result, rows, args.data_manifest.parent)
        print(f"Verified {len(result['frames'])}/{len(rows)} cached images; continuing without rewriting them", flush=True)
    else:
        seen = {}
        result["selection"] = [dict(source=r["image"], role=r["split"], group=str(r["group"])) for r in rows]
        result["probability_fix"] = dict(policy=PROBABILITY_POLICY, first_frame=1)
    write_json(output / "manifest.json", result)
    start = len(result["frames"])
    for index, row in enumerate(rows[start:], start + 1):
        path = (args.data_manifest.parent / row["image"]).resolve()
        with Image.open(path) as im:
            rgb = im.convert("RGB"); values = np.asarray(rgb).copy(); gray = np.asarray(rgb.convert("L")).copy()
        identity = hashlib.sha256(values.tobytes()).hexdigest()
        if identity in seen and seen[identity] != row["split"]:
            raise ValueError("Decoded RGB duplicate leaks into validation")
        seen[identity] = row["split"]
        with torch.inference_mode():
            inputs = processor(images=rgb, return_tensors="pt").to(args.device)
            logits = teacher(**inputs).logits.float()
            p = F.interpolate(logits, size=gray.shape, mode="bilinear", align_corners=False).softmax(1)[0].cpu().numpy()
        p = map_probabilities(p, names)
        image = ((gray.astype(np.float32) / 255 - .5) / .5 + 1) / 2
        maps, _ = clarity_components(image, p, include_probes=False)
        relative = f"maps/{row['split']}_{path.stem}.npz"
        np.savez_compressed(output / relative, RGB_uint8=values, E=maps["clarity_v1"])
        result["frames"].append(dict(stem=path.stem, filename=path.name, group=str(row["group"]),
             role=row["split"], native_size=list(rgb.size), source=row["image"], source_sha256=sha256(path),
             condition=relative, condition_sha256=sha256(output / relative)))
        write_json(output / "manifest.json", result)
        write_json(output / "status.json", dict(phase="preparing", completed=index, total=len(rows)))
        print(f"RGB E [{index}/{len(rows)}] {row['split']} {path.name}", flush=True)
    write_json(output / "status.json", dict(phase="completed", completed=len(rows), total=len(rows)))
    print(output / "manifest.json", flush=True)


if __name__ == "__main__":
    main()
