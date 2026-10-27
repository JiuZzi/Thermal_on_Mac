"""CPU protocol helpers; importing D never downloads a model."""

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "D/configs/sd15_clarity_seed42.json"
DEFAULT_CONDITIONS = ROOT / "runs/D/conditions/val_seed42"
DEFAULT_OUTPUT = ROOT / "runs/D/previews/val_seed42_v1"
DEFAULT_CACHE = ROOT / "runs/D/model_cache"
GEOMETRY = "existing FLIR_datasets pixels; no recrop; model resize only, output restored to source size"
EDGE_SOURCES = (
    "pytorch-CycleGAN-and-pix2pix/models/clarity_semantic_edges.py",
    "pytorch-CycleGAN-and-pix2pix/models/selective_semantic_edges.py",
    "pytorch-CycleGAN-and-pix2pix/models/semantic_edges.py",
    "pytorch-CycleGAN-and-pix2pix/models/saliency_edges.py",
    "pytorch-CycleGAN-and-pix2pix/models/semantic_contour.py",
    "scripts/semantic_preview/checkpoint.py",
    "scripts/semantic_preview/inference.py",
    "scripts/semantic_preview/model.py",
    "scripts/semantic_preview/_vendor/mix_transformer.py",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    pending = path.with_name(path.name + ".pending")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)


def load_config(path=DEFAULT_CONFIG):
    config = read_json(path)
    if config.get("format") != "thermal_D_preview_v1" or config.get("scheduler") != "DDIM":
        raise ValueError("Expected the D v1 DDIM preview configuration")
    for key in ("seed", "width", "height", "steps"):
        if type(config.get(key)) is not int or config[key] < (0 if key == "seed" else 1):
            raise ValueError(f"Invalid integer {key}")
    if config["width"] % 64 or config["height"] % 64:
        raise ValueError("Use model dimensions divisible by 64")
    for key in ("strength", "guidance_scale", "control_scale", "eta"):
        value = config.get(key)
        if not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError(f"Invalid finite {key}")
    if not 0 < config["strength"] <= 1 or int(config["steps"] * config["strength"]) < 1:
        raise ValueError("strength * steps must provide at least one denoising step")
    if config["guidance_scale"] < 1 or config["control_scale"] <= 0 or config["eta"] != 0:
        raise ValueError("Use guidance >= 1, positive D1 control and deterministic DDIM eta=0")
    for key in ("base_model", "control_model", "prompt"):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError(f"Missing {key}")
    for key in ("base_revision", "control_revision"):
        value = config.get(key, "")
        if len(value) != 40 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("Pin both upstream models to full commit hashes")
    if not isinstance(config.get("negative_prompt"), str):
        raise ValueError("negative_prompt must be a string")
    return config


def select_sources(manifest_path, phase="val", limit=12, seed=42):
    """Keep the split; balanced deterministic preview across video groups."""
    manifest_path = Path(manifest_path).resolve()
    data = read_json(manifest_path)
    if data.get("format") != "grouped_unaligned_v1" or set(data.get("domains", {})) != {"A", "B"}:
        raise ValueError("Use the existing grouped A/B cycle_manifest.json")
    if phase not in ("train", "val", "test") or limit < 0:
        raise ValueError("Invalid phase or limit (0 means all images)")
    groups, paths, counts = {}, set(), {}
    for domain, rows in data["domains"].items():
        counts[domain] = dict(Counter(row["split"] for row in rows))
        for row in rows:
            role, group = row.get("split"), str(row.get("group", ""))
            if role not in ("train", "val", "test") or not group:
                raise ValueError("Missing split/group")
            if group in groups and groups[group] != role:
                raise ValueError("Video group leaks across roles or modalities")
            groups[group] = role
            path = (manifest_path.parent / row["image"]).resolve()
            if path in paths or not path.is_file():
                raise ValueError(f"Missing or repeated source path: {path}")
            paths.add(path)
    candidates = []
    for row in data["domains"]["B"]:
        if row["split"] == phase:
            path = (manifest_path.parent / row["image"]).resolve()
            candidates.append(dict(filename=path.name, stem=path.stem, group=str(row["group"]),
                                   source=row["image"], path=path))
    if not candidates or len({r["stem"] for r in candidates}) != len(candidates):
        raise ValueError("Missing TIR split or colliding output stems")
    candidates.sort(key=lambda row: row["filename"])
    if limit and limit < len(candidates):
        buckets = defaultdict(list)
        for row in candidates:
            buckets[row["group"]].append(row)
        rng = random.Random(seed)
        names = sorted(buckets)
        rng.shuffle(names)
        for name in names:
            rng.shuffle(buckets[name])
        chosen = []
        while len(chosen) < limit:
            for name in names:
                if buckets[name] and len(chosen) < limit:
                    chosen.append(buckets[name].pop())
        candidates = sorted(chosen, key=lambda row: row["filename"])
    return candidates, counts


def control_tensor(edge, size):
    """Replicate soft E and resize; preserve [0,1] amplitude without thresholding."""
    import torch
    import torch.nn.functional as F
    edge = np.asarray(edge, dtype=np.float32)
    if edge.ndim != 2 or not edge.size or not np.isfinite(edge).all() or np.any((edge < 0) | (edge > 1)):
        raise ValueError("Expected a finite nonempty soft E map in [0,1]")
    tensor = torch.from_numpy(edge.copy())[None, None]
    return F.interpolate(tensor, size=(size[1], size[0]), mode="bilinear", align_corners=False).repeat(1, 3, 1, 1)


def read_condition(folder, frame):
    path = Path(folder) / frame["condition"]
    if sha256(path) != frame["condition_sha256"]:
        raise ValueError(f"Condition cache changed: {path}")
    with np.load(path, allow_pickle=False) as values:
        gray, edge = values["TIR_uint8"].copy(), values["E"].copy()
    if gray.ndim != 2 or gray.dtype != np.uint8 or edge.shape != gray.shape:
        raise ValueError("Cached TIR and E must have identical native geometry")
    if not np.isfinite(edge).all() or np.any((edge < 0) | (edge > 1)):
        raise ValueError("Cached soft E must be finite and in [0,1]")
    if frame["native_size"] != [gray.shape[1], gray.shape[0]]:
        raise ValueError("Native image size differs from condition manifest")
    return Image.fromarray(gray).convert("RGB"), edge
