"""Grouped manifests and bounded, balanced RGB chroma sampling."""

from collections import defaultdict
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.color import rgb2lab


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else hashlib.sha256(handle.read()).hexdigest()


def write_json(path, data):
    path = Path(path)
    if path.exists():
        raise ValueError(f"Preserve existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def load_manifest(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    names = manifest.get("class_names", [])
    if (manifest.get("format") != "rgb_chroma_manifest_v1" or not isinstance(names,list) or not names
            or any(not isinstance(n,str) or not n for n in names)
            or len(names) != len(set(names)) or len(names) >= 255
            or not isinstance(manifest.get('dataset_id'),str) or not manifest['dataset_id']):
        raise ValueError("Invalid RGB chroma manifest/class names")
    rows = manifest.get("images", [])
    ids, paths, assignments = set(), {}, {}
    for row in rows:
        if (row.get("group") in (None, "") or not isinstance(row.get("id"), str)
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", row["id"]) or row["id"] in ids):
            raise ValueError("Each image requires a unique ID and explicit video/scene group")
        ids.add(row["id"])
        if row.get("split") not in ("train", "val", "test"):
            raise ValueError("Expected explicit train/val/test roles")
        group = str(row["group"])
        if group in assignments and assignments[group] != row["split"]:
            raise ValueError("Group leaks across splits")
        assignments[group] = row["split"]
        for key in ("image", "label"):
            resolved = (path.parent / row[key]).resolve()
            if key == "image" and resolved in paths:
                raise ValueError("Image path reused in manifest")
            if key == "image":
                paths[resolved] = row["id"]
    if not rows:
        raise ValueError("Empty RGB manifest")
    return manifest, path.parent


def sample_manifest(path, image_cap=256, group_cap=1024, seed=42, roles=("train", "val")):
    if image_cap < 1 or group_cap < 1:
        raise ValueError("Sampling caps must be positive")
    manifest, base = load_manifest(path)
    rng = np.random.default_rng(seed)
    pools, priorities = {}, {}
    identities, content_roles = [], {}
    for row in sorted(manifest["images"], key=lambda r:r["id"]):
        if row["split"] not in roles:
            continue
        image_path, label_path = ((base / row[key]).resolve() for key in ("image", "label"))
        with Image.open(image_path) as handle:
            rgb = np.asarray(handle.convert("RGB"))
        with Image.open(label_path) as handle:
            labels = np.asarray(handle)
        if labels.ndim != 2 or labels.shape != rgb.shape[:2] or not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("Labels must be original-coordinate integer class IDs")
        if not set(np.unique(labels)).issubset({*range(len(manifest["class_names"])), 255}):
            raise ValueError("Label IDs differ from the canonical manifest mapping")
        content = hashlib.sha256(str(rgb.shape).encode()+rgb.tobytes()).hexdigest()
        if content in content_roles and content_roles[content] != row["split"]:
            raise ValueError("Decoded RGB duplicate leaks across roles, even with another filename")
        if content in content_roles:
            raise ValueError("Repeated decoded RGB: deduplicate before prior fitting")
        content_roles[content] = row["split"]
        identities.append(dict(id=row["id"], group=row["group"], split=row["split"],
                               image_sha256=digest(image_path), label_sha256=digest(label_path), rgb_content_sha256=content))
        ab = rgb2lab(rgb)[..., 1:].astype(np.float64)
        for index, name in enumerate(manifest["class_names"]):
            candidates = ab[labels == index]
            if not len(candidates):
                continue
            candidates = candidates[rng.choice(len(candidates), min(image_cap, len(candidates)), replace=False)]
            key = (row["split"], name, str(row["group"]))
            priority = rng.random(len(candidates))
            if key in pools:
                candidates = np.concatenate((pools[key], candidates))
                priority = np.concatenate((priorities[key], priority))
            keep = np.argsort(priority)[:group_cap]
            pools[key], priorities[key] = candidates[keep], priority[keep]
    return manifest, pools, identities


def balanced_groups(pools, split, name, min_group_pixels=32):
    selected = {group:values for (role, class_name, group),values in pools.items()
                if role == split and class_name == name and len(values) >= min_group_pixels}
    if not selected:
        return {}, np.empty((0, 2))
    # Exactly equal samples per retained group; a long video cannot dominate EM.
    count = min(len(values) for values in selected.values())
    selected = {group:values[:count] for group,values in sorted(selected.items())}
    return selected, np.concatenate(list(selected.values()))
