"""Strict partial-label interface and identical sampling for both variants."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_annotations(csv_path, protocol=None, processed_dir=None):
    """Reject video leakage, duplicate images, invalid labels and geometry.

    With protocol supplied, every annotated image must match a processed trainB
    file, including its content hash. testB is not allowed for model selection.
    """
    csv_path = Path(csv_path).resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing annotation CSV: {csv_path}; no Canny labels are substituted")
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"image", "edge", "split", "video_id"}.issubset(reader.fieldnames or []):
            raise ValueError("CSV requires image,edge,split,video_id")
        rows = list(reader)
    if not rows:
        raise ValueError("Annotation CSV is empty")
    allowed = None
    if protocol is not None:
        if processed_dir is None:
            raise ValueError("processed_dir is required with the FLIR protocol")
        with Path(protocol).open(encoding="utf-8-sig", newline="") as stream:
            allowed = {row["destination_name"]: row["video_id"] for row in csv.DictReader(stream)
                       if row["destination_split"] == "trainB"}
    seen, videos, counts = set(), {"train": set(), "val": set()}, {"train": [0, 0], "val": [0, 0]}
    records = []
    for row in rows:
        if row["split"] not in videos or not row["video_id"].strip():
            raise ValueError("split must be train/val, and video_id must be nonempty")
        record = {key: row[key] for key in ("split", "video_id")}
        for key in ("image", "edge"):
            path = Path(row[key])
            path = path if path.is_absolute() else csv_path.parent / path
            record[key] = path.resolve()
            if not record[key].is_file():
                raise FileNotFoundError(record[key])
        image_hash = file_hash(record["image"])
        if record["image"] in seen or any(r["image_sha256"] == image_hash for r in records):
            raise ValueError(f"Duplicate image/path/content: {record['image']}")
        seen.add(record["image"])
        if allowed is not None:
            name = record["image"].name
            expected = Path(processed_dir) / name
            if allowed.get(name) != row["video_id"] or not expected.is_file():
                raise ValueError(f"Image is not in processed FLIR trainB with this video_id: {name}")
            if image_hash != file_hash(expected):
                raise ValueError(f"Image differs from processed trainB: {name}")
        with Image.open(record["image"]) as image, Image.open(record["edge"]) as edge:
            if image.mode != "L" or edge.mode != "L" or image.size != edge.size:
                raise ValueError("Image/label must be 8-bit grayscale L with identical geometry")
            if min(image.size) < 4:
                raise ValueError("Image dimensions must be >= 4")
            labels = np.asarray(edge)
            if not np.isin(labels, (0, 128, 255)).all():
                raise ValueError(f"Invalid label values in {record['edge']}; use 0/128/255 only")
            if not np.any(labels != 128):
                raise ValueError(f"All pixels are unknown: {record['edge']}")
            counts[row["split"]][0] += int(np.count_nonzero(labels == 0))
            counts[row["split"]][1] += int(np.count_nonzero(labels == 255))
        videos[row["split"]].add(row["video_id"])
        record.update(image_sha256=image_hash, edge_sha256=file_hash(record["edge"]))
        records.append(record)
    if videos["train"] & videos["val"]:
        raise ValueError("Train/val video overlap; split entire videos before annotation")
    if any(min(counts[split]) == 0 for split in counts):
        raise ValueError("Both train and val need explicit boundary AND non-boundary pixels")
    # Paths excluded so a dataset copied from Mac to Windows keeps its identity.
    identity = [{"image_name": r["image"].name, "split": r["split"], "video_id": r["video_id"],
                 "image_sha256": r["image_sha256"], "edge_sha256": r["edge_sha256"]} for r in records]
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return records, {"dataset_sha256": fingerprint,
                     "protocol_sha256": file_hash(protocol) if protocol else None,
                     "annotations": identity,
                     "pixel_counts": counts,
                     "train_videos": sorted(videos["train"]), "val_videos": sorted(videos["val"])}


class ContourDataset(Dataset):
    def __init__(self, records, split, crop_size=256, seed=2026, edge_crop_probability=0.5):
        self.records = [r for r in records if r["split"] == split]
        self.split, self.crop_size, self.seed = split, crop_size, seed
        self.edge_crop_probability, self.epoch = edge_crop_probability, 0
        if crop_size < 4 or not 0 <= edge_crop_probability <= 1:
            raise ValueError("Invalid crop settings")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(record["image"]) as image, Image.open(record["edge"]) as edge:
            gray, labels = np.array(image, dtype=np.float32) / 127.5 - 1, np.array(edge)
        if self.split == "train":
            # Stateless sampler: contrastive RNG cannot change crop choices.
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
            size = self.crop_size
            pad = ((0, max(0, size - gray.shape[0])), (0, max(0, size - gray.shape[1])))
            gray = np.pad(gray, pad, mode="edge")
            labels = np.pad(labels, pad, constant_values=128)
            h, w = gray.shape
            edge_points = np.argwhere(labels == 255)
            points = edge_points if len(edge_points) and rng.random() < self.edge_crop_probability else None
            if points is not None:
                y, x = points[rng.integers(len(points))]
                top = int(rng.integers(max(0, y - size + 1), min(y, h - size) + 1))
                left = int(rng.integers(max(0, x - size + 1), min(x, w - size) + 1))
            else:
                top, left = int(rng.integers(h - size + 1)), int(rng.integers(w - size + 1))
            if np.all(labels[top:top + size, left:left + size] == 128):
                points = np.argwhere(labels != 128)
                y, x = points[rng.integers(len(points))]
                top, left = min(max(0, y - size // 2), h - size), min(max(0, x - size // 2), w - size)
            gray = gray[top:top + size, left:left + size].copy()
            labels = labels[top:top + size, left:left + size].copy()
        return torch.from_numpy(gray[None]), torch.from_numpy(labels[None].copy())
