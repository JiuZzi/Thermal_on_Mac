"""Lazy, SHA-checked formal-D conditions. Training never consumes test rows."""

import io
import hashlib
from pathlib import Path

import numpy as np
import torch

from D.common import read_json, sha256
from .protocol import validate_arrays


class ConditionDataset:
    def __init__(self, folder, domain, role):
        self.folder = Path(folder)
        self.manifest = read_json(self.folder / "manifest.json")
        if (self.manifest.get("format") != "D_unpaired_conditions_v1" or
            read_json(self.folder / "status.json").get("phase") != "completed"):
            raise ValueError("Prepare completed formal-D conditions first")
        self.frames = [r for r in self.manifest["frames"] if r["domain"] == domain and r["role"] == role]
        if not self.frames:
            raise ValueError(f"Missing {domain} {role} conditions; no fallback to another split")

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        frame = self.frames[index]
        payload = (self.folder / frame["condition"]).read_bytes()
        if hashlib.sha256(payload).hexdigest() != frame["condition_sha256"]:
            raise ValueError(f"Changed condition: {frame['id']}")
        with np.load(io.BytesIO(payload), allow_pickle=False) as cached:
            gray, edge, p, q, rgb = [cached[k].copy() for k in ("gray", "E", "P", "Q", "RGB")]
        validate_arrays(gray, edge, p, q)
        if rgb.shape != (*gray.shape, 3) or rgb.dtype != np.uint8:
            raise ValueError("Invalid cached RGB")
        tir = torch.from_numpy(gray.astype(np.float32) / 255)[None, None]
        e = torch.from_numpy(edge)[None, None]
        return dict(x=torch.cat((tir, e, torch.zeros_like(e)), 1), p=torch.from_numpy(p)[None],
                    q=torch.from_numpy(q)[None, None], rgb=torch.from_numpy(rgb.transpose(2,0,1).copy())[None].float()/127.5-1,
                    frame=frame)


def on_device(sample, device):
    return {key:value.to(device) if torch.is_tensor(value) else value for key,value in sample.items()}


def data_identity(folder):
    path = Path(folder) / "manifest.json"
    data = read_json(path)
    groups = {}
    for row in data["frames"]:
        group, role = row["group"], row["role"]
        if group in groups and groups[group] != role:
            raise ValueError("Video group leaks across roles")
        groups[group] = role
    return dict(manifest_sha256=sha256(path), source=data["identity"])
