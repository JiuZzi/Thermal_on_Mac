"""Build labeled geometric toys and exercise both training/inference variants.

This is an implementation check, NOT a FLIR edge-quality benchmark.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from contour_learning.model import FrozenContourExtractor, SoftContourNet
from preview_learned_contours import parse_args as preview_args, run as preview
from train_learned_contours import parse_args as train_args, run as train, state_hash


def make_synthetic_data(folder):
    folder.mkdir(parents=True, exist_ok=False)
    rows = []
    rng = np.random.default_rng(417)
    for i in range(6):
        h, w = 32, 40
        mask = np.zeros((h, w), dtype=bool)
        # One larger object and one 4x6-pixel small object, no resize.
        mask[8:24, 6 + i:17 + i] = True
        mask[12:18, 29:33] = True
        eroded = mask.copy()
        eroded[1:] &= mask[:-1]
        eroded[:-1] &= mask[1:]
        eroded[:, 1:] &= mask[:, :-1]
        eroded[:, :-1] &= mask[:, 1:]
        label = np.where(mask & ~eroded, 255, 0).astype(np.uint8)
        label[:4] = 128
        gray = np.uint8(np.clip(0.2 + 0.55 * mask + rng.normal(0, 0.025, mask.shape), 0, 1) * 255)
        image, edge = f"toy_{i}.png", f"toy_{i}_edge.png"
        Image.fromarray(gray).save(folder / image)
        Image.fromarray(label).save(folder / edge)
        rows.append({"image": image, "edge": edge, "split": "train" if i < 4 else "val",
                     "video_id": f"synthetic_train_{i // 2}" if i < 4 else "synthetic_val"})
    annotations = folder / "annotations.csv"
    with annotations.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("image", "edge", "split", "video_id"))
        writer.writeheader()
        writer.writerows(rows)
    preview_dir = folder / "preview_inputs"
    preview_dir.mkdir()
    for row in rows[-2:]:
        with Image.open(folder / row["image"]) as image:
            image.save(preview_dir / row["image"])
    return annotations, preview_dir


def run(output_dir):
    if output_dir.exists():
        raise FileExistsError("Use a new smoke output directory to preserve previous evidence")
    torch.set_num_threads(1)
    annotations, images = make_synthetic_data(output_dir / "data")
    for variant in ("supervised", "contrastive"):
        train(train_args([
            "--annotations", str(annotations), "--variant", variant,
            "--output-dir", str(output_dir / variant), "--synthetic-smoke",
            "--n-epochs", "1", "--n-epochs-decay", "0", "--crop-size", "32",
            "--width", "8", "--embedding-dim", "8", "--device", "cpu",
            "--edge-crop-probability", "1", "--max-anchors", "32", "--max-negatives", "64",
        ]))
    checkpoints = [torch.load(output_dir / variant / "latest.pt", weights_only=True)
                   for variant in ("supervised", "contrastive")]
    a, b = checkpoints
    assert a["metadata"]["initial_model_sha256"] == b["metadata"]["initial_model_sha256"]
    assert a["metadata"]["dataset_sha256"] == b["metadata"]["dataset_sha256"]
    assert b["history"][0]["active_anchors"] > 0
    torch.manual_seed(2026)
    initial = SoftContourNet(8, 8)
    assert state_hash(initial) == a["metadata"]["initial_model_sha256"]
    initial_weights = initial.state_dict()
    for checkpoint in checkpoints:
        assert not torch.equal(initial_weights["edge_head.weight"], checkpoint["model"]["edge_head.weight"])
        assert not torch.equal(initial_weights["enc1.0.weight"], checkpoint["model"]["enc1.0.weight"])
    for key in initial_weights:
        if key.startswith("projection."):
            assert torch.equal(initial_weights[key], a["model"][key])
    assert not torch.equal(initial_weights["projection.2.weight"], b["model"]["projection.2.weight"])
    extractor = FrozenContourExtractor(output_dir / "contrastive/latest.pt")
    scores = extractor(torch.zeros(1, 1, 31, 37, requires_grad=True))
    assert scores.shape == (1, 1, 31, 37) and not scores.requires_grad
    assert torch.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all()
    preview(preview_args([
        "--checkpoint-a", str(output_dir / "supervised/latest.pt"),
        "--checkpoint-b", str(output_dir / "contrastive/latest.pt"),
        "--image-dir", str(images), "--output-dir", str(output_dir / "preview"),
    ]))
    report = {"purpose": "synthetic_implementation_check", "passed": True,
              "checks": ["same initial weights and dataset", "both task gradients update encoder/head",
                         "contrastive anchors active and projection updated only in CL variant",
                         "frozen odd-size inference preserves shape and range", "two-variant preview exported"],
              "not_verified": ["real FLIR boundary quality", "small-person boundary recall",
                               "CycleGAN use of contour channel", "RGB improvement"]}
    (output_dir / "smoke_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"PASS: synthetic implementation checks; report={output_dir / 'smoke_report.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args().output_dir)
