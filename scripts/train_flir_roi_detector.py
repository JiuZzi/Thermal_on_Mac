#!/usr/bin/env python3
"""Fine-tune a thermal person/vehicle detector on FLIR trainB only.

The detector is an auxiliary, explicitly supervised C2c component. Validation
holds out entire trainB videos; FLIR testB is never used for model selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import ssdlite320_mobilenet_v3_large


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VAL_VIDEOS = ("5An82wT7iBmfZSwx7", "5YffDt2oYT6CDzYHk")
# COCO-pretrained class IDs: person, bicycle, car, motorcycle, bus, train, truck.
# FLIR-specific rider and other-vehicle labels map to the nearest target class.
LABEL_MAP = {1: 1, 2: 2, 3: 3, 4: 4, 6: 6, 7: 7, 8: 8, 74: 1, 79: 3}
TARGET_LABELS = frozenset(LABEL_MAP.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "FLIR_datasets/trainB")
    parser.add_argument("--manifest", type=Path, default=ROOT / "FLIR_protocol_v2/manifest.csv")
    parser.add_argument("--coco", type=Path, default=ROOT / "FLIR_ADAS_v2/images_thermal_train/coco.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/detectors/flir_roi_ssdlite")
    parser.add_argument("--initial-weights", type=Path,
                        help="Official COCO SSDLite .pth; omit to download through TorchVision")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--max-train-images", type=int, default=None,
                        help="Fixed random subset for a short pilot; omit for full trainB")
    parser.add_argument("--max-val-images", type=int, default=None)
    parser.add_argument("--val-videos", nargs="+", default=DEFAULT_VAL_VIDEOS)
    parser.add_argument("--score-threshold", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 2 or args.learning_rate <= 0:
        parser.error("epochs >= 1, batch-size >= 2, and learning-rate > 0 are required")
    if not 0 < args.score_threshold < 1:
        parser.error("score-threshold must be in (0, 1)")
    for name in ("max_train_images", "max_val_images"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def load_records(manifest: Path, coco: Path) -> list[dict]:
    data = json.loads(coco.read_text(encoding="utf-8"))
    image_by_name = {Path(image["file_name"]).name: image for image in data["images"]}
    annotations = defaultdict(list)
    for annotation in data["annotations"]:
        if annotation["category_id"] in LABEL_MAP:
            annotations[annotation["image_id"]].append(annotation)
    with manifest.open(newline="", encoding="utf-8") as handle:
        selected = [row for row in csv.DictReader(handle) if row["destination_split"] == "trainB"]
    records = []
    for row in selected:
        image = image_by_name[row["destination_name"]]
        records.append({
            "filename": row["destination_name"],
            "video_id": row["video_id"],
            "source_width": image["width"],
            "source_height": image["height"],
            "annotations": annotations[image["id"]],
        })
    return records


def processed_boxes(record: dict) -> tuple[torch.Tensor, torch.Tensor]:
    boxes = []
    labels = []
    for item in record["annotations"]:
        x, y, width, height = item["bbox"]
        x1 = max(0.0, min(360.0, x * 500 / record["source_width"] - 70))
        y1 = max(0.0, min(288.0, y * 400 / record["source_height"] - 56))
        x2 = max(0.0, min(360.0, (x + width) * 500 / record["source_width"] - 70))
        y2 = max(0.0, min(288.0, (y + height) * 400 / record["source_height"] - 56))
        if x2 - x1 >= 2 and y2 - y1 >= 2:
            boxes.append((x1, y1, x2, y2))
            labels.append(LABEL_MAP[item["category_id"]])
    return torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4), torch.tensor(labels, dtype=torch.int64)


class FlirRoiDataset(Dataset):
    def __init__(self, image_dir: Path, records: list[dict]):
        self.image_dir = image_dir
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        with Image.open(self.image_dir / record["filename"]) as image:
            gray = np.asarray(image.convert("L"), dtype=np.float32).copy() / 255.0
        if gray.shape != (288, 360):
            raise ValueError(f"Expected processed FLIR image 360x288: {record['filename']}")
        image = torch.from_numpy(gray).unsqueeze(0).repeat(3, 1, 1)
        boxes, labels = processed_boxes(record)
        return image, {"boxes": boxes, "labels": labels}


def select_records(records: list[dict], val_videos: set[str], seed: int,
                   max_train: int | None, max_val: int | None) -> tuple[list[dict], list[dict]]:
    train = [item for item in records if item["video_id"] not in val_videos]
    val = [item for item in records if item["video_id"] in val_videos]
    if not train or not val:
        raise ValueError("Train and video-held-out validation sets must both be nonempty")
    rng = random.Random(seed)
    if max_train is not None:
        train = sorted(rng.sample(train, min(max_train, len(train))), key=lambda x: x["filename"])
    if max_val is not None:
        val = sorted(rng.sample(val, min(max_val, len(val))), key=lambda x: x["filename"])
    return train, val


def make_model(initial_weights: Path | None) -> torch.nn.Module:
    if initial_weights is None:
        from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights
        return ssdlite320_mobilenet_v3_large(
            weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT,
        )
    model = ssdlite320_mobilenet_v3_large(
        weights=None, weights_backbone=None, num_classes=91,
    )
    model.load_state_dict(torch.load(initial_weights, map_location="cpu", weights_only=True))
    return model


def collate(batch):
    return tuple(zip(*batch))


def box_iou(a: torch.Tensor, b: torch.Tensor) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    return intersection / max(area_a + area_b - intersection, 1e-8)


@torch.inference_mode()
def evaluate(model: torch.nn.Module, loader: DataLoader, score_threshold: float) -> dict[str, float | int]:
    model.eval()
    true_positive = false_positive = false_negative = 0
    for images, targets in loader:
        predictions = model(list(images))
        for prediction, target in zip(predictions, targets):
            selected = [index for index, score in enumerate(prediction["scores"])
                        if float(score) >= score_threshold and int(prediction["labels"][index]) in TARGET_LABELS]
            used_gt: set[int] = set()
            for pred_index in selected:
                label = int(prediction["labels"][pred_index])
                matches = [
                    (box_iou(prediction["boxes"][pred_index], target["boxes"][gt_index]), gt_index)
                    for gt_index in range(len(target["boxes"]))
                    if gt_index not in used_gt and int(target["labels"][gt_index]) == label
                ]
                if matches:
                    best_iou, best_index = max(matches)
                    if best_iou >= 0.5:
                        true_positive += 1
                        used_gt.add(best_index)
                        continue
                false_positive += 1
            false_negative += len(target["boxes"]) - len(used_gt)
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision_at_iou_0_5": true_positive / max(true_positive + false_positive, 1),
        "recall_at_iou_0_5": true_positive / max(true_positive + false_negative, 1),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    records = load_records(args.manifest, args.coco)
    train_records, val_records = select_records(
        records, set(args.val_videos), args.seed,
        args.max_train_images, args.max_val_images,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_loader = DataLoader(
        FlirRoiDataset(args.processed_dir, train_records), batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers, collate_fn=collate,
    )
    val_loader = DataLoader(
        FlirRoiDataset(args.processed_dir, val_records), batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, collate_fn=collate,
    )
    model = make_model(args.initial_weights).cpu()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0001)
    history = []
    baseline = evaluate(model, val_loader, args.score_threshold)
    print(f"COCO initial detector on held-out FLIR trainB videos: {baseline}", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch_index, (images, targets) in enumerate(train_loader, start=1):
            components = model(list(images), list(targets))
            loss = sum(components.values())
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite detector loss at epoch {epoch}, batch {batch_index}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            if batch_index % 25 == 0:
                print(f"epoch {epoch} batch {batch_index}/{len(train_loader)} loss={np.mean(losses[-25:]):.4f}", flush=True)
        metrics = evaluate(model, val_loader, args.score_threshold)
        row = {"epoch": epoch, "mean_train_loss": float(np.mean(losses)), **metrics}
        precision = metrics["precision_at_iou_0_5"]
        recall = metrics["recall_at_iou_0_5"]
        row["f1_at_iou_0_5"] = 2 * precision * recall / max(precision + recall, 1e-8)
        history.append(row)
        print(f"epoch {epoch}: {row}", flush=True)
        torch.save({"state_dict": model.state_dict(), "epoch": epoch, "metrics": metrics},
                   args.output_dir / "latest.pt")
        if metrics["recall_at_iou_0_5"] >= max(
            [entry["recall_at_iou_0_5"] for entry in history[:-1]], default=-1.0,
        ):
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "metrics": metrics},
                       args.output_dir / "best_recall.pt")
        if row["f1_at_iou_0_5"] >= max(
            [entry["f1_at_iou_0_5"] for entry in history[:-1]], default=-1.0,
        ):
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "metrics": metrics},
                       args.output_dir / "best_f1.pt")
    manifest = {
        "architecture": "torchvision_ssdlite320_mobilenet_v3_large_coco_91_classes",
        "source": "FLIR trainB thermal only",
        "train_images": len(train_records),
        "val_images": len(val_records),
        "val_video_ids": sorted(args.val_videos),
        "testB_used": False,
        "label_map": LABEL_MAP,
        "score_threshold": args.score_threshold,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "initial_weights_sha256": file_sha256(args.initial_weights) if args.initial_weights else None,
        "initial_validation": baseline,
        "history": history,
        "note": "Box detection measures approximate object localization, not pixel contour quality.",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved detector: {args.output_dir / 'best_f1.pt'}", flush=True)


if __name__ == "__main__":
    main()
