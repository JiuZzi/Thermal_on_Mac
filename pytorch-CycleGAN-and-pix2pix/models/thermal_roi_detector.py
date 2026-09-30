"""Frozen FLIR thermal detector and image-aligned soft object ROI maps."""

from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from torchvision.models.detection import ssdlite320_mobilenet_v3_large


TARGET_LABELS = frozenset((1, 2, 3, 4, 6, 7, 8))


class ThermalRoiDetector:
    """CPU inference is intentional: Canny conditions are also made on CPU."""

    def __init__(self, checkpoint: str | Path):
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"Missing thermal ROI detector checkpoint: {path}")
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if "state_dict" not in saved:
            raise ValueError("Expected a train_flir_roi_detector.py checkpoint")
        self.model = ssdlite320_mobilenet_v3_large(
            weights=None, weights_backbone=None, num_classes=91,
        )
        self.model.load_state_dict(saved["state_dict"], strict=True)
        self.model.cpu().eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.checkpoint = path

    @torch.inference_mode()
    def predict(self, gray: np.ndarray, score_threshold: float = 0.10) -> list[tuple[tuple[float, ...], float, int]]:
        if gray.ndim != 2 or not np.isfinite(gray).all() or np.any((gray < 0) | (gray > 1)):
            raise ValueError("Expected a finite [0, 1] grayscale TIR image")
        image = torch.from_numpy(gray.astype(np.float32, copy=True)).unsqueeze(0).repeat(3, 1, 1)
        output = self.model([image])[0]
        return [
            (tuple(float(value) for value in box), float(score), int(label))
            for box, score, label in zip(output["boxes"], output["scores"], output["labels"])
            if float(score) >= score_threshold and int(label) in TARGET_LABELS
        ]


def boxes_to_importance(
    shape: tuple[int, int],
    detections: list[tuple[tuple[float, ...], float, int]],
    floor: float = 0.1,
    shoulder_px: float = 5.0,
) -> np.ndarray:
    """Soft full-box regions; box borders themselves are never edge labels."""
    height, width = shape
    inside = np.zeros((height, width), dtype=bool)
    for box, _, _ in detections:
        x1, y1, x2, y2 = box
        left, top = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
        right, bottom = min(width, int(np.ceil(x2))), min(height, int(np.ceil(y2)))
        if right > left and bottom > top:
            inside[top:bottom, left:right] = True
    if not inside.any():
        return np.full(shape, floor, dtype=np.float32)
    distance = distance_transform_edt(~inside)
    return (floor + (1.0 - floor) * np.exp(-0.5 * (distance / shoulder_px) ** 2)).astype(np.float32)
