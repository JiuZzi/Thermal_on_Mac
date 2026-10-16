"""Author-compatible grayscale preparation and optional separate detail probe."""

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

CLASS_NAMES = ("background", "road", "sidewalk", "building", "lamp", "sign",
               "vegetation", "sky", "person", "car", "truck", "bus", "motorcycle", "bicycle", "pole")
CLASS_LABELS = ("背景", "道路", "路侧/人行道", "建筑", "灯", "标志", "植被", "天空",
                "行人", "汽车", "卡车", "公交车", "摩托车", "自行车", "杆")
PALETTE = np.array(((0, 0, 0), (179, 228, 228), (181, 57, 133), (67, 162, 177),
                    (200, 178, 50), (132, 45, 199), (66, 172, 84), (179, 73, 79),
                    (76, 99, 166), (66, 121, 253), (137, 165, 91), (155, 97, 152),
                    (105, 153, 140), (222, 215, 158), (135, 113, 90)), dtype=np.uint8)
MEAN = np.array((123.675, 116.28, 103.53), np.float32)
STD = np.array((58.395, 57.12, 57.375), np.float32)


def preprocess(gray, short_side=512):
    gray = np.asarray(gray)
    if gray.ndim != 2 or gray.dtype != np.uint8 or min(gray.shape) < 1:
        raise ValueError("Expected a nonempty processed uint8 grayscale image")
    if short_side < 32:
        raise ValueError("Short side must be >= 32")
    image = np.repeat(gray[..., None], 3, axis=2)
    h, w = gray.shape
    if min(h, w) < short_side:
        scale = float(short_side) / min(h, w)
        image = np.asarray(Image.fromarray(image).resize((int(scale * w), int(scale * h)), Image.Resampling.BILINEAR))
    normalized = (image.astype(np.float32) - MEAN) / STD
    tensor = torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1)))[None]
    return tensor, tuple(image.shape[:2])


@torch.inference_mode()
def _infer_probabilities(model, gray, device, short_side=512, flip_tta=False):
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("Semantic review requires an eval-mode frozen model")
    tensor, input_size = preprocess(gray, short_side)
    tensor = tensor.to(device)
    logits = model(tensor)
    if logits.ndim != 4 or logits.shape[:2] != (1, 15) or not torch.isfinite(logits).all():
        raise ValueError("Expected finite [1,15,H,W] semantic logits")
    # Exactly match upstream: interpolate logits with align_corners=True, then softmax.
    resized = F.interpolate(logits.float(), size=gray.shape, mode="bilinear", align_corners=True)
    probabilities = resized.softmax(1)
    if flip_tta:
        second = model(tensor.flip(-1)).float()
        second = F.interpolate(second, size=gray.shape, mode="bilinear", align_corners=True).flip(-1).softmax(1)
        probabilities = (probabilities + second) * 0.5
    probabilities = probabilities[0].float().cpu().numpy()
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(0), 1, atol=2e-6):
        raise ValueError("Invalid categorical soft responses")
    return probabilities, input_size, tuple(logits.shape[-2:])


def predict_probabilities(model, gray, device, short_side=512, flip_tta=False):
    """Frozen inference for training; omit labels and diagnostic score maps."""
    return _infer_probabilities(model, gray, device, short_side, flip_tta)[0]


def predict(model, gray, device, short_side=512, flip_tta=False):
    """Frozen review inference, including the original diagnostic outputs."""
    probabilities, input_size, logit_size = _infer_probabilities(
        model, gray, device, short_side, flip_tta,
    )
    labels = probabilities.argmax(0).astype(np.uint8)
    # Diagnostic uncertainty only; neither entropy nor max score is edge confidence.
    entropy = -(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum(0) / np.log(15)
    return dict(probabilities=probabilities, labels=labels, normalized_entropy=entropy.astype(np.float32),
                max_score=probabilities.max(0).astype(np.float32), input_size=input_size,
                logit_size=logit_size, flip_tta=flip_tta)
