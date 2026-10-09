"""One architecture for the supervised and local-contrastive controls."""

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.GroupNorm(4, out_channels), nn.SiLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.GroupNorm(4, out_channels), nn.SiLU(),
    )


class SoftContourNet(nn.Module):
    """Full-resolution skip features, only two downsampling stages.

    Input is grayscale TIR in [-1, 1]. Both variants include the same projection
    head; it is only used by the contrastive objective in the second variant.
    Sigmoid outputs are boundary scores, not calibrated correctness confidence.
    """

    def __init__(self, width=24, embedding_dim=32):
        super().__init__()
        if width < 4 or width % 4 or embedding_dim < 1:
            raise ValueError("width must be a positive multiple of 4; embedding_dim > 0")
        self.config = {"width": width, "embedding_dim": embedding_dim}
        self.enc1 = block(1, width)
        self.enc2 = block(width, width * 2)
        self.enc3 = block(width * 2, width * 4)
        self.dec2 = block(width * 6, width * 2)
        self.dec1 = block(width * 3, width)
        self.edge_head = nn.Conv2d(width, 1, 1)
        self.projection = nn.Sequential(
            nn.Conv2d(width, width, 1), nn.SiLU(),
            nn.Conv2d(width, embedding_dim, 1),
        )

    def forward(self, image):
        if image.ndim != 4 or image.shape[1] != 1 or min(image.shape[-2:]) < 4:
            raise ValueError("Expected [N, 1, H, W], H and W >= 4")
        a = self.enc1(image)
        b = self.enc2(F.avg_pool2d(a, 2))
        c = self.enc3(F.avg_pool2d(b, 2))
        # Nearest upsampling + trainable convolutions keeps this path compatible
        # with deterministic CUDA training, including interpolation backward.
        d = self.dec2(torch.cat((F.interpolate(c, size=b.shape[-2:], mode="nearest"), b), 1))
        features = self.dec1(torch.cat((F.interpolate(d, size=a.shape[-2:], mode="nearest"), a), 1))
        return self.edge_head(features), features


def transform_view(tensor, code, inverse=False):
    """Exact flip/90-degree rotation; supports rectangular images, no resampling.

    code = rotation + 4 * horizontal_flip. Undo rotation before undoing flip.
    """
    if not 0 <= code < 8:
        raise ValueError("Transform code must be in [0, 7]")
    if inverse:
        tensor = torch.rot90(tensor, -(code % 4), (-2, -1))
        return tensor.flip(-1) if code >= 4 else tensor
    tensor = tensor.flip(-1) if code >= 4 else tensor
    return torch.rot90(tensor, code % 4, (-2, -1))


class FrozenContourExtractor:
    """Standalone frozen inference API, compatible with edge_override.

    This plain wrapper is deliberately not registered inside a GAN. Call it on
    the *already cropped/flipped* TIR tensor and pass its output as edge_override.
    Existing train.py is not changed or automatically wired to this wrapper.
    """

    def __init__(self, checkpoint, device="cpu"):
        self.device = torch.device(device)
        state = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
        if state.get("format") != "learned_contours_v1":
            raise ValueError("Unsupported contour checkpoint format")
        self.metadata = state["metadata"]
        self.model = SoftContourNet(**state["model_config"])
        self.model.load_state_dict(state["model"], strict=True)
        self.model.to(self.device).eval().requires_grad_(False)

    @torch.no_grad()
    def __call__(self, tir):
        if not torch.is_floating_point(tir) or not torch.isfinite(tir).all():
            raise ValueError("TIR must be a finite float tensor in [-1, 1]")
        if tir.min() < -1.0001 or tir.max() > 1.0001:
            raise ValueError("TIR must be normalized to [-1, 1]")
        logits, _ = self.model(tir.to(self.device))
        return logits.sigmoid().to(device=tir.device, dtype=tir.dtype)
