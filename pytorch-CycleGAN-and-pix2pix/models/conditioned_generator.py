"""A small, inspectable condition interface for the TIR-to-RGB generator."""

import numpy as np
import torch
from torch import nn


class ConditionedGenerator(nn.Module):
    """Adapt [TIR, edge, trusted edge] to the original one-channel generator.

    The zero mode is the C0-cap control. Canny mode is the C1 hard-edge
    control. In both modes, the trusted-edge channel stays zero.
    """

    def __init__(
        self,
        backbone: nn.Module,
        condition_mode: str,
        edge_sigma: float = 1.0,
        edge_low_threshold: float = 0.08,
        edge_high_threshold: float = 0.16,
    ):
        super().__init__()
        if condition_mode not in ("zero", "canny"):
            raise ValueError(f"Unsupported condition mode: {condition_mode}")
        if edge_sigma <= 0 or not (0 <= edge_low_threshold < edge_high_threshold <= 1):
            raise ValueError("Canny requires sigma > 0 and 0 <= low < high <= 1")
        self.condition_mode = condition_mode
        self.edge_sigma = edge_sigma
        self.edge_low_threshold = edge_low_threshold
        self.edge_high_threshold = edge_high_threshold
        self.adapter = nn.Conv2d(3, 1, kernel_size=3, padding=1)
        self.backbone = backbone

    def reset_adapter_to_identity(self) -> None:
        """Initially pass TIR through unchanged, despite the extra interface."""
        with torch.no_grad():
            self.adapter.weight.zero_()
            self.adapter.bias.zero_()
            self.adapter.weight[0, 0, 1, 1] = 1.0

    def make_condition(self, tir: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.condition_mode == "zero":
            return torch.zeros_like(tir), torch.zeros_like(tir)
        if self.condition_mode == "canny":
            try:
                from skimage.feature import canny
            except ImportError as error:
                raise ImportError(
                    "C1 requires scikit-image in the training/test Python environment: "
                    "python -m pip install scikit-image"
                ) from error
            # The loader has already resized, cropped, flipped, and normalized
            # the TIR image. Derive edges here so they match the actual input.
            # The same rule also applies to synthetic TIR in the cycle path.
            gray = np.clip((tir.detach().float().cpu().numpy()[:, 0] + 1.0) / 2.0, 0.0, 1.0)
            edge_maps = np.stack(
                [
                    canny(
                        image,
                        sigma=self.edge_sigma,
                        low_threshold=self.edge_low_threshold,
                        high_threshold=self.edge_high_threshold,
                    )
                    for image in gray
                ]
            )
            edge = torch.from_numpy(edge_maps[:, None]).to(device=tir.device, dtype=tir.dtype)
            return edge, torch.zeros_like(edge)
        raise ValueError(f"Unsupported condition mode: {self.condition_mode}")

    def forward(self, tir: torch.Tensor) -> torch.Tensor:
        if tir.ndim != 4 or tir.shape[1] != 1:
            raise ValueError(f"Expected one-channel TIR tensor [N, 1, H, W], got {tuple(tir.shape)}")
        edge, trusted_edge = self.make_condition(tir)
        adapted_tir = self.adapter(torch.cat((tir, edge, trusted_edge), dim=1))
        return self.backbone(adapted_tir)
