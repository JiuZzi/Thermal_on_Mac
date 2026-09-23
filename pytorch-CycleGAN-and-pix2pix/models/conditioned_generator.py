"""A small, inspectable condition interface for the TIR-to-RGB generator."""

import torch
from torch import nn


class ConditionedGenerator(nn.Module):
    """Adapt [TIR, edge, trusted edge] to the original one-channel generator.

    The zero mode is the C0-cap control: both condition maps are identically
    zero for real TIR inputs and for the synthetic TIR used in the cycle path.
    """

    def __init__(self, backbone: nn.Module, condition_mode: str):
        super().__init__()
        if condition_mode != "zero":
            raise ValueError(f"Unsupported condition mode: {condition_mode}")
        self.condition_mode = condition_mode
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
        raise ValueError(f"Unsupported condition mode: {self.condition_mode}")

    def forward(self, tir: torch.Tensor) -> torch.Tensor:
        if tir.ndim != 4 or tir.shape[1] != 1:
            raise ValueError(f"Expected one-channel TIR tensor [N, 1, H, W], got {tuple(tir.shape)}")
        edge, trusted_edge = self.make_condition(tir)
        adapted_tir = self.adapter(torch.cat((tir, edge, trusted_edge), dim=1))
        return self.backbone(adapted_tir)
