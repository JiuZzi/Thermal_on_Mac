"""Inspectable hard and soft edge conditions for the TIR-to-RGB generator."""

from typing import Optional

import numpy as np
import torch
from torch import nn


class ConditionedGenerator(nn.Module):
    """Pass [TIR, edge, trusted edge] through the selected fusion interface.

    Zero is the C0-cap control; Canny is the C1 hard-edge condition.
    Soft-single spreads the same Canny edges over a fixed pixel width.
    Soft-multi averages maps from fixed Canny high thresholds. Soft-saliency
    uses a spatial importance prior to blend fine and coarse candidates.
    The reserved third channel stays zero in all modes. Direct fusion keeps all three
    channels for the generator's first convolution.
    """

    def __init__(
        self,
        backbone: nn.Module,
        condition_mode: str,
        edge_sigma: float = 1.0,
        edge_low_threshold: float = 0.08,
        edge_high_threshold: float = 0.16,
        edge_soft_width: float = 1.0,
        soft_high_thresholds: tuple[float, ...] = (0.12, 0.16, 0.20),
        fusion_mode: str = "adapter",
        saliency_sigmas: tuple[float, float, float] = (0.7, 1.0, 1.6),
        saliency_inner_weight: float = 0.8,
        saliency_outer_weight: float = 0.2,
        saliency_transition_fraction: float = 24.0 / 256.0,
        saliency_background_gain: float = 0.5,
    ):
        super().__init__()
        if condition_mode not in ("zero", "canny", "soft_single", "soft_multi", "soft_saliency"):
            raise ValueError(f"Unsupported condition mode: {condition_mode}")
        if edge_sigma <= 0 or not (0 <= edge_low_threshold < edge_high_threshold <= 1):
            raise ValueError("Canny requires sigma > 0 and 0 <= low < high <= 1")
        if edge_soft_width <= 0:
            raise ValueError("Soft edge width must be positive")
        if len(soft_high_thresholds) < 2 or any(
            not (0 < threshold <= 1) for threshold in soft_high_thresholds
        ) or tuple(sorted(set(soft_high_thresholds))) != tuple(soft_high_thresholds):
            raise ValueError("Soft multi thresholds must contain at least two increasing values in (0, 1]")
        if not any(np.isclose(threshold, edge_high_threshold) for threshold in soft_high_thresholds):
            raise ValueError("Soft multi thresholds must include the C1 high threshold")
        if fusion_mode not in ("adapter", "direct"):
            raise ValueError(f"Unsupported fusion mode: {fusion_mode}")
        if condition_mode == "soft_saliency" and fusion_mode != "direct":
            raise ValueError("C2c requires direct fusion")
        if len(saliency_sigmas) != 3 or any(sigma <= 0 for sigma in saliency_sigmas):
            raise ValueError("Saliency sigmas must contain three positive values")
        if tuple(sorted(saliency_sigmas)) != tuple(saliency_sigmas):
            raise ValueError("Saliency sigmas must be increasing")
        if not (0 <= saliency_outer_weight <= saliency_inner_weight <= 1):
            raise ValueError("Saliency weights must satisfy 0 <= outer <= inner <= 1")
        if not (0 < saliency_transition_fraction < 1 / 3):
            raise ValueError("Saliency transition fraction must be in (0, 1/3)")
        if not (0 < saliency_background_gain <= 1):
            raise ValueError("Saliency background gain must be in (0, 1]")
        self.condition_mode = condition_mode
        self.fusion_mode = fusion_mode
        self.edge_sigma = edge_sigma
        self.edge_low_threshold = edge_low_threshold
        self.edge_high_threshold = edge_high_threshold
        self.edge_soft_width = edge_soft_width
        self.soft_high_thresholds = soft_high_thresholds
        self.saliency_sigmas = tuple(saliency_sigmas)
        self.saliency_inner_weight = saliency_inner_weight
        self.saliency_outer_weight = saliency_outer_weight
        self.saliency_transition_fraction = saliency_transition_fraction
        self.saliency_background_gain = saliency_background_gain
        self.adapter = nn.Conv2d(3, 1, kernel_size=3, padding=1) if fusion_mode == "adapter" else None
        self.backbone = backbone

    def reset_adapter_to_identity(self) -> None:
        """Initially pass TIR through unchanged, despite the extra interface."""
        if self.adapter is None:
            raise RuntimeError("Direct fusion has no adapter to reset")
        with torch.no_grad():
            self.adapter.weight.zero_()
            self.adapter.bias.zero_()
            self.adapter.weight[0, 0, 1, 1] = 1.0

    def make_condition(
        self, tir: torch.Tensor, saliency_override: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if saliency_override is not None and self.condition_mode != "soft_saliency":
            raise ValueError("Saliency override requires soft_saliency mode")
        if self.condition_mode == "zero":
            return torch.zeros_like(tir), torch.zeros_like(tir)
        if self.condition_mode in ("canny", "soft_single", "soft_multi", "soft_saliency"):
            try:
                from skimage.feature import canny
            except ImportError as error:
                raise ImportError(
                    "Edge conditions require scikit-image in the training/test Python environment: "
                    "python -m pip install scikit-image"
                ) from error
            if self.condition_mode != "canny":
                try:
                    from scipy.ndimage import distance_transform_edt
                except ImportError as error:
                    raise ImportError("Soft edges require scipy: python -m pip install scipy") from error
            # The loader has already resized, cropped, flipped, and normalized
            # the TIR image. Derive edges here so they match the actual input.
            # The same rule also applies to synthetic TIR in the cycle path.
            gray = np.clip((tir.detach().float().cpu().numpy()[:, 0] + 1.0) / 2.0, 0.0, 1.0)
            low_ratio = self.edge_low_threshold / self.edge_high_threshold

            def hard_edge(image: np.ndarray, high: float) -> np.ndarray:
                return canny(
                    image,
                    sigma=self.edge_sigma,
                    low_threshold=high * low_ratio,
                    high_threshold=high,
                )

            def soft_edge(hard: np.ndarray) -> np.ndarray:
                if not hard.any():
                    return np.zeros(hard.shape, dtype=np.float32)
                distance = distance_transform_edt(~hard)
                return np.exp(-0.5 * (distance / self.edge_soft_width) ** 2).astype(np.float32)

            if self.condition_mode == "canny":
                edge_maps = np.stack([hard_edge(image, self.edge_high_threshold) for image in gray])
            elif self.condition_mode == "soft_single":
                edge_maps = np.stack([soft_edge(hard_edge(image, self.edge_high_threshold)) for image in gray])
            elif self.condition_mode == "soft_multi":
                edge_maps = np.stack([
                    np.mean(
                        [soft_edge(hard_edge(image, high)) for high in self.soft_high_thresholds],
                        axis=0,
                        dtype=np.float32,
                    )
                    for image in gray
                ])
            else:
                from .saliency_edges import center_importance, make_saliency_edge

                if saliency_override is not None:
                    if saliency_override.shape != tir.shape:
                        raise ValueError("Saliency override must have shape [N, 1, H, W] matching TIR")
                    saliency_maps = saliency_override.detach().float().cpu().numpy()[:, 0]
                else:
                    prior = center_importance(
                        gray.shape[1], gray.shape[2],
                        self.saliency_inner_weight, self.saliency_outer_weight,
                        self.saliency_transition_fraction,
                    )
                    saliency_maps = np.broadcast_to(prior, gray.shape)
                edge_maps = np.stack([
                    make_saliency_edge(
                        image, self.soft_high_thresholds, low_ratio,
                        self.edge_soft_width, self.saliency_sigmas, importance,
                        self.saliency_background_gain,
                    )[0]
                    for image, importance in zip(gray, saliency_maps)
                ])
            edge = torch.from_numpy(edge_maps[:, None]).to(device=tir.device, dtype=tir.dtype)
            return edge, torch.zeros_like(edge)
        raise ValueError(f"Unsupported condition mode: {self.condition_mode}")

    def forward(
        self, tir: torch.Tensor, edge_override: Optional[torch.Tensor] = None,
        saliency_override: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if tir.ndim != 4 or tir.shape[1] != 1:
            raise ValueError(f"Expected one-channel TIR tensor [N, 1, H, W], got {tuple(tir.shape)}")
        if edge_override is not None and saliency_override is not None:
            raise ValueError("Pass either edge override or saliency override, not both")
        if edge_override is None:
            edge, trusted_edge = self.make_condition(tir, saliency_override=saliency_override)
        else:
            if edge_override.shape != tir.shape:
                raise ValueError("Edge override must have the same [N, 1, H, W] shape as the edge map")
            edge = edge_override.to(device=tir.device, dtype=tir.dtype)
            trusted_edge = torch.zeros_like(tir)
        conditioned_tir = torch.cat((tir, edge, trusted_edge), dim=1)
        if self.adapter is not None:
            conditioned_tir = self.adapter(conditioned_tir)
        return self.backbone(conditioned_tir)
