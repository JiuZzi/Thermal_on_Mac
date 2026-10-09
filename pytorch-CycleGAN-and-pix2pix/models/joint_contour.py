"""Label-free soft-contour refinement and local contour-map contrast.

Canny/saliency supplies a heuristic starting map, not boundary ground truth.
Both experiment variants have exactly the same architecture.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .conditioned_generator import ConditionedGenerator


def transform_view(tensor, code, inverse=False):
    if not 0 <= code < 8:
        raise ValueError("Expected flip/90-degree rotation code in [0,7]")
    if inverse:
        tensor = torch.rot90(tensor, -(code % 4), (-2, -1))
        return tensor.flip(-1) if code >= 4 else tensor
    tensor = tensor.flip(-1) if code >= 4 else tensor
    return torch.rot90(tensor, code % 4, (-2, -1))


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.GroupNorm(4, out_channels), nn.SiLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.GroupNorm(4, out_channels), nn.SiLU(),
    )


class ContourRefiner(nn.Module):
    """Two downsamplings + full-resolution skips, output a logit correction."""

    def __init__(self, width=16, epsilon=0.02, residual_limit=6.0):
        super().__init__()
        if width < 4 or width % 4 or not 0 < epsilon < 0.1 or residual_limit <= 0:
            raise ValueError("Invalid contour width/epsilon/residual limit")
        self.epsilon, self.residual_limit = epsilon, residual_limit
        self.enc1 = conv_block(2, width)
        self.enc2 = conv_block(width, width * 2)
        self.enc3 = conv_block(width * 2, width * 4)
        self.dec2 = conv_block(width * 6, width * 2)
        self.dec1 = conv_block(width * 3, width)
        self.residual_head = nn.Conv2d(width, 1, 1)
        self.reset_to_prior()

    def reset_to_prior(self):
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def forward(self, tir, prior):
        a = self.enc1(torch.cat((tir, prior), 1))
        b = self.enc2(F.avg_pool2d(a, 2))
        c = self.enc3(F.avg_pool2d(b, 2))
        d = self.dec2(torch.cat((F.interpolate(c, size=b.shape[-2:], mode="nearest"), b), 1))
        features = self.dec1(torch.cat((F.interpolate(d, size=a.shape[-2:], mode="nearest"), a), 1))
        residual = self.residual_limit * torch.tanh(self.residual_head(features) / self.residual_limit)
        base = prior.clamp(self.epsilon, 1 - self.epsilon)
        edge = (torch.logit(base) + residual).sigmoid()
        return edge, residual


class JointContourGenerator(ConditionedGenerator):
    """[TIR, learned soft contour, zero] -> original RGB backbone.

    The projection head reads ONLY the refined contour map. Local contrast
    therefore has a direct autograd path into the contour output head.
    """

    def __init__(self, backbone, condition_mode, contour_width=16, embedding_dim=16,
                 epsilon=0.02, residual_limit=6.0, **condition_kwargs):
        if condition_mode not in ("soft_multi", "soft_saliency"):
            raise ValueError("Joint contour experiments use soft_multi (C2b) or soft_saliency (C2c)")
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be positive")
        super().__init__(backbone, condition_mode, fusion_mode="direct", **condition_kwargs)
        self.refiner = ContourRefiner(contour_width, epsilon, residual_limit)
        # Same projection parameters in both variants, unused by the plain loss.
        self.projection = nn.Sequential(nn.Conv2d(1, embedding_dim, 5, padding=2),
                                        nn.SiLU(), nn.Conv2d(embedding_dim, embedding_dim, 1))

    def forward(self, tir, return_details=False, contour_only=False,
                prior_override=None, condition_override=None, project=False):
        if tir.ndim != 4 or tir.shape[1] != 1 or min(tir.shape[-2:]) < 4:
            raise ValueError("Expected TIR [N,1,H,W], dimensions >= 4")
        prior = self.make_condition(tir)[0] if prior_override is None else prior_override
        if prior.shape != tir.shape:
            raise ValueError("Prior must have the same geometry as TIR")
        prior = prior.detach().to(tir)
        if not torch.isfinite(prior).all() or prior.min() < 0 or prior.max() > 1:
            raise ValueError("Prior must be finite and in [0,1]")
        edge, residual = self.refiner(tir, prior)
        details = {"prior": prior, "edge": edge, "residual": residual}
        if project:
            details["embedding"] = self.projection(edge)
        if not contour_only:
            condition = edge
            if condition_override == "zero":
                condition = torch.zeros_like(edge)
            elif condition_override == "shift":
                condition = edge.roll((max(1, edge.shape[-2] // 4), max(1, edge.shape[-1] // 4)), (-2, -1))
            elif condition_override == "prior":
                condition = prior
            elif condition_override not in (None, "normal"):
                raise ValueError("Unsupported contour intervention")
            details["rgb"] = self.backbone(torch.cat((tir, condition, torch.zeros_like(tir)), 1))
        if contour_only or return_details:
            return details
        return details["rgb"]


@torch.no_grad()
def heuristic_anchors(tir, prior, positive_threshold=0.6, background_threshold=0.02,
                      gradient_threshold=0.02, exclusion_radius=3):
    """Heuristic positives / low-gradient distant background; remaining unknown.

    A missing Canny candidate is NOT automatically a negative. These anchors
    can still be wrong; they are not semantic masks or calibrated confidence.
    """
    gray = (tir.detach() + 1) / 2
    dx = F.pad((gray[..., 1:] - gray[..., :-1]).abs(), (0, 1, 0, 0))
    dy = F.pad((gray[..., 1:, :] - gray[..., :-1, :]).abs(), (0, 0, 0, 1))
    # Include differences on both sides of a pixel.
    gradient = torch.maximum(torch.maximum(dx, F.pad(dx[..., :-1], (1, 0, 0, 0))),
                             torch.maximum(dy, F.pad(dy[..., :-1, :], (0, 0, 1, 0))))
    nearby = F.max_pool2d(prior, 2 * exclusion_radius + 1, 1, exclusion_radius)
    positive = prior >= positive_threshold
    negative = (prior <= background_threshold) & (nearby <= 0.1) & (gradient <= gradient_threshold)
    return positive, negative


def anchor_loss(edge, prior, positive, negative, epsilon=0.02):
    target = prior.detach().clamp(epsilon, 1 - epsilon)
    squared = (edge - target).square()
    terms = [squared[mask].mean() for mask in (positive, negative) if mask.any()]
    return torch.stack(terms).mean() if terms else edge.sum() * 0


def local_map_contrast(embedding_a, embedding_b, positive, negative, generator,
                       max_anchors=64, max_negatives=128, temperature=0.1,
                       exclusion_radius=3):
    """Same coordinate across aligned views positive, heuristic background negative."""
    if embedding_a.shape != embedding_b.shape or embedding_a.shape[-2:] != positive.shape[-2:]:
        raise ValueError("Local embeddings and masks must be aligned")
    width = positive.shape[-1]
    losses, active = [], 0
    for i in range(embedding_a.shape[0]):
        p = torch.nonzero(positive[i, 0].flatten()).flatten().cpu()
        n = torch.nonzero(negative[i, 0].flatten()).flatten().cpu()
        if not len(p) or not len(n):
            continue
        p = p[torch.randperm(len(p), generator=generator)[:max_anchors]].to(embedding_a.device)
        n = n[torch.randperm(len(n), generator=generator)[:max_negatives]].to(embedding_a.device)
        distance = torch.maximum((p[:, None] // width - n[None, :] // width).abs(),
                                 (p[:, None] % width - n[None, :] % width).abs())
        usable = distance > exclusion_radius
        keep = usable.any(1)
        if not keep.any():
            continue
        p, usable = p[keep], usable[keep]
        a = F.normalize(embedding_a[i].flatten(1).T, dim=1, eps=1e-6)
        b = F.normalize(embedding_b[i].flatten(1).T, dim=1, eps=1e-6)
        for source, destination in ((a, b), (b, a)):
            pos = (source[p] * destination[p]).sum(1, keepdim=True)
            neg = (source[p] @ destination[n].T).masked_fill(~usable, -torch.inf)
            logits = torch.cat((pos, neg), 1) / temperature
            losses.append(F.cross_entropy(logits, torch.zeros(len(p), device=logits.device, dtype=torch.long)))
        active += len(p)
    return (torch.stack(losses).mean(), active) if losses else ((embedding_a.sum() + embedding_b.sum()) * 0, 0)
