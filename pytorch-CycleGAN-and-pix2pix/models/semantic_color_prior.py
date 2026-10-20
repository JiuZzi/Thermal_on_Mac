"""A small, explicit color-family prior; not recovered ground-truth RGB.

The frozen FMB scores retain all 15 classes. Only four classes get broad
chroma envelopes. The top-two margin is a semantic separation proxy, never
an edge confidence or a calibrated correctness probability.
"""

import copy

import torch
from torch import nn

from .semantic_edges import FMB_CLASS_NAMES


COLOR_CONFIG = {
    "version": "manual_color_families_v1",
    "source": "human-specified sRGB swatches; no paired FLIR color supervision",
    "classes": list(FMB_CLASS_NAMES),
    "families": {
        "road": {"rgb": [[100, 104, 106], [133, 134, 133], [158, 154, 143]], "tolerance_ab": [9, 10]},
        "sidewalk": {"rgb": [[133, 131, 124], [167, 160, 142], [190, 186, 175]], "tolerance_ab": [11, 12]},
        "vegetation": {"rgb": [[78, 108, 57], [123, 140, 74], [139, 115, 73]], "tolerance_ab": [13, 15]},
        "sky": {"rgb": [[104, 157, 200], [145, 180, 203], [183, 193, 198]], "tolerance_ab": [13, 15]},
    },
    "strength": "(top1-top2)*supported_probability_mass; zero if top1 unsupported",
    "features": ["gated_mean_a/128", "gated_mean_b/128", "gated_std_a/128", "gated_std_b/128", "strength", "eligible_semantic_margin"],
    "loss": "nearest color-family ab envelope; no target luminance, no saturation maximization",
}


def color_config():
    return copy.deepcopy(COLOR_CONFIG)


def rgb_to_lab(rgb):
    """Differentiable sRGB [N,3,H,W] in [0,1] to D65 CIELAB."""
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError("Expected RGB [N,3,H,W]")
    rgb = rgb.float().clamp(0, 1)
    linear = torch.where(rgb > .04045, ((rgb + .055) / 1.055).pow(2.4), rgb / 12.92)
    matrix = rgb.new_tensor(((.412453, .357580, .180423),
                             (.212671, .715160, .072169),
                             (.019334, .119193, .950227)))
    xyz = torch.einsum("ij,njhw->nihw", matrix, linear)
    xyz = xyz / rgb.new_tensor((.95047, 1., 1.08883))[None, :, None, None]
    # Clamp the unused cube-root branch too, so black pixels have finite grads.
    transformed = torch.where(xyz > .008856, xyz.clamp_min(1e-12).pow(1 / 3),
                              7.787 * xyz + 16 / 116)
    x, y, z = transformed.unbind(1)
    return torch.stack((116 * y - 16, 500 * (x - y), 200 * (y - z)), 1)


class SemanticColorPrior(nn.Module):
    feature_channels = 6

    def __init__(self):
        super().__init__()
        palette = torch.full((15, 3, 3), .5)
        tolerances = torch.full((15, 2), 32.)
        supported = torch.zeros(15, dtype=torch.bool)
        for name, family in COLOR_CONFIG["families"].items():
            index = FMB_CLASS_NAMES.index(name)
            palette[index] = torch.tensor(family["rgb"]) / 255.
            tolerances[index] = torch.tensor(family["tolerance_ab"])
            supported[index] = True
        lab = rgb_to_lab(palette.reshape(45, 3, 1, 1)).reshape(15, 3, 3)
        self.register_buffer("palette_rgb", palette)
        self.register_buffer("palette_ab", lab[:, :, 1:])
        self.register_buffer("tolerance_ab", tolerances)
        self.register_buffer("supported", supported)

    def forward(self, probabilities):
        p = probabilities.detach().float()
        if p.ndim != 4 or p.shape[1] != 15 or min(p.shape[-2:]) < 1:
            raise ValueError("Expected semantic probabilities [N,15,H,W]")
        if (not torch.isfinite(p).all() or (p < 0).any() or (p > 1).any()
                or not torch.allclose(p.sum(1), torch.ones_like(p[:, 0]), atol=4e-6, rtol=0)):
            raise ValueError("Invalid categorical semantic probabilities")
        top = p.topk(2, dim=1)
        labels = top.indices[:, 0]
        margin = top.values[:, :1] - top.values[:, 1:2]
        eligible = self.supported[labels].unsqueeze(1).float()
        selected = p * self.supported[None, :, None, None]
        mass = selected.sum(1, keepdim=True)
        weights = selected * margin * eligible
        strength = weights.sum(1, keepdim=True)
        means = self.palette_ab.mean(1)
        moments = (self.palette_ab.square().mean(1) + self.tolerance_ab.square())
        mean = torch.einsum("nchw,cd->ndhw", selected, means) / mass.clamp_min(1e-8)
        second = torch.einsum("nchw,cd->ndhw", selected, moments) / mass.clamp_min(1e-8)
        std = (second - mean.square()).clamp_min(0).sqrt()
        mean, std = mean * strength, std * strength
        features = torch.cat((mean / 128, std / 128, strength, margin * eligible), 1)
        # Display swatches, not a colorized photograph; unsupported classes gray.
        hint = self.palette_rgb.mean(1)[labels].permute(0, 3, 1, 2)
        hint = hint * strength + .5 * (1 - strength)
        return dict(labels=labels, margin=margin, eligible=eligible, supported_mass=mass,
                    strength=strength, class_weights=weights, features=features, hint_rgb=hint)

    def chroma_loss(self, generated, condition):
        """Soft ab-range penalty; class uncertainty reduces the absolute loss."""
        if generated.ndim != 4 or generated.shape[1] != 3:
            raise ValueError("Expected generated RGB [N,3,H,W] in [-1,1]")
        if generated.shape[0] != condition["class_weights"].shape[0] or generated.shape[-2:] != condition["class_weights"].shape[-2:]:
            raise ValueError("Color condition and output geometry differ")
        ab = rgb_to_lab((generated + 1) / 2)[:, 1:]
        penalty = ab[:, 0] * 0
        for index in self.supported.nonzero().flatten().tolist():
            distances = (ab[:, None] - self.palette_ab[index][None, :, :, None, None]).abs()
            outside = (distances - self.tolerance_ab[index][None, None, :, None, None]).relu()
            nearest = outside.sum(2).amin(1) / 128
            penalty = penalty + nearest * condition["class_weights"][:, index]
        # Do not divide by sum(strength), which would cancel uncertainty gating.
        count = condition["eligible"].flatten(1).sum(1).clamp_min(1)
        return (penalty.flatten(1).sum(1) / count).mean()
