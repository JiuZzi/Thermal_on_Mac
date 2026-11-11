"""Opt-in D-v3 appearance objectives; no additional pretrained models or C channel."""

import hashlib
import math

import torch
from torch.nn import functional as F

from .losses import gray01, gradients

FORMAT = "D_unpaired_v3"
APPEARANCE_LUMA_POLICY = "smooth_field_v1"
APPEARANCE_REGION_POLICY = "class_balanced_lab_v1"
APPEARANCE_SYNTHETIC_TIR = "synthetic_tir"
APPEARANCE_DISTRIBUTION = "appearance_distribution"
APPEARANCE_REGION_BLEND = .25
APPEARANCE_REGION_MIN_QUALITY = .5
APPEARANCE_REGION_MIN_MASS = 16.
EXTRA_LOSSES = {"rgb_appearance", "decoder_range", "color_attraction", "edge_precision"}
QUALITY_KEYS = {
    "decoder_policy", "boundary_context", "appearance_every", "appearance_rgb",
    "appearance_chroma", "appearance_detail", "source_residual_rms_limit",
    "gradient_probe_every", "split_gradient_clip", "structure_tolerance", "structure_min_quality",
}


def _lab(rgb):
    from .semantics import legacy_models
    legacy_models()
    from models.semantic_color_prior import rgb_to_lab
    return rgb_to_lab(rgb)


def bounded_decode(raw, *, straight_through=False):
    """Identical legal RGB values; optional identity backward at saturation.

    The separate raw range penalty directly corrects out-of-range values.
    """
    raw = raw.float()
    bounded = raw.clamp(-1, 1)
    output = bounded.detach() + (raw - raw.detach()) if straight_through else bounded
    return output, F.smooth_l1_loss(raw, bounded.detach(), beta=1.)


def bound_source_residual(residual, limit):
    """A per-image RMS budget which leaves small residuals unchanged."""
    if not limit:
        return residual
    rms = residual.float().square().mean((1, 2, 3), keepdim=True).clamp_min(1e-12).sqrt()
    return (residual.float() / (rms / limit).clamp_min(1)).to(residual.dtype)


def structure_quality(quality, edge, radius=2):
    """Nearby trusted semantics at measured E; color Q is kept unchanged.

    Untrusted probabilities still never enter the masked semantic encoder.
    This deterministic context gate is not learned contour confidence C.
    """
    nearby = F.max_pool2d(quality.detach().float(), radius * 2 + 1, 1, radius)
    support = F.max_pool2d(edge.detach().float(), 3, 1, 1)
    return (quality + (nearby - quality) * support).clamp(0, 1)


@torch.no_grad()
def augment_appearance_gray(condition, *, seed, step):
    """Vary the gray input of an RGB self-pair while retaining its geometry."""
    # The same committed-update index yields the same field on retry or resume,
    # without consuming the diffusion noise, patch or global RNG streams.
    key = hashlib.sha256(f"{APPEARANCE_LUMA_POLICY}:{seed}:{step}".encode()).digest()[:8]
    rng = torch.Generator(device="cpu").manual_seed(int.from_bytes(key, "little"))
    grid = 2 * torch.rand((condition.shape[0], 1, 3, 3), generator=rng, dtype=torch.float32, device="cpu") - 1
    field = F.interpolate(grid.to(condition.device), size=condition.shape[-2:], mode="bicubic", align_corners=False)
    offset = 0.15 * field / field.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1)
    gray = condition[:, :1].float()
    # At each position this curve is monotone and fixes black and white.
    warped = (gray + offset * (4 * gray * (1 - gray))).clamp(0, 1)
    return torch.cat((warped.to(condition.dtype), condition[:, 1:]), dim=1)


@torch.no_grad()
def appearance_class_weights(probabilities, quality, shape, minimum):
    """Weight reliable semantic interiors in the image's own coordinates."""
    batch, _, height, width = shape
    if (probabilities.ndim != 4 or probabilities.shape[1] < 2 or
        probabilities.shape[0] != batch or probabilities.shape[-2:] != (height, width) or
        quality.shape != (batch, 1, height, width)):
        raise ValueError("Appearance P/Q must share the RGB image geometry")
    probabilities, quality = probabilities.float(), quality.float()
    labels = probabilities.argmax(1, keepdim=True)
    # Channel zero combines unrelated background categories.
    classes = torch.arange(1, probabilities.shape[1], device=labels.device)[None, :, None, None]
    weights = probabilities[:, 1:] * quality
    return weights.mul_(labels == classes).mul_(quality >= minimum)


@torch.no_grad()
def _appearance_region_weights(probabilities, quality, shape):
    """Equal weight per reliable labeled class in each aligned RGB image."""
    weights = appearance_class_weights(probabilities, quality, shape, APPEARANCE_REGION_MIN_QUALITY)
    mass = weights.sum((-2, -1), keepdim=True)
    active = mass >= APPEARANCE_REGION_MIN_MASS
    weights.mul_(active).div_(mass.clamp_min(APPEARANCE_REGION_MIN_MASS))
    count = active.sum(1, keepdim=True)
    return weights.sum(1, keepdim=True) / count.clamp_min(1), count.flatten()


def rgb_appearance(generated, reference, settings, *, semantics=None, chroma_detail=0.):
    """Aligned visible targets for gray or detached synthetic thermal inputs."""
    if generated.shape != reference.shape or generated.ndim != 4 or generated.shape[1] != 3:
        raise ValueError("Aligned RGB self-pairs must share [N,3,H,W] geometry")
    target = reference.detach().float()
    reconstruction, detail = generated.sum() * 0, generated.sum() * 0
    for scale in (1, 2, 4):
        y, t = generated.float(), target
        if scale > 1:
            y, t = F.avg_pool2d(y, scale), F.avg_pool2d(t, scale)
        reconstruction = reconstruction + F.l1_loss(y, t) / 3
        gx, gy = gradients(gray01(y)); tx, ty = gradients(gray01(t))
        detail = detail + (F.l1_loss(gx, tx) + F.l1_loss(gy, ty)) / 6
    y_lab, t_lab = _lab((generated.float() + 1) / 2), _lab((target + 1) / 2)
    y_ab, t_ab = y_lab[:, 1:] / 128, t_lab[:, 1:] / 128
    chroma = F.l1_loss(y_ab, t_ab)
    components = {}
    if chroma_detail:
        # Match the target's chroma transitions, including legitimate boundaries.
        # A constant color offset remains the job of the Lab reconstruction.
        delta = y_ab - t_ab
        color_detail = generated.sum() * 0
        for scale in (1, 2, 4):
            pooled = F.avg_pool2d(delta, scale) if scale > 1 else delta
            dx, dy = pooled.diff(dim=-1), pooled.diff(dim=-2)
            color_detail = color_detail + (dx.abs().mean() + dy.abs().mean()) / 6
        detail = detail + chroma_detail * color_detail
        components["rgb_appearance_chroma_detail"] = color_detail.detach()
    if semantics is not None:
        region_weights, class_count = _appearance_region_weights(*semantics, target.shape)
        active = class_count > 0
        mix = APPEARANCE_REGION_BLEND * active.float()
        # L*/50 has the same nominal range as the full-resolution RGB error.
        # Blend only that reconstruction scale and chroma, retaining coarse RGB and detail.
        lightness_error = (y_lab[:, :1] - t_lab[:, :1]).abs() / 50
        chroma_error = (y_ab - t_ab).abs()
        region_lightness = (lightness_error * region_weights).sum((1, 2, 3))
        region_chroma = (chroma_error.mean(1, keepdim=True) * region_weights).sum((1, 2, 3))
        full_rgb = (generated.float() - target).abs().mean((1, 2, 3))
        full_chroma = chroma_error.mean((1, 2, 3))
        reconstruction = reconstruction + ((region_lightness - full_rgb) * mix).mean() / 3
        chroma = chroma + ((region_chroma - full_chroma) * mix).mean()
        active_images = active.sum().clamp_min(1)
        components.update(rgb_appearance_region_classes=class_count.float().mean(),
                          rgb_appearance_region_lightness=region_lightness.detach().sum() / active_images,
                          rgb_appearance_region_chroma=region_chroma.detach().sum() / active_images)
    total = (settings["appearance_rgb"] * reconstruction +
             settings["appearance_chroma"] * chroma + settings["appearance_detail"] * detail)
    components.update(rgb_appearance_reconstruction=reconstruction.detach(),
                      rgb_appearance_chroma=chroma.detach(), rgb_appearance_detail=detail.detach())
    return total, components


def color_attraction(prior, generated, probabilities, quality, minimum=.5):
    """Weak continuous attraction to the original mixture, not a fixed class color.

    The original support hinge stays a separate safeguard. An actually neutral
    learned mode remains valid; gray roads/night skies are not forced colorful.
    """
    ab = _lab((generated.float() + 1) / 2)[:, 1:]
    p, q = probabilities.detach().float(), quality.detach().float()[:, 0]
    labels = p.argmax(1)
    values, active = [], []
    for index, _ in prior.loss_classes:
        means = getattr(prior, f"loss_means_ab_{index}")
        variance = getattr(prior, f"loss_variances_ab_{index}")
        weights = getattr(prior, f"loss_weights_{index}")
        normalizer = 2 * math.log(2 * math.pi) + variance.log().sum(1)
        delta = ab[:, None] - means[None, :, :, None, None]
        nll = -torch.logsumexp(weights.log()[None, :, None, None] - .5 * (
            normalizer[None, :, None, None] +
            (delta.square() / variance[None, :, :, None, None]).sum(2)), 1)
        with torch.no_grad():
            centers = means[:, None] - means[None]
            center_nll = -torch.logsumexp(weights.log()[None] - .5 * (
                normalizer[None] + (centers.square() / variance[None]).sum(2)), 1)
            peak = center_nll.min()
        penalty = (nll - peak).clamp_min(0).log1p()
        eligible = (labels == index) & (q >= minimum)
        count = eligible.flatten(1).sum(1)
        values.append((penalty * eligible * q * p[:, index]).flatten(1).sum(1) / count.clamp_min(1))
        active.append(count > 0)
    if not values:
        return generated.sum() * 0
    return (torch.stack(values, 1).sum(1) / torch.stack(active, 1).sum(1).clamp_min(1)).mean()


def unsupported_structure(generated, tir01, edge, probabilities, quality, *, tolerance=3, minimum=.5):
    """Weak coarse-edge precision in trusted, flat interiors; permit fine RGB texture.

    E/TIR gradients and semantic transitions exempt a tolerance band. Pooling
    before gradients avoids classifying every photographic texture as an error.
    This is not object-ground-truth supervision.
    """
    labels = probabilities.detach().argmax(1, keepdim=True)
    boundary = torch.zeros_like(edge)
    for axis in (-1, -2):
        a, b = [slice(None)] * 4, [slice(None)] * 4
        a[axis], b[axis] = slice(1, None), slice(None, -1)
        a, b = tuple(a), tuple(b)
        changed = (labels[a] != labels[b]).float()
        boundary[a] = torch.maximum(boundary[a], changed)
        boundary[b] = torch.maximum(boundary[b], changed)
    radius = tolerance
    support = F.max_pool2d(torch.maximum(edge.detach().float(), boundary), 2 * radius + 1, 1, radius)
    q = quality.detach().float()
    mask = q * (q >= minimum) * (1 - support).clamp(0, 1)
    result = generated.sum() * 0
    for size in (5, 9):
        def response(image):
            image = F.avg_pool2d(F.pad(image, (size // 2,) * 4, mode="reflect"), size, 1)
            gx, gy = gradients(image)
            magnitude = (gx.square() + gy.square() + 1e-8).sqrt()
            local = F.avg_pool2d(image.square(), 9, 1, 4) - F.avg_pool2d(image, 9, 1, 4).square()
            return 1 - torch.exp(-magnitude / (local.clamp_min(1e-8).sqrt() + .04))
        reference = response(tir01.detach().float())
        predicted = response(gray01(generated))
        weight = mask * (1 - reference).square()
        result = result + ((predicted - reference - .15).relu() * weight).sum() / weight.sum().clamp_min(1) / 2
    return result
