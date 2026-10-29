"""Differentiable structural losses; no Canny/NumPy in the generated path."""

import torch
from torch.nn import functional as F


def gray01(rgb):
    return ((rgb.float() + 1) / 2 * rgb.new_tensor([.299, .587, .114])[None, :, None, None]).sum(1, keepdim=True)


def gradients(gray):
    kernel = gray.new_tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])[None, None] / 8
    padded = F.pad(gray, (1, 1, 1, 1), mode="reflect")
    return F.conv2d(padded, kernel), F.conv2d(padded, kernel.transpose(-1, -2))


def edge_preservation(generated, tir01, edge):
    """Preserve important support/direction; do not force identical brightness."""
    gray = gray01(generated)
    gx, gy = gradients(gray); tx, ty = gradients(tir01.float())
    magnitude = (gx.square() + gy.square() + 1e-8).sqrt()
    local = F.avg_pool2d(gray.square(), 9, 1, 4) - F.avg_pool2d(gray, 9, 1, 4).square()
    # sqrt(0) has an infinite derivative. Even an unselected/zero-weighted
    # flat pixel then yields 0*inf=NaN in backward. Floor BEFORE sqrt.
    deviation = local.clamp_min(1e-8).sqrt()
    response = 1 - torch.exp(-magnitude / (deviation + .02))
    support = F.max_pool2d(response, 3, 1, 1)
    # Polarity invariance: thermal and visible-light contrast may be inverted.
    dot = (gx * tx + gy * ty).abs()
    cosine = (dot / ((gx.square()+gy.square()+1e-6) * (tx.square()+ty.square()+1e-6)).sqrt()).clamp(0, 1)
    direction = (tx.square()+ty.square()) > 1e-5
    weights = edge.detach().float()
    recall = ((1 - support) * weights).sum() / weights.sum().clamp_min(1.)
    orientation = ((1 - cosine) * weights * direction).sum() / (weights * direction).sum().clamp_min(1.)
    return recall + .25 * orientation


def self_similarity(gray):
    """Fixed local descriptors, invariant to intensity offset/polarity.

    This is a small self-similarity contrastive objective, not a CUT/MIND reproduction.
    """
    padded = F.pad(gray, (2, 2, 2, 2), mode="reflect")
    h, w = gray.shape[-2:]
    distances = []
    for dy, dx in ((-2, 0), (2, 0), (0, -2), (0, 2), (-2, -2), (-2, 2), (2, -2), (2, 2)):
        neighbor = padded[:, :, 2+dy:2+dy+h, 2+dx:2+dx+w]
        distances.append(F.avg_pool2d((gray-neighbor).square(), 3, 1, 1))
    distance = torch.cat(distances, 1)
    return torch.exp(-distance / (distance.mean(1, keepdim=True) + 1e-4))


def spatial_content(generated, tir01, edge, *, generator=None, patches=128, temperature=.1):
    query = F.avg_pool2d(self_similarity(gray01(generated)), 4)
    key = F.avg_pool2d(self_similarity(tir01.float()), 4).detach()
    mask = F.adaptive_max_pool2d(edge.detach().float(), query.shape[-2:]).amax(0).flatten().cpu() > .05
    candidates = torch.nonzero(mask, as_tuple=False).flatten()
    if len(candidates) < 2:
        return generated.sum() * 0
    if generator is None:
        generator = torch.Generator("cpu").manual_seed(42)
    positions = candidates[torch.randperm(len(candidates), generator=generator)[:patches]].to(generated.device)
    q = F.normalize(query.flatten(2).index_select(2, positions), dim=1).transpose(1, 2)
    k = F.normalize(key.flatten(2).index_select(2, positions), dim=1)
    logits = torch.bmm(q, k) / temperature
    # Same diagonal cross-entropy without CUDA's nondeterministic nll_loss2d kernel.
    return (torch.logsumexp(logits, dim=-1) - logits.diagonal(dim1=-2, dim2=-1)).mean()


def semantic_consistency(predicted, source, quality, minimum=.1):
    ce = -(source.detach().float() * predicted.float().clamp_min(1e-7).log()).sum(1)
    labels = source.argmax(1)
    q = quality.detach()[:, 0].float()
    losses = []
    for index in range(source.shape[1]):
        mask = (labels == index) & (q >= minimum)
        weight = q * mask
        losses.append(((ce * weight).sum() / weight.sum().clamp_min(1.), (weight.sum() > 0).float()))
    values, active = zip(*losses)
    return torch.stack(values).sum() / torch.stack(active).sum().clamp_min(1.)


def chroma_smoothness(generated, edge, probabilities):
    """Weak within-class chroma regularization; allow changes at source boundaries."""
    rgb = (generated.float() + 1) / 2
    luminance = (rgb * rgb.new_tensor([.299,.587,.114])[None,:,None,None]).sum(1,keepdim=True)
    chroma = rgb - luminance
    labels = probabilities.detach().argmax(1, keepdim=True)
    total = generated.sum() * 0
    for axis in (-1, -2):
        a, b = [slice(None)] * 4, [slice(None)] * 4
        a[axis], b[axis] = slice(1, None), slice(None, -1)
        a, b = tuple(a), tuple(b)
        weight = torch.exp(-5 * torch.maximum(edge[a], edge[b]).detach()) * (labels[a] == labels[b])
        total = total + ((chroma[a] - chroma[b]).abs() * weight).mean()
    return total


def lsgan(predicted, real):
    return (predicted.float() - (1. if real else 0.)).square().mean()
