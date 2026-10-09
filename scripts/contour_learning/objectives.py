"""Masked boundary supervision and coordinate-aligned local contrast."""

import torch
from torch.nn import functional as F


def decode_labels(labels):
    """PNG: 255=boundary, 0=confirmed non-boundary, 128=unknown/ignored."""
    if not torch.all((labels == 0) | (labels == 128) | (labels == 255)):
        raise ValueError("Labels must contain only 0, 128, 255")
    return (labels == 255).float(), labels != 128


def boundary_loss(logits, labels):
    target, valid = decode_labels(labels)
    if not valid.any():
        raise ValueError("No annotated pixels in this crop")
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    terms = [loss[valid & (target == value)].mean() for value in (0, 1)
             if (valid & (target == value)).any()]
    return torch.stack(terms).mean()


def local_contrastive_loss(embedding_a, embedding_b, labels, generator,
                           max_anchors=128, max_negatives=256,
                           temperature=0.1, exclusion_radius=2):
    """Edge anchors match the SAME pixel across aligned views.

    Only explicitly annotated non-boundaries are negatives. Other boundary
    pixels and unknown pixels are never negatives. The geometric transforms
    must already have been inverted. Both directions receive gradients.
    Returns loss and number of anchors with at least one usable negative.
    """
    if embedding_a.shape != embedding_b.shape or embedding_a.shape[-2:] != labels.shape[-2:]:
        raise ValueError("Embeddings and labels must be spatially aligned")
    if temperature <= 0 or max_anchors < 1 or max_negatives < 1 or exclusion_radius < 0:
        raise ValueError("Invalid contrastive sampling settings")
    target, valid = decode_labels(labels)
    h, w = labels.shape[-2:]
    losses, active = [], 0
    for i in range(labels.shape[0]):
        positives = torch.nonzero((valid[i, 0] & (target[i, 0] == 1)).flatten()).flatten().cpu()
        negatives = torch.nonzero((valid[i, 0] & (target[i, 0] == 0)).flatten()).flatten().cpu()
        if positives.numel() == 0 or negatives.numel() == 0:
            continue
        p = positives[torch.randperm(len(positives), generator=generator)[:max_anchors]].to(embedding_a.device)
        n = negatives[torch.randperm(len(negatives), generator=generator)[:max_negatives]].to(embedding_a.device)
        distance = torch.maximum((p[:, None] // w - n[None, :] // w).abs(),
                                 (p[:, None] % w - n[None, :] % w).abs())
        usable = distance > exclusion_radius
        keep = usable.any(1)
        if not keep.any():
            continue
        p, usable = p[keep], usable[keep]
        a = F.normalize(embedding_a[i].flatten(1).T, dim=1)
        b = F.normalize(embedding_b[i].flatten(1).T, dim=1)
        for source, destination in ((a, b), (b, a)):
            positive_score = (source[p] * destination[p]).sum(1, keepdim=True)
            negative_scores = (source[p] @ destination[n].T).masked_fill(~usable, -torch.inf)
            scores = torch.cat((positive_score, negative_scores), 1) / temperature
            losses.append(F.cross_entropy(scores, torch.zeros(len(p), device=scores.device, dtype=torch.long)))
        active += len(p)
    if not losses:
        return (embedding_a.sum() + embedding_b.sum()) * 0, 0
    return torch.stack(losses).mean(), active
