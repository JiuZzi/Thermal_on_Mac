"""Detached descriptive coverage statistics, never semantic/edge accuracy."""

import json
from pathlib import Path

import torch
from torch.nn import functional as F


def guidance_report(probabilities, quality, names, supported=None, min_quality=.5,
                    boundary_radius=1, feature_gate=None, scope="current minibatch"):
    p, q = probabilities.detach().float(), quality.detach().float()
    if (p.ndim != 4 or p.shape[1] != len(names) or q.shape != p[:, :1].shape
            or not 0 < min_quality <= 1 or not torch.isfinite(p).all() or not torch.isfinite(q).all()
            or (p < 0).any() or (q < 0).any() or (q > 1).any()
            or not torch.allclose(p.sum(1), torch.ones_like(p[:, 0]), atol=4e-6, rtol=0)):
        raise ValueError("Coverage requires aligned categorical probabilities and Q")
    if not isinstance(boundary_radius, int) or boundary_radius < 0:
        raise ValueError("Boundary radius must be a nonnegative integer")
    labels = p.argmax(1, keepdim=True)
    boundary = torch.zeros_like(q)
    vertical, horizontal = labels[:, :, 1:] != labels[:, :, :-1], labels[:, :, :, 1:] != labels[:, :, :, :-1]
    boundary[:, :, 1:] = torch.maximum(boundary[:, :, 1:], vertical.float())
    boundary[:, :, :-1] = torch.maximum(boundary[:, :, :-1], vertical.float())
    boundary[:, :, :, 1:] = torch.maximum(boundary[:, :, :, 1:], horizontal.float())
    boundary[:, :, :, :-1] = torch.maximum(boundary[:, :, :, :-1], horizontal.float())
    if boundary_radius:
        boundary = F.max_pool2d(boundary, 2 * boundary_radius + 1, stride=1, padding=boundary_radius)
    boundary = boundary.bool()
    supported_names = None if supported is None else set(supported)
    if supported_names is not None and not supported_names.issubset(names):
        raise ValueError("Unknown supported class")
    if feature_gate is not None:
        feature_gate = feature_gate.detach()
        if (feature_gate.shape != q.shape or not torch.isfinite(feature_gate).all()
                or (feature_gate < 0).any() or (feature_gate > 1).any()):
            raise ValueError("Feature gate must be finite, aligned and in [0,1]")
    report = dict(scope=scope, images=p.shape[0], pixels=q.numel(), min_quality=min_quality,
                  mean_quality=float(q.mean()), classes={}, supported_classes=None if supported is None else sorted(supported_names),
                  limits="Argmax regions/coverage are diagnostics, not labels, accuracy or calibrated confidence.")
    eligible_all = torch.zeros_like(boundary)
    for index, name in enumerate(names):
        mask = labels == index
        count = int(mask.sum())
        eligible = mask & (q >= min_quality)
        boundary_count = int((mask & boundary).sum())
        row = dict(pixels=count, quality_sum=float(q[mask].sum()),
                   nonzero_quality_pixels=int((mask & (q > 0)).sum()),
                   threshold_pixels=int(eligible.sum()), boundary_pixels=boundary_count,
                   boundary_nonzero_pixels=int((mask & boundary & (q > 0)).sum()))
        row.update(mean_quality=row["quality_sum"] / max(count, 1),
                   threshold_fraction=row["threshold_pixels"] / max(count, 1),
                   boundary_fraction=boundary_count / max(count, 1))
        if supported_names is not None:
            row["color_loss_eligible_pixels"] = row["threshold_pixels"] if name in supported_names else 0
            eligible_all |= eligible & (name in supported_names)
        if feature_gate is not None:
            row["condition_nonzero_pixels"] = int((mask & (feature_gate > 0)).sum())
        report["classes"][name] = row
    if supported_names is not None:
        report["color_loss_eligible_fraction"] = float(eligible_all.float().mean())
    if feature_gate is not None:
        report["condition_nonzero_fraction"] = float((feature_gate > 0).float().mean())
    return report


def save_report(report, directory):
    """Keep only the latest sampled batch, so diagnostic storage stays bounded."""
    directory = Path(directory)
    destination = directory / "guidance_diagnostics_latest.json"
    temporary = directory / ".guidance_diagnostics_latest.tmp"
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return destination
