"""Dataset-independent, frozen multimodal chroma tokens learned from RGB."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def read_bank(path):
    raw = Path(path).read_bytes()
    bank = json.loads(raw)
    if bank.get("format") != "learned_chroma_gmm_v1":
        raise ValueError("Expected a learned_chroma_gmm_v1 artifact")
    names = bank.get("class_names", [])
    if not names or len(names) != len(set(names)) or any(not isinstance(n, str) or not n for n in names):
        raise ValueError("Class names must be nonempty and unique")
    if set(bank.get("classes", {})) != set(names):
        raise ValueError("Artifact class mapping differs")
    supported = 0
    for name in names:
        item = bank["classes"][name]
        if item.get("status") == "unsupported":
            continue
        if item.get("status") != "supported":
            raise ValueError("Invalid class support status")
        weight = np.asarray(item["weights"], dtype=float)
        mean, variance = (np.asarray(item[key], dtype=float) for key in ("means_ab", "variances_ab"))
        if (weight.ndim != 1 or not 1 <= len(weight) <= 8 or mean.shape != (len(weight), 2)
                or variance.shape != mean.shape or not np.isfinite(weight).all()
                or not np.isfinite(mean).all() or not np.isfinite(variance).all()
                or (weight <= 0).any() or (variance <= 0).any()
                or not np.isclose(weight.sum(), 1, atol=1e-6)):
            raise ValueError("Invalid learned mixture parameters")
        support = item.get("loss_support")
        if support is not None and (support.get("source") != "group_balanced_train_only"
                or not 0 < support.get("quantile", 0) < 1
                or not math.isfinite(support.get("max_nll", float("nan")))):
            raise ValueError("Invalid training-only color loss support")
        supported += 1
    if not supported:
        raise ValueError("No supported class in learned artifact")
    return bank, hashlib.sha256(raw).hexdigest()


def semantic_reliability(probabilities, second=None, boundary_radius=1):
    """Detached heuristic Q; neither calibrated correctness nor edge confidence C."""
    p = probabilities.detach().float()
    if (p.ndim != 4 or p.shape[1] < 2 or min(p.shape[-2:]) < 1 or not torch.isfinite(p).all()
            or (p < 0).any() or (p > 1).any()
            or not torch.allclose(p.sum(1), torch.ones_like(p[:, 0]), atol=4e-6, rtol=0)):
        raise ValueError("Expected valid categorical probabilities")
    if not isinstance(boundary_radius, int) or not 0 <= boundary_radius <= 8:
        raise ValueError("Invalid boundary radius")
    top = p.topk(2, dim=1).values
    entropy = -(p * p.clamp_min(1e-12).log()).sum(1, keepdim=True) / math.log(p.shape[1])
    quality = (top[:, :1] - top[:, 1:2]) * (1 - entropy).clamp(0, 1)
    if second is not None:
        if second.shape != p.shape:
            raise ValueError("Reliability views must share class mapping and coordinates")
        semantic_reliability(second, boundary_radius=0)
        quality *= (1 - .5 * (p - second.detach()).abs().sum(1, keepdim=True)).clamp(0, 1)
    if boundary_radius:
        labels = p.argmax(1, keepdim=True)
        boundary = torch.zeros_like(quality)
        vertical = labels[:, :, 1:] != labels[:, :, :-1]
        horizontal = labels[:, :, :, 1:] != labels[:, :, :, :-1]
        boundary[:, :, 1:] = torch.maximum(boundary[:, :, 1:], vertical.float())
        boundary[:, :, :-1] = torch.maximum(boundary[:, :, :-1], vertical.float())
        boundary[:, :, :, 1:] = torch.maximum(boundary[:, :, :, 1:], horizontal.float())
        boundary[:, :, :, :-1] = torch.maximum(boundary[:, :, :, :-1], horizontal.float())
        boundary = F.max_pool2d(boundary, 2*boundary_radius+1, stride=1, padding=boundary_radius)
        quality *= 1 - boundary
    return quality.detach()


class LearnedChromaPrior(nn.Module):
    def __init__(self, artifact, semantic_class_names, require_loss_support=False):
        super().__init__()
        bank, digest = read_bank(artifact)
        names = tuple(semantic_class_names)
        if len(names) != len(set(names)):
            raise ValueError("Semantic class names must be unique")
        selected = [name for name in bank["class_names"] if bank["classes"][name]["status"] == "supported"]
        if any(name not in names for name in selected):
            raise ValueError("Explicitly align learned and semantic class names")
        self.loss_classes = []
        for name in selected:
            item = bank["classes"][name]
            if "loss_support" not in item:
                if require_loss_support:
                    raise ValueError("Color loss requires loss_support fitted from training RGB; "
                                     "refit the prior with the updated learn_color_prior.py fit, "
                                     "or set learned_color_loss_weight=0 for the old conditioning-only control")
                continue
            index = names.index(name)
            # Derived buffers are reconstructed from the SHA-checked JSON, keeping
            # conditioning-only checkpoints compatible with the original branch.
            for key in ("weights", "means_ab", "variances_ab"):
                self.register_buffer(f"loss_{key}_{index}", torch.tensor(item[key], dtype=torch.float32), persistent=False)
            self.loss_classes.append((index, item["loss_support"]["max_nll"]))
        coefficients, indices, display_ab = [], [], np.zeros((len(names), 2), dtype=float)
        for name in selected:
            item = bank["classes"][name]
            weights, means, variances = (np.asarray(item[key]) for key in ("weights", "means_ab", "variances_ab"))
            index = names.index(name)
            display_ab[index] = (weights[:, None]*means).sum(0)
            for weight, mean, variance in zip(weights, means, variances):
                # Keep every component; never replace a multimodal bank with its mean.
                coefficients.extend((weight, *(weight*mean/128), *(weight*np.sqrt(variance)/128)))
                indices.extend([index]*5)
        from skimage.color import lab2rgb
        lab = np.column_stack((np.full(len(names), 65.), display_ab))
        display = lab2rgb(lab[None])[0].astype(np.float32)
        supported = torch.tensor([name in selected for name in names])
        self.register_buffer("indices", torch.tensor(indices, dtype=torch.long))
        self.register_buffer("coefficients", torch.tensor(coefficients, dtype=torch.float32))
        self.register_buffer("supported", supported)
        self.register_buffer("display_palette", torch.from_numpy(display))
        self.feature_channels = len(coefficients)
        self.metadata = dict(algorithm="rgb_learned_chroma_tokens_v1", artifact_sha256=digest,
                             semantic_class_names=list(names), supported_classes=selected,
                             feature_channels=self.feature_channels, mode_tokens="weight, weighted ab center and diagonal std for each component",
                             display_only="L*=65 and mixture-mean hue; not used as model input or ground truth")

    def forward(self, probabilities, quality):
        p = probabilities.detach().float()
        if p.shape[1] != len(self.supported) or quality.shape != p[:, :1].shape:
            raise ValueError("Prior scores/Q do not match current coordinates")
        if not torch.isfinite(quality).all() or (quality < 0).any() or (quality > 1).any():
            raise ValueError("Invalid reliability proxy")
        labels = p.argmax(1)
        mass = p[:, self.supported].sum(1, keepdim=True)
        gate = quality.detach() * mass * self.supported[labels][:, None]
        features = p.index_select(1, self.indices) * self.coefficients[None, :, None, None]
        hint = self.display_palette[labels].permute(0, 3, 1, 2)
        return dict(features=features, strength=gate, hint_rgb=hint*gate+.5*(1-gate))

    def chroma_loss(self, generated, probabilities, quality, min_quality=.5):
        """Bounded excess GMM NLL on reliable interiors; no pull within support.

        Average eligible pixels within each class, then active classes within each
        image. The denominator excludes Q so weak confidence stays a weak loss.
        Semantic probabilities, Q and the fitted distribution remain detached.
        """
        if len(self.loss_classes) != int(self.supported.sum()):
            raise ValueError("Refit training-only loss_support before enabling the color loss")
        if not math.isfinite(min_quality) or not 0 < min_quality <= 1:
            raise ValueError("Color loss min_quality must lie in (0,1]")
        if (generated.ndim != 4 or generated.shape[1] != 3 or not torch.isfinite(generated).all()
                or probabilities.shape != (generated.shape[0], len(self.supported), *generated.shape[-2:])
                or quality.shape != generated[:, :1].shape
                or not torch.isfinite(quality).all() or (quality < 0).any() or (quality > 1).any()):
            raise ValueError("Generated RGB, semantic probabilities and Q must share finite coordinates")
        from .semantic_color_prior import rgb_to_lab
        ab = rgb_to_lab((generated.float()+1)/2)[:, 1:]
        p, q = probabilities.detach().float(), quality.detach().float()[:, 0]
        labels = p.argmax(1)
        values, active, covered = [], [], torch.zeros_like(q)
        for index, max_nll in self.loss_classes:
            means = getattr(self, f"loss_means_ab_{index}")
            variance = getattr(self, f"loss_variances_ab_{index}")
            weights = getattr(self, f"loss_weights_{index}")
            delta = ab[:, None] - means[None, :, :, None, None]
            nll = -torch.logsumexp(weights.log()[None, :, None, None] - .5 * (
                2*math.log(2*math.pi) + variance.log().sum(1)[None, :, None, None]
                + (delta.square()/variance[None, :, :, None, None]).sum(2)), dim=1)
            excess = (nll-max_nll).relu()
            penalty = excess/(1+excess)
            eligible = (labels == index) & (q >= min_quality)
            count = eligible.flatten(1).sum(1)
            values.append((penalty*eligible*q*p[:, index]).flatten(1).sum(1)/count.clamp_min(1))
            active.append(count > 0)
            covered += eligible
        loss = torch.stack(values, 1).sum(1)/torch.stack(active, 1).sum(1).clamp_min(1)
        return loss.mean(), covered.mean().detach()
