"""Add learned RGB chroma tokens around the existing semantic generator."""

import hashlib
import json

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .learned_color_prior import LearnedChromaPrior, semantic_reliability
from .semantic_edges import FMB_CLASS_NAMES, make_structure_semantic_condition


class LearnedColorGenerator(nn.Module):
    def __init__(self, semantic, artifact, width=16, mode="learned", flip=False,
                 boundary_radius=1, gate_semantic=False, dropout=.1,
                 loss_weight=0., loss_min_quality=.5, loss_warmup_steps=1000):
        super().__init__()
        if semantic.color_condition != "semantic" or mode not in ("learned", "zero_tokens"):
            raise ValueError("Warm-start the semantic-only generator; choose learned/zero_tokens")
        if width < 1 or not 0 <= dropout < 1:
            raise ValueError("Invalid learned branch width/dropout")
        self.semantic = semantic
        self.learned_prior = LearnedChromaPrior(artifact, FMB_CLASS_NAMES, require_loss_support=loss_weight > 0)
        self.mode, self.flip, self.boundary_radius = mode, flip, boundary_radius
        self.gate_semantic, self.dropout = gate_semantic, dropout
        self.feature_override = "normal"
        with torch.random.fork_rng(devices=[]):
            self.learned_encoder = nn.Sequential(nn.Conv2d(self.learned_prior.feature_channels, width, 3, padding=1),
                                                 nn.ReLU(), nn.Conv2d(width, width, 3, padding=1), nn.ReLU())
            self.learned_projections = nn.ModuleDict({key: nn.Conv2d(width, layer.out_channels, 1)
                                                     for key, layer in semantic.condition_projections.items()})
        self.color_metadata = dict(self.learned_prior.metadata, condition=mode, width=width,
                                   reliability=dict(flip=flip, boundary_radius=boundary_radius,
                                                    gate_semantic=gate_semantic, calibrated=False),
                                   condition_dropout=dropout, chroma_loss="disabled; conditioning-only first experiment",
                                   third_channel="reserved zero")
        if loss_weight > 0:
            self.color_metadata["chroma_loss"] = dict(algorithm="bounded_excess_gmm_nll_v1", weight=loss_weight,
                min_quality=loss_min_quality, warmup_steps=loss_warmup_steps, support="group-balanced train NLL quantile",
                normalization="eligible pixels per class, then active classes per image; Q stays in numerator",
                bound="raw loss in [0,1]; weighted loss <= configured weight")
            self.register_buffer("color_loss_updates", torch.tensor(0, dtype=torch.long))
        digest = hashlib.sha256(json.dumps(self.color_metadata, sort_keys=True).encode()).digest()
        self.register_buffer("learned_signature", torch.tensor(list(digest), dtype=torch.uint8))
        self.reset_color_fusion()

    @property
    def backbone(self):
        return self.semantic.backbone

    @property
    def semantic_prior(self):
        return self.semantic.semantic_prior

    @property
    def fusion_mode(self):
        return self.semantic.fusion_mode

    @property
    def semantic_metadata(self):
        return self.semantic.semantic_metadata

    def reset_color_fusion(self):
        self.semantic.reset_color_fusion()
        self.reset_learned_fusion()

    def reset_learned_fusion(self):
        for layer in self.learned_projections.values():
            nn.init.zeros_(layer.weight); nn.init.zeros_(layer.bias)

    def initialize_from_semantic(self, state):
        self.semantic.initialize_from_semantic(state)
        self.reset_learned_fusion()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        saved = state_dict.get(prefix+"learned_signature")
        if saved is not None and not torch.equal(saved.cpu(), self.learned_signature.cpu()):
            error_msgs.append("Learned color signature mismatch: artifact, mode or reliability configuration changed")
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def forward(self, tir, return_aux=False, probabilities_override=None, second_override=None):
        if tir.ndim != 4 or tir.shape[1] != 1 or not torch.isfinite(tir).all():
            raise ValueError("Expected finite TIR [N,1,H,W]")
        gray = np.clip((tir.detach().float().cpu().numpy()[:, 0]+1)/2, 0, 1)
        if probabilities_override is None:
            scores = np.stack([self.semantic_prior.predict(image, tir.device) for image in gray])
            p = torch.from_numpy(scores).to(tir.device)
            second = None
            if self.flip:
                flipped = np.stack([self.semantic_prior.predict(np.fliplr(image).copy(), tir.device)[:, :, ::-1].copy() for image in gray])
                second = torch.from_numpy(flipped).to(tir.device)
        else:
            p = probabilities_override.detach().to(tir.device)
            if p.shape != (tir.shape[0], len(FMB_CLASS_NAMES), *tir.shape[-2:]):
                raise ValueError("Semantic override coordinates differ")
            scores, second = p.cpu().numpy(), second_override
            if self.flip and second is None:
                raise ValueError("Supply the second view for a flip-reliability override")
        quality = semantic_reliability(p, second, self.boundary_radius)
        color = self.learned_prior(p, quality)
        tokens, gate = color["features"].to(tir), color["strength"].to(tir)
        if self.mode == "zero_tokens":
            tokens = torch.zeros_like(tokens)
        if self.feature_override == "shuffle":
            tokens = tokens.roll(max(1, tir.shape[-1]//2), -1)
        elif self.feature_override not in ("normal", "off"):
            raise ValueError("Unknown learned feature override")
        # Both training controls consume the same RNG; Q cannot learn to hide errors.
        if self.training and self.dropout:
            gate = gate * (torch.rand((tir.shape[0], 1, 1, 1), device=tir.device) >= self.dropout)
        if self.feature_override == "off":
            gate = torch.zeros_like(gate)
        edge = torch.from_numpy(np.stack([make_structure_semantic_condition(image, score)
                                         for image, score in zip(gray, scores)])[:, None]).to(tir)
        raw_semantic = self.semantic.condition_encoder(torch.cat((p, torch.zeros_like(p[:, :6])), 1).to(tir))
        learned = self.learned_encoder(tokens)
        rgb = torch.cat((tir, edge, torch.zeros_like(tir)), 1)
        for index, layer in enumerate(self.backbone.model):
            rgb = layer(rgb)
            key = str(index)
            if key in self.learned_projections:
                sem = F.interpolate(raw_semantic, size=rgb.shape[-2:], mode="bilinear", align_corners=False)
                sem = self.semantic.condition_projections[key](sem)
                if self.gate_semantic:
                    sem = sem * F.interpolate(quality, size=rgb.shape[-2:], mode="bilinear", align_corners=False)
                local = F.interpolate(learned, size=rgb.shape[-2:], mode="bilinear", align_corners=False)
                local = self.learned_projections[key](local)
                local = local * F.interpolate(gate, size=rgb.shape[-2:], mode="bilinear", align_corners=False)
                rgb = rgb + sem + local
        aux = dict(edge=edge, probabilities=p, quality=quality, color=color)
        return (rgb, aux) if return_aux else rgb
