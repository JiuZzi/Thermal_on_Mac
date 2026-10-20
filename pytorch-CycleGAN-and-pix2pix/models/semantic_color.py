"""Independent semantic/color feature injection around the unchanged joint512."""

import hashlib
import json

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .semantic_color_prior import SemanticColorPrior, color_config
from .semantic_contour import SemanticContourGenerator
from .semantic_edges import make_structure_semantic_condition


CONDITION_MODES = ("none", "semantic", "semantic_color")


class SemanticColorGenerator(SemanticContourGenerator):
    def __init__(self, backbone, checkpoint, color_condition="semantic_color", color_width=16, **kwargs):
        if color_condition not in CONDITION_MODES or color_width < 1:
            raise ValueError("Unknown color condition or nonpositive branch width")
        if not isinstance(getattr(backbone, "model", None), nn.Sequential):
            raise ValueError("The semantic/color prototype requires a ResNet generator")
        super().__init__(backbone, checkpoint, **kwargs)
        self.color_condition = color_condition
        self.color_prior = SemanticColorPrior()
        # Preserve sampler RNG when adding the independently initialized branch.
        with torch.random.fork_rng(devices=[]):
            self.condition_encoder = nn.Sequential(
                nn.Conv2d(21, color_width, 3, padding=1), nn.ReLU(),
                nn.Conv2d(color_width, color_width, 3, padding=1), nn.ReLU(),
            )
            self.condition_projections = nn.ModuleDict()
            channels = None
            for index, layer in enumerate(self.backbone.model):
                if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
                    channels = layer.out_channels
                if isinstance(layer, nn.ReLU):
                    self.condition_projections[str(index)] = nn.Conv2d(color_width, channels, 1)
        self.reset_color_fusion()
        self.color_metadata = dict(
            algorithm="semantic_color_feature_v1", condition=color_condition, width=color_width,
            semantic_channels=15, color_channels=6, color_prior=color_config(),
            injections=list(self.condition_projections), third_channel="reserved zero",
        )
        digest = hashlib.sha256(json.dumps(self.color_metadata, sort_keys=True).encode()).digest()
        self.register_buffer("color_signature", torch.tensor(list(digest), dtype=torch.uint8))

    def reset_color_fusion(self):
        for layer in self.condition_projections.values():
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        saved = state_dict.get(prefix + "color_signature")
        if saved is not None and not torch.equal(saved.cpu(), self.color_signature.cpu()):
            error_msgs.append("Color condition signature mismatch: use the trained mode and prior")
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def initialize_from_semantic(self, state):
        """Explicit old-generator warm start; never silently relax checkpoint loads."""
        expected = {"semantic_signature"} | {"backbone." + key for key in self.backbone.state_dict()}
        if set(state) != expected:
            raise ValueError("Expected an original semantic_contour generator checkpoint")
        if not torch.equal(state["semantic_signature"].cpu(), self.semantic_signature.cpu()):
            raise ValueError("Legacy semantic prior signature differs")
        self.backbone.load_state_dict({key[len("backbone."):]: value for key, value in state.items()
                                      if key.startswith("backbone.")}, strict=True)
        self.reset_color_fusion()

    def forward(self, tir, return_aux=False, probabilities_override=None):
        if tir.ndim != 4 or tir.shape[1] != 1 or not torch.isfinite(tir).all():
            raise ValueError("Expected finite TIR [N,1,H,W]")
        gray = np.clip((tir.detach().float().cpu().numpy()[:, 0] + 1) / 2, 0, 1)
        if probabilities_override is None:
            scores = np.stack([self.semantic_prior.predict(image, tir.device) for image in gray])
            p = torch.from_numpy(scores).to(device=tir.device, dtype=torch.float32)
        else:
            p = probabilities_override.detach().to(device=tir.device, dtype=torch.float32)
            if p.shape != (tir.shape[0], 15, *tir.shape[-2:]):
                raise ValueError("Semantic override must match current TIR coordinates")
            scores = p.cpu().numpy()
        color = self.color_prior(p)
        edges = np.stack([make_structure_semantic_condition(image, score)
                          for image, score in zip(gray, scores)])
        edge = torch.from_numpy(edges[:, None]).to(tir)
        base = torch.cat((tir, edge, torch.zeros_like(tir)), 1)
        encoded = None
        if self.color_condition != "none":
            features = color["features"] if self.color_condition == "semantic_color" else torch.zeros_like(color["features"])
            encoded = self.condition_encoder(torch.cat((p, features), 1).to(tir))
        rgb = base
        for index, layer in enumerate(self.backbone.model):
            rgb = layer(rgb)
            key = str(index)
            if encoded is not None and key in self.condition_projections:
                local = F.interpolate(encoded, size=rgb.shape[-2:], mode="bilinear", align_corners=False)
                # Addition produces a new tensor; later inplace ReLUs stay safe.
                rgb = rgb + self.condition_projections[key](local)
        aux = dict(edge=edge, probabilities=p, color=color)
        return (rgb, aux) if return_aux else rgb
