"""Online, frozen 512 semantics for the existing structure-protected contour.

The external model is held by a plain wrapper, so GAN initialization, train(),
optimizers and checkpoints cannot update or reinitialize its weights.
"""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

from .conditioned_generator import ConditionedGenerator
from .semantic_edges import SETTINGS, IMPORTANT_CLASSES, make_structure_semantic_condition
from .selective_semantic_edges import VARIANTS, SELECTIVE_SETTINGS, make_selective_semantic_condition
from .clarity_semantic_edges import CLARITY_SETTINGS, make_clarity_semantic_condition
from .recall_semantic_edges import RECALL_SETTINGS, make_recall_semantic_condition


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = ROOT / "runs/pretrained/segformer_fmb_b3/best_model.pth"


class FrozenSemanticPrior:
    def __init__(self, checkpoint):
        scripts = str(ROOT / "scripts")
        if scripts not in sys.path:
            sys.path.insert(0, scripts)
        from semantic_preview.checkpoint import load_checkpoint
        from semantic_preview.inference import predict_probabilities

        # Loading an externally initialized network must not alter the seed
        # used to initialize the GAN or sample its training data.
        with torch.random.fork_rng(devices=[]):
            self.model, self.info = load_checkpoint(checkpoint, torch.device("cpu"))
        self._predict = predict_probabilities
        self.device = torch.device("cpu")

    def predict(self, gray, device):
        device = torch.device(device)
        if self.device != device:
            self.model.to(device)
            self.device = device
        # Preserve the author's uint8 grayscale input, including for synthetic
        # TIR. Canny still receives the original continuous normalized image.
        image = np.rint(np.clip(gray, 0, 1) * 255).astype(np.uint8)
        return self._predict(self.model, image, device, short_side=512,
                             flip_tta=False)


class SemanticContourGenerator(ConditionedGenerator):
    """[TIR, selected contour, zero] -> RGB on transformed real and cycle TIR."""

    def __init__(self, backbone, checkpoint, contour_variant="joint512", **condition_kwargs):
        if contour_variant not in VARIANTS:
            raise ValueError("Unknown semantic contour variant")
        super().__init__(backbone, "soft_saliency", fusion_mode="direct", **condition_kwargs)
        fixed = (
            np.isclose(self.edge_sigma, 1.0)
            and np.isclose(self.edge_low_threshold, 0.08)
            and np.isclose(self.edge_high_threshold, 0.16)
            and np.isclose(self.edge_soft_width, SETTINGS["soft_width"])
            and len(self.soft_high_thresholds) == len(SETTINGS["high_thresholds"])
            and np.allclose(self.soft_high_thresholds, SETTINGS["high_thresholds"])
            and np.allclose(self.saliency_sigmas, SETTINGS["sigmas"])
            and np.isclose(self.saliency_background_gain, SETTINGS["background_gain"])
            and self.saliency_variant == "structure_texture"
        )
        if not fixed:
            raise ValueError("Protected semantic contours use the fixed v2 candidate/fusion settings; do not change Canny or saliency settings")
        self.semantic_prior = FrozenSemanticPrior(checkpoint)
        self.contour_variant = contour_variant
        self.semantic_metadata = {
            "algorithm": "structure_semantic_protection_v1",
            "checkpoint_sha256": self.semantic_prior.info["sha256"],
            "short_side": 512, "flip_tta": False,
            "classes": IMPORTANT_CLASSES, "settings": SETTINGS,
            "input": "current transformed TIR; rounded uint8 for semantic inference",
            "cycle": "recompute on current synthetic TIR; detached fixed condition",
        }
        # Leave the legacy dictionary/signature byte-for-byte unchanged. New
        # contours must never silently load a checkpoint trained on joint512.
        if contour_variant != "joint512":
            selection = {"selective_v1": SELECTIVE_SETTINGS, "clarity_v1": CLARITY_SETTINGS,
                         "recall_v1": RECALL_SETTINGS}[contour_variant]
            self.semantic_metadata.update(algorithm=selection["algorithm"],
                                          contour_variant=contour_variant, selection=selection)
        digest = hashlib.sha256(json.dumps(self.semantic_metadata, sort_keys=True).encode()).digest()
        self.register_buffer("semantic_signature", torch.tensor(list(digest), dtype=torch.uint8))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        saved = state_dict.get(prefix + "semantic_signature")
        if saved is not None and not torch.equal(saved.cpu(), self.semantic_signature.cpu()):
            error_msgs.append("Semantic condition signature mismatch: use the same frozen checkpoint and protected-contour settings as training")
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def contour_from_probabilities(self, image, probabilities):
        if self.contour_variant == "joint512":
            return make_structure_semantic_condition(image, probabilities)
        if self.contour_variant == "selective_v1":
            return make_selective_semantic_condition(image, probabilities)
        if self.contour_variant == "clarity_v1":
            return make_clarity_semantic_condition(image, probabilities)
        return make_recall_semantic_condition(image, probabilities)

    def make_condition(self, tir, saliency_override=None):
        if saliency_override is not None:
            raise ValueError("Protected semantic contours compute importance online; external saliency overrides are not supported")
        if tir.ndim != 4 or tir.shape[1] != 1 or not torch.isfinite(tir).all():
            raise ValueError("Expected finite TIR [N,1,H,W]")
        gray = np.clip((tir.detach().float().cpu().numpy()[:, 0] + 1) / 2, 0, 1)
        maps = []
        for image in gray:
            probabilities = self.semantic_prior.predict(image, tir.device)
            maps.append(self.contour_from_probabilities(image, probabilities))
        edge = torch.from_numpy(np.stack(maps)[:, None]).to(device=tir.device, dtype=tir.dtype)
        return edge, torch.zeros_like(edge)

    def forward(self, tir, edge_override=None, saliency_override=None, return_condition=False):
        if tir.ndim != 4 or tir.shape[1] != 1:
            raise ValueError("Expected one-channel TIR [N,1,H,W]")
        if saliency_override is not None:
            raise ValueError("Protected semantic contours do not accept external saliency overrides")
        if edge_override is None:
            edge, reserved = self.make_condition(tir)
        else:
            if edge_override.shape != tir.shape:
                raise ValueError("Edge override must match the current TIR geometry")
            edge = edge_override.to(tir)
            reserved = torch.zeros_like(tir)
        rgb = self.backbone(torch.cat((tir, edge, reserved), 1))
        return (rgb, edge) if return_condition else rgb
