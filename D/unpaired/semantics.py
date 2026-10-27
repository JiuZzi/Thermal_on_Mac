"""Frozen source semantics; differentiable RGB teacher for generated images."""

from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from D.common import ROOT, DEFAULT_CACHE, sha256
from D.prepare_rgb import TEACHER, REVISION, CITY_TO_FMB, NAMES, map_probabilities


def legacy_models():
    path = str(ROOT / "pytorch-CycleGAN-and-pix2pix")
    if path not in sys.path:
        sys.path.insert(0, path)


def default_semantic_checkpoint():
    for name in ("runs/pretrained/segformer_fmb_b3/best_model.pth",
                 "runs/checkpoints/semantic_epoch80_for_mac/best_model.pth"):
        if (ROOT / name).is_file():
            return ROOT / name
    return ROOT / "runs/pretrained/segformer_fmb_b3/best_model.pth"


class RGBTeacher(nn.Module):
    def __init__(self, *, allow_download=False, snapshot=None):
        super().__init__()
        from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
        if snapshot is None:
            from huggingface_hub import snapshot_download
            snapshot = snapshot_download(TEACHER, revision=REVISION, cache_dir=str(DEFAULT_CACHE),
                local_files_only=not allow_download,
                allow_patterns=["config.json", "preprocessor_config.json", "pytorch_model.bin", "README.md"])
        snapshot = Path(snapshot)
        self.processor = AutoImageProcessor.from_pretrained(snapshot, local_files_only=True,
                                                             size={"height":512, "width":512})
        with torch.random.fork_rng(devices=[]):
            self.model = AutoModelForSemanticSegmentation.from_pretrained(snapshot, local_files_only=True,
                            trust_remote_code=False, weights_only=True).eval().requires_grad_(False)
        self.names = [str(self.model.config.id2label[i]).lower() for i in range(self.model.config.num_labels)]
        if set(self.names) != set(CITY_TO_FMB):
            raise ValueError("Unexpected RGB teacher taxonomy")
        matrix = torch.zeros(15, 19)
        for i, name in enumerate(self.names):
            matrix[NAMES.index(CITY_TO_FMB[name]), i] = 1
        self.register_buffer("mapping", matrix)
        self.register_buffer("mean", torch.tensor(self.processor.image_mean)[None, :, None, None])
        self.register_buffer("std", torch.tensor(self.processor.image_std)[None, :, None, None])
        self.info = dict(model=TEACHER, revision=REVISION,
                         files={name:sha256(snapshot / name) for name in
                                ("config.json", "preprocessor_config.json", "pytorch_model.bin")})

    @torch.no_grad()
    def predict_numpy(self, rgb):
        """Only source preparation uses PIL; generated images use forward()."""
        image = Image.fromarray(rgb).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.mean.device)
        logits = self.model(**inputs).logits.float()
        p = F.interpolate(logits, size=rgb.shape[:2], mode="bilinear", align_corners=False).softmax(1)[0]
        return map_probabilities(p.cpu().numpy(), self.names)

    def forward(self, rgb, side=256):
        """RGB in [-1,1]. Frozen parameters, but preserve gradient to rgb."""
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError("Expected generated RGB [N,3,H,W]")
        output_device = rgb.device
        rgb = rgb.to(self.mean.device)
        self.model.eval()
        pixels = F.interpolate((rgb.float() + 1) / 2, size=(side, side), mode="bilinear",
                               align_corners=False, antialias=True)
        pixels = (pixels - self.mean) / self.std
        def infer(value):
            return self.model(pixel_values=value).logits.float()
        if torch.is_grad_enabled() and pixels.requires_grad:
            logits = checkpoint(infer, pixels, use_reentrant=False)
        else:
            logits = infer(pixels)
        p = F.interpolate(logits, size=rgb.shape[-2:], mode="bilinear", align_corners=False).softmax(1)
        mapped = torch.einsum("kc,nchw->nkhw", self.mapping, p)
        return (mapped / mapped.sum(1, keepdim=True).clamp_min(1e-12)).to(output_device)


class ConditionBuilder:
    """Same clarity/R rules for real and reverse-cycle synthetic TIR."""
    def __init__(self, semantic_checkpoint=None, *, rgb=False, allow_download=False, flip=True, device="cpu"):
        legacy_models()
        from models.semantic_contour import FrozenSemanticPrior
        from models.clarity_semantic_edges import clarity_components
        from models.learned_color_prior import semantic_reliability
        self.tir = FrozenSemanticPrior(semantic_checkpoint or default_semantic_checkpoint())
        self.rgb = RGBTeacher(allow_download=allow_download).to(device) if rgb else None
        self.clarity, self.reliability, self.flip = clarity_components, semantic_reliability, flip
        self.device = device

    def compute(self, image, domain):
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("Expected processed RGB/replicated TIR uint8")
        gray = np.asarray(Image.fromarray(image).convert("L")).copy()
        values = ((gray.astype(np.float32) / 255 - .5) / .5 + 1) / 2
        if domain == "TIR":
            p = self.tir.predict(values, self.device)
            second = self.tir.predict(np.fliplr(values).copy(), self.device)[:, :, ::-1].copy() if self.flip else None
        elif domain == "RGB" and self.rgb is not None:
            p = self.rgb.predict_numpy(image)
            second = self.rgb.predict_numpy(image[:, ::-1].copy())[:, :, ::-1].copy() if self.flip else None
        else:
            raise ValueError("Unknown/unavailable semantic domain")
        q = self.reliability(torch.from_numpy(p)[None],
                torch.from_numpy(second)[None] if second is not None else None, boundary_radius=1)[0, 0].numpy()
        edge = self.clarity(values, p, include_probes=False)[0]["clarity_v1"]
        return gray, edge, p, q

    def synthetic(self, tir):
        """Conditions are detached as before; raw synthetic TIR retains its graph."""
        values = np.rint(np.clip((tir.detach().float().cpu().numpy()[0, 0] + 1) / 2, 0, 1) * 255).astype(np.uint8)
        image = np.repeat(values[:, :, None], 3, axis=2)
        _, edge, p, q = self.compute(image, "TIR")
        device = tir.device
        e = torch.from_numpy(edge)[None, None].to(device)
        return torch.cat(((tir.float() + 1) / 2, e, torch.zeros_like(e)), 1), \
               torch.from_numpy(p)[None].to(device), torch.from_numpy(q)[None, None].to(device)
