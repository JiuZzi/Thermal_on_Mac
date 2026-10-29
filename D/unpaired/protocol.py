"""Versioned formal-D configuration, state and data contracts."""

import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import torch

from D.common import ROOT, read_json, sha256

DEFAULT_CONFIG = ROOT / "D/unpaired/config.json"
DEFAULT_DATA = ROOT / "runs/D/unpaired/conditions/clarity_seed42"
DEFAULT_RUNS = ROOT / "runs/D/unpaired/training"
DEFAULT_PRIOR = ROOT / "runs/learned_color/rgb_prior_seed42_loss_v1.json"
DEFAULT_MANIFEST = ROOT / "runs/learned_color/data_seed42_v1/cycle_manifest.json"


def config_from(path=DEFAULT_CONFIG):
    config = read_json(path)
    if config.get("format") not in ("D_unpaired_v1", "D_unpaired_v2") or config.get("seed") != 42:
        raise ValueError("Formal D uses the versioned seed42 configuration")
    policies = {"fusion_policy", "fusion_min_quality", "chroma_smooth_policy"}
    if config["format"] == "D_unpaired_v2":
        if (not policies.issubset(config) or config["fusion_policy"] not in ("legacy", "quality_masked") or
            config["chroma_smooth_policy"] not in ("legacy", "quality_soft") or
            not isinstance(config["fusion_min_quality"], (int, float)) or not 0 <= config["fusion_min_quality"] <= 1):
            raise ValueError("Specify valid D-v2 fusion and chroma smoothing policies")
    elif policies.intersection(config):
        raise ValueError("Use D_unpaired_v2 for new fusion/smoothing policies; preserve D-v1 runs")
    for flag in ("semantic_gate", "use_semantic", "use_color", "flip_reliability"):
        if type(config.get(flag)) is not bool:
            raise ValueError(f"{flag} must be a boolean")
    if config.get("confidence_enabled") is not False:
        raise ValueError("Confidence C is not implemented; third channel must remain zero")
    for key in ("width", "height", "sampling_steps", "semantic_width", "semantic_loss_size", "replay_every"):
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError(f"Invalid {key}")
    if min(config["width"], config["height"]) < 64 or config["width"] % 8 or config["height"] % 8:
        raise ValueError("Use image dimensions >=64 and divisible by 8")
    if (not 0 < config["strength"] <= 1 or config["sampling_steps"] > 1000 or
        int(config["sampling_steps"] * config["strength"]) < 1 or config["guidance_scale"] < 1):
        raise ValueError("Invalid diffusion schedule/guidance")
    if type(config.get("lora_rank")) is not int or config["lora_rank"] < 0:
        raise ValueError("lora_rank must be a nonnegative integer")
    expected_losses = {"adversarial", "cycle_tir", "cycle_rgb", "edge", "content", "semantic", "color_prior", "chroma_smooth", "rgb_replay"}
    if set(config.get("loss_weights", {})) != expected_losses:
        raise ValueError("Specify all documented formal-D losses; set a weight to zero for ablations")
    for key, value in config["loss_weights"].items():
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid loss weight {key}")
    for key in ("lr_diffusion", "lr_reverse", "lr_discriminator", "max_grad_norm"):
        if not 0 < config[key] < float("inf"):
            raise ValueError(f"Invalid {key}")
    if not math.isfinite(config["guidance_scale"]) or config["semantic_loss_size"] < 32:
        raise ValueError("Invalid guidance or semantic-teacher size")
    if config.get("semantic_loss_device", "cuda:0") not in ("cpu", "cuda:0"):
        raise ValueError("semantic_loss_device must be cpu or cuda:0")
    for key in ("warmup_steps", "color_warmup_steps", "pool_size"):
        if type(config[key]) is not int or config[key] < 0:
            raise ValueError(f"Invalid {key}")
    for key in ("semantic_min_quality", "color_min_quality"):
        if not 0 <= config[key] <= 1:
            raise ValueError(f"Invalid {key}")
    if config["loss_weights"]["color_prior"] and config["color_min_quality"] <= 0:
        raise ValueError("Active color prior requires color_min_quality in (0,1]")
    for key in ("base_revision", "control_revision"):
        if len(config[key]) != 40 or any(c not in "0123456789abcdef" for c in config[key]):
            raise ValueError("Pin pretrained models to full commits")
    return config


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def model_config(config):
    """Existing D loader needs model IDs; do not change its preview config."""
    return {key:config[key] for key in ("base_model", "base_revision", "control_model", "control_revision")}


def seed_runtime():
    torch.set_num_threads(4)
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Legacy reflection padding and some CUDA resize backwards lack a strict
    # deterministic kernel. Keep seed42 and request deterministic algorithms,
    # but report unsupported kernels rather than making training impossible.
    torch.use_deterministic_algorithms(True, warn_only=True)


class IndexStream:
    def __init__(self, count, namespace="data"):
        if count < 1:
            raise ValueError("Empty domain")
        self.count, self.order, self.position, self.namespace = count, [], 0, namespace
        child_seed = int.from_bytes(hashlib.sha256(f"42:{namespace}".encode()).digest()[:8], "little") % (2**63)
        self.generator = torch.Generator("cpu").manual_seed(child_seed)

    def next(self):
        if self.position >= len(self.order):
            self.order = torch.randperm(self.count, generator=self.generator).tolist()
            self.position = 0
        value = self.order[self.position]; self.position += 1
        return value

    def state(self):
        return dict(count=self.count, namespace=self.namespace, order=self.order, position=self.position, rng=self.generator.get_state())

    def restore(self, state):
        if state["count"] != self.count or state["namespace"] != self.namespace or not 0 <= state["position"] <= len(state["order"]):
            raise ValueError("Sampler/data count mismatch")
        self.order, self.position = state["order"], state["position"]
        self.generator.set_state(state["rng"])


class ReplayPool:
    def __init__(self, size=50):
        self.size, self.images, self.rng = size, [], random.Random(42)

    def query(self, image):
        detached = image.detach()
        if self.size == 0:
            return detached
        stored = detached.cpu().clone()
        if len(self.images) < self.size:
            self.images.append(stored)
            return detached
        if self.rng.random() > .5:
            index = self.rng.randrange(self.size)
            old, self.images[index] = self.images[index], stored
            return old.to(image)
        return detached

    def state(self):
        return dict(size=self.size, images=self.images, rng=self.rng.getstate())

    def restore(self, state):
        if state["size"] != self.size:
            raise ValueError("Replay-pool size mismatch")
        self.images = state["images"]; self.rng.setstate(state["rng"])


def validate_arrays(gray, edge, probabilities, quality):
    if (gray.ndim != 2 or gray.dtype != np.uint8 or edge.shape != gray.shape or
        probabilities.shape != (15, *gray.shape) or quality.shape != gray.shape):
        raise ValueError("Unpaired cache geometry/dtype mismatch")
    for values in (edge, probabilities, quality):
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError("Cache values must be finite and in [0,1]")
    if not np.allclose(probabilities.sum(0), 1., atol=4e-6, rtol=0):
        raise ValueError("Semantic class mass differs from one")
