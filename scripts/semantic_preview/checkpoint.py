"""Restricted checkpoint loading and strict architecture validation."""

from collections import defaultdict
import hashlib
from pathlib import Path
import sys
import typing

import numpy as np
import torch

from .model import InfraredSegFormer

ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metadata_safe_globals():
    try:
        import omegaconf
    except ImportError:
        local = ROOT / "runs/semantic_preview_dependencies"
        if local.is_dir():
            sys.path.insert(0, str(local))
        try:
            import omegaconf
        except ImportError as error:
            raise ImportError("Install the small metadata dependency: python -m pip install -r scripts/requirements_semantic_preview.txt") from error
    from omegaconf.base import ContainerMetadata, Metadata
    from omegaconf.nodes import AnyNode
    from omegaconf import DictConfig, ListConfig
    # Never fall back to unrestricted pickle. These are data/config classes.
    scalar = np._core.multiarray.scalar if hasattr(np, "_core") else np.core.multiarray.scalar
    allowed = [DictConfig, ListConfig, ContainerMetadata, Metadata, AnyNode,
               defaultdict, typing.Any, list, dict, int, float, str, bool, np.dtype,
               (scalar, "numpy.core.multiarray.scalar"),
               (scalar, "numpy._core.multiarray.scalar")]
    allowed += list({type(np.dtype(kind)) for kind in ("f4", "f8", "i4", "i8", "u4", "u8", "b1")})
    return allowed


def validate_state_dict(state):
    if not isinstance(state, dict) or not state:
        raise ValueError("Checkpoint contains no model state dictionary")
    if not all(isinstance(name, str) and torch.is_tensor(value) for name, value in state.items()):
        raise ValueError("The model state dictionary must contain tensors only")
    if all(name.startswith("module.") for name in state):
        state = {name[7:]: value for name, value in state.items()}
    expected = {"encoder.patch_embed1.proj.weight": (64, 3, 7, 7),
                "decoder.linear_c1.proj.weight": (256, 64),
                "decoder.linear_pred.weight": (15, 256, 1, 1),
                "decoder.linear_pred.bias": (15,), "classifier.weight": (15, 512, 1, 1)}
    for name, shape in expected.items():
        if name not in state or tuple(state[name].shape) != shape:
            raise ValueError(f"Expected complete FMB 15-class SegFormer-B3; incompatible tensor: {name}. Use best_model.pth, not the encoder-only mit_b3.pth.")
    for stage, depth in enumerate((3, 4, 18, 3), 1):
        blocks = {int(name.split('.')[2]) for name in state if name.startswith(f"encoder.block{stage}.")}
        if blocks != set(range(depth)):
            raise ValueError(f"Unexpected MiT-B3 depth in stage {stage}: {sorted(blocks)}")
    for name, value in state.items():
        if value.is_floating_point() and not torch.isfinite(value).all().item():
            raise ValueError(f"Non-finite checkpoint tensor: {name}")
    return state


def load_checkpoint(path, device):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing complete segmentation checkpoint: {path}")
    if not hasattr(torch.serialization, "safe_globals"):
        raise RuntimeError("Restricted checkpoint loading requires a recent PyTorch (recommended >=2.6); use a separate preview environment rather than changing an active training environment")
    allowed = metadata_safe_globals()
    if hasattr(torch.serialization, "get_unsafe_globals_in_checkpoint"):
        present = torch.serialization.get_unsafe_globals_in_checkpoint(path)
        names = {item[1] if isinstance(item, tuple) else f"{item.__module__}.{item.__qualname__}" for item in allowed}
        if set(present) - names:
            raise ValueError(f"Unrecognized checkpoint metadata types: {sorted(set(present) - names)}")
    with torch.serialization.safe_globals(allowed):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    state = validate_state_dict(state)
    cfg = checkpoint.get("config", checkpoint.get("cfg"))
    if cfg is not None:
        from omegaconf import OmegaConf
        cfg = OmegaConf.to_container(cfg, resolve=False) if OmegaConf.is_config(cfg) else cfg
        if not isinstance(cfg, dict):
            raise ValueError("Unexpected checkpoint config format")
        if cfg.get("exp", {}).get("backbone", "mit_b3") != "mit_b3" or cfg.get("dataset", {}).get("num_classes", 15) != 15:
            raise ValueError("Checkpoint training configuration does not match FMB B3 / 15 classes")
    info = {"sha256": sha256_file(path), "file_bytes": path.stat().st_size,
            "backbone": "mit_b3", "num_classes": 15, "embedding_dim": 256,
            "load": "restricted weights_only=True, strict=True",
            "config": cfg,
            "checkpoint_fields": list(checkpoint),
            "tensor_count": len(state)}
    for name in ("epoch", "iteration", "iter", "best_miou", "best_mIoU"):
        value = checkpoint.get(name)
        if isinstance(value, (int, float, str, np.generic)):
            info[name] = value.item() if isinstance(value, np.generic) else value
    model = InfraredSegFormer()
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False).to(device)
    info["parameter_count"] = sum(value.numel() for value in model.parameters())
    return model, info
