"""Fixed, class-agnostic edge-use interventions; NOT learned confidence.

These probes test spatial attenuation before authorizing a trainable gate.
No image IDs, coordinates, class names, target boxes, or generated RGB enter
the rule. Original TIR/P/Q and the reserved third channel are unchanged.
"""

from dataclasses import asdict, dataclass
import math

import numpy as np
import torch
from torch.nn import functional as F

from D.unpaired.losses import gradients


VARIANTS = ("original", "uniform", "selective")


@dataclass(frozen=True)
class ProbeSettings:
    suppression: float = .75
    window_fraction: float = 9 / 288
    trace_threshold: float = .03
    semantic_radius_fraction: float = 2 / 288

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 < value < 1:
                raise ValueError(f"Finite fraction inside (0,1) required: {name}")
        if self.window_fraction > .25 or self.semantic_radius_fraction > .05:
            raise ValueError("Local context required")


def validate_inputs(x, p, q):
    if (not torch.is_tensor(x) or x.ndim != 4 or x.shape[1] != 3 or x.shape[0] < 1 or
        min(x.shape[-2:]) < 8 or not x.is_floating_point() or not torch.isfinite(x).all() or
        (x < 0).any() or (x > 1).any() or torch.count_nonzero(x[:, 2])):
        raise ValueError("Expected finite aligned [TIR01,E01,zero]")
    if (not torch.is_tensor(p) or not torch.is_tensor(q) or p.ndim != 4 or p.shape[0] != x.shape[0] or p.shape[1] < 2 or
        p.shape[-2:] != x.shape[-2:] or q.shape != x[:, :1].shape or
        p.device != x.device or q.device != x.device):
        raise ValueError("P/Q coordinates and device must match X")
    for values in (p, q):
        if not values.is_floating_point() or not torch.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            raise ValueError("Finite probability/quality values in [0,1] required")
    if not torch.allclose(p.float().sum(1), torch.ones_like(p[:, 0], dtype=torch.float32), atol=4e-6, rtol=0):
        raise ValueError("Semantic probability mass differs from one")


def _mean(values, size):
    radius = size // 2
    return F.avg_pool2d(F.pad(values, (radius,)*4, mode="reflect"), size, 1)


def _window(shape, fraction):
    size = max(3, int(round(min(shape)*fraction)))
    size += 1-size % 2
    return min(size, 2*(min(shape)-1)+1)


@torch.no_grad()
def probe_maps(x, p, q, *, settings=None):
    settings = ProbeSettings() if settings is None else settings
    if not isinstance(settings, ProbeSettings):
        raise TypeError("Use validated ProbeSettings")
    validate_inputs(x, p, q)
    image, E, p, q = x[:, :1].float(), x[:, 1:2].float(), p.float(), q.float()
    window = _window(E.shape[-2:], settings.window_fraction)
    radius = max(1, int(round(min(E.shape[-2:])*settings.semantic_radius_fraction)))
    # Q is eroded near boundaries in the existing cache. Read nearby support,
    # without claiming it detects confidently wrong semantic classifications.
    context_q = F.max_pool2d(q, 2*radius+1, 1, radius)
    boundary = torch.zeros_like(E)
    for axis in (-1, -2):
        a, b = [slice(None)]*4, [slice(None)]*4
        a[axis], b[axis] = slice(1, None), slice(None, -1)
        a, b = tuple(a), tuple(b)
        difference = .5*(p[a]-p[b]).abs().sum(1, keepdim=True)
        boundary[a] = torch.maximum(boundary[a], difference)
        boundary[b] = torch.maximum(boundary[b], difference)
    boundary = F.max_pool2d(boundary, 2*radius+1, 1, radius) * context_q
    density = _mean((E >= settings.trace_threshold).float(), window)
    gx, gy = gradients(image)
    xx, yy, xy = _mean(gx.square(), window), _mean(gy.square(), window), _mean(gx*gy, window)
    coherence = ((xx-yy).square()+4*xy.square()).clamp_min(0).sqrt() / (xx+yy+1e-12)
    # Per-image disjoint Haar-HH noise estimate. Never pool across the batch.
    even = image[:, :, :image.shape[-2]//2*2, :image.shape[-1]//2*2]
    hh = .5*(even[:,:,::2,::2]-even[:,:,1::2,::2]-even[:,:,::2,1::2]+even[:,:,1::2,1::2])
    noise = hh.abs().flatten(1).median(1).values[:,None,None,None] / .6744897501960817
    magnitude = _mean((gx.square()+gy.square()).sqrt(), window)
    structural_backup = coherence.clamp(0,1) * magnitude/(magnitude+.75*noise+1/255)
    # Interior context favors attenuation; reliable class transitions and
    # coherent measured structures weaken it. No whole class is set to zero.
    suppression = settings.suppression * context_q * (1-boundary) * (.5+.5*density) * (1-.5*structural_backup)
    G = (1-suppression).clamp(1-settings.suppression, 1)
    maps = dict(gate=G, semantic_boundary=boundary, context_quality=context_q,
        edge_density=density, structural_backup=structural_backup, selective_edge=E*G)
    for name, values in maps.items():
        if not torch.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            raise FloatingPointError("Invalid edge-use diagnostic: "+name)
    return maps


def _resize(values, size):
    return F.interpolate(values.float(), size=size, mode="bilinear", align_corners=False)


@torch.no_grad()
def interventions(x, p, q, *, model_size=None, settings=None):
    """Three X variants, matching uniform/selective mass AFTER model resize.

    model_size is (height,width). Formal D uses native X; the zero-shot pipeline
    resizes E before ControlNet. Means here match the actual float E input.
    """
    maps = probe_maps(x, p, q, settings=settings)
    E = x[:, 1:2].float()
    size = tuple(E.shape[-2:]) if model_size is None else tuple(model_size)
    if len(size) != 2 or any(type(v) is not int or v < 8 for v in size):
        raise ValueError("Model size must contain two integers >=8")
    original_model, selective_model = _resize(E, size), _resize(maps['selective_edge'], size)
    before = original_model.cpu().numpy().sum((1,2,3), dtype=np.float64)
    after = selective_model.cpu().numpy().sum((1,2,3), dtype=np.float64)
    ratios = np.divide(after, before, out=np.ones_like(after), where=before > 0)
    ratio = torch.as_tensor(ratios, device=E.device, dtype=torch.float32)[:,None,None,None]
    variants = {name:x.clone() for name in VARIANTS}
    variants['uniform'][:,1:2] = E*ratio
    variants['selective'][:,1:2] = maps['selective_edge']
    for name, condition in variants.items():
        assert torch.equal(condition[:, :1], x[:, :1]) and torch.equal(condition[:, 2:], x[:, 2:])
        if (condition[:,1:2] > E+1e-7).any(): raise AssertionError("A suppression probe cannot add edges")
    uniform_model = _resize(variants['uniform'][:,1:2], size)
    uniform_mass = uniform_model.cpu().numpy().sum((1,2,3), dtype=np.float64)
    if not np.allclose(uniform_mass, after, atol=2e-6, rtol=2e-6):
        raise AssertionError("Actual model-input mass is not matched")
    diagnostics = [dict(uniform_gain=float(ratios[i]), original_model_mass=float(before[i]),
        selective_model_mass=float(after[i]), uniform_model_mass=float(uniform_mass[i]),
        edge_weighted_gate=float(ratios[i]), gate_is_learned=False, correctness_probability=False)
        for i in range(len(x))]
    return variants, maps, diagnostics
