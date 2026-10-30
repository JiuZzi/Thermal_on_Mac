"""Reweight existing completion paths without changing geometry or reliability.

Task relevance R is separate from structural reliability C. There is no fit,
per-image ranking quota, position mask, box input, or dataset-name branch.
"""

import copy
from dataclasses import dataclass, asdict
import math

import numpy as np
from scipy.ndimage import map_coordinates

from .contour_completion import _render_paths
from .semantic_edges import FMB_CLASS_NAMES, _unit_map


@dataclass(frozen=True)
class ImportanceSettings:
    minimum_gain: float = .25
    structural_backup: float = .5
    side_offset_fraction: float = .35
    task_classes: tuple = ("person", "car", "truck", "bus", "motorcycle", "bicycle", "lamp", "sign", "pole")

    def __post_init__(self):
        for name in ("minimum_gain", "structural_backup", "side_offset_fraction"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"Importance setting must be inside (0,1]: {name}")
        if not isinstance(self.task_classes, tuple) or len(set(self.task_classes)) != len(self.task_classes):
            raise ValueError("Task classes must be a tuple of unique names")
        if any(not isinstance(name, str) or not name for name in self.task_classes):
            raise ValueError("Nonempty task class names required")


def _sample(values, positions):
    return map_coordinates(values, positions.T, order=1, mode="nearest")


def _context(path, gap, shape, fraction):
    middle = path[1:-1].astype(np.float64)
    index = np.arange(1, len(path)-1)
    window = max(1, int(math.ceil(gap / 4)))
    direction = path[np.minimum(index + window, len(path)-1)] - path[np.maximum(index-window, 0)]
    direction = direction / np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-12)
    normal = np.stack((-direction[:, 1], direction[:, 0]), axis=1)
    offset = max(1., gap * fraction)
    left, right = middle + offset * normal, middle - offset * normal
    def inside(points):
        return ((points >= 0) & (points <= np.asarray(shape)-1)).all(1)
    return middle, left, right, inside(left), inside(right)


def _validate_candidate(candidate, shape):
    if not isinstance(candidate.get("id"), int) or isinstance(candidate["id"], bool):
        raise ValueError("Candidate ID must be an integer")
    for name in ("geometry_selected", "reliability_accepted"):
        if not isinstance(candidate.get(name), bool):
            raise ValueError(f"Candidate flag must be boolean: {name}")
    if candidate["reliability_accepted"] and not candidate["geometry_selected"]:
        raise ValueError("Reliable path must belong to the selected geometry")
    for name in ("strength", "reliability_score"):
        value = candidate.get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Invalid candidate value: {name}")
    raw = np.asarray(candidate.get("path_yx"))
    if raw.ndim != 2 or raw.shape[1] != 2 or len(raw) < 3 or raw.dtype.kind not in "iu":
        raise ValueError("Integer candidate path [N>=3,2] required")
    path = raw.astype(np.int64)
    if (path < 0).any() or (path >= np.asarray(shape)).any():
        raise ValueError("Candidate path outside original coordinates")
    steps = np.abs(np.diff(path, axis=0))
    if (steps > 1).any() or np.any(steps.max(1) == 0):
        raise ValueError("Candidate path must have adjacent distinct grid points")
    return path


def prioritize_completion(observed, metadata, structure, clutter, probabilities=None, quality=None,
                          *, class_names=FMB_CLASS_NAMES, settings=None):
    """Apply fixed per-path R to an immutable completed-path record.

    structure is normalized to [0,1] by the caller. quality is semantic Q, not C.
    Missing semantics or zero Q uses the exact same structural fallback. Side
    sampling avoids reading eroded semantic Q only on the boundary itself.
    """
    settings = ImportanceSettings() if settings is None else settings
    if not isinstance(settings, ImportanceSettings):
        raise TypeError("Use a validated ImportanceSettings instance")
    observed = _unit_map(observed, "observed")
    structure = _unit_map(structure, "normalized structure importance", observed.shape)
    clutter = _unit_map(clutter, "clutter", observed.shape)
    width = metadata.get("settings", {}).get("soft_width")
    if not isinstance(width, (int, float)) or not math.isfinite(width) or width <= 0:
        raise ValueError("Source metadata must define a positive soft width")
    if not isinstance(metadata.get("candidates"), list):
        raise ValueError("Source candidates must be an explicit list")
    candidates = metadata["candidates"]
    if len({c.get("id") for c in candidates}) != len(candidates):
        raise ValueError("Duplicate candidate identities")
    has_semantics = probabilities is not None
    if not has_semantics and quality is not None:
        raise ValueError("Semantic quality requires probabilities")
    if has_semantics:
        p = np.asarray(probabilities, dtype=np.float32)
        names = tuple(class_names)
        if (p.ndim != 3 or p.shape[1:] != observed.shape or p.shape[0] != len(names)
            or len(set(names)) != len(names) or len(names) < 2 or not np.isfinite(p).all()
            or (p < 0).any() or (p > 1).any()
            or not np.allclose(p.sum(0), 1, atol=4e-6, rtol=0)):
            raise ValueError("Invalid semantic coordinates, taxonomy or probabilities")
        if set(settings.task_classes) - set(names):
            raise ValueError("Configure task_classes explicitly for this semantic taxonomy")
        if quality is None:
            raise ValueError("Explicit semantic Q required; do not assume correctness")
        q = _unit_map(quality, "semantic quality", observed.shape)
        task_indices = [names.index(name) for name in settings.task_classes]
        task = p[task_indices].sum(0) if task_indices else np.zeros_like(q)

    updated = copy.deepcopy(metadata)
    gain_map = np.zeros_like(observed)
    importance_map = np.zeros_like(observed)
    weighted_candidates = []
    for original, candidate in zip(candidates, updated["candidates"]):
        path = _validate_candidate(original, observed.shape)
        gap = float(np.linalg.norm(path[-1] - path[0]))
        middle, left, right, valid_left, valid_right = _context(path, gap, observed.shape, settings.side_offset_fraction)
        structural = float(np.mean(_sample(structure, middle) * (1 - _sample(clutter, middle))))
        semantic, context_q, task_support, boundary_support = 0., 0., 0., 0.
        if has_semantics:
            qc = _sample(q, middle)
            ql, qr = _sample(q, left) * valid_left, _sample(q, right) * valid_right
            context_q = float(np.maximum.reduce((qc, ql, qr)).mean())
            task_support = float(np.maximum.reduce((_sample(task, middle)*qc,
                _sample(task, left)*ql, _sample(task, right)*qr)).mean())
            # Any reliable region boundary can matter, including curbs, trees,
            # buildings and categories outside the selected task-class group.
            disagreement = .5 * sum(np.abs(_sample(channel, left) - _sample(channel, right)) for channel in p)
            boundary_support = float(np.mean(np.minimum(ql, qr) * disagreement))
            semantic = max(task_support, boundary_support)
        fused = semantic + (1 - context_q) * structural
        importance = float(np.clip(max(fused, settings.structural_backup * structural), 0, 1))
        gain = settings.minimum_gain + (1 - settings.minimum_gain) * importance
        candidate.update(add_importance=importance, importance_gain=float(gain), importance_features=dict(
            structural_fallback=structural, semantic_task_support=task_support,
            semantic_boundary_support=boundary_support, semantic_context_quality=context_q,
            semantic_evidence=semantic, backup_floor=settings.structural_backup * structural))
        weighted_candidates.append(dict(original, strength=original["strength"] * gain))
        if original["reliability_accepted"]:
            y, x = path[1:-1].T
            gain_map[y, x] = np.maximum(gain_map[y, x], gain)
            importance_map[y, x] = np.maximum(importance_map[y, x], importance)

    _, original_added = _render_paths(observed.shape, candidates, width)
    _, weighted_added = _render_paths(observed.shape, weighted_candidates, width)
    baseline = np.maximum(observed, original_added)
    priority = np.maximum(observed, weighted_added)
    original_increment, priority_increment = baseline-observed, priority-observed
    old_sum, new_sum = (float(v.sum(dtype=np.float64)) for v in (original_increment, priority_increment))
    uniform_gain = min(1., max(0., new_sum / old_sum)) if old_sum > 0 else 1.
    uniform = observed + original_increment * uniform_gain
    if (priority > baseline).any() or (priority < observed).any():
        raise AssertionError("Importance may only attenuate additions, never change observed E")
    maps = dict(observed=observed.copy(), baseline=baseline, uniform=uniform, priority=priority,
        baseline_increment=original_increment, uniform_increment=uniform-observed,
        priority_increment=priority_increment, importance_map=importance_map, gain_map=gain_map)
    for key, values in maps.items():
        if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            raise FloatingPointError(f"Invalid importance output: {key}")
    updated["importance"] = dict(format="completion_importance_v1", settings=asdict(settings),
        semantic_available=has_semantics, uniform_mean_control_gain=uniform_gain,
        uniform_control="Only additions scaled, global mean matched AFTER max with observed E",
        semantics="Class-agnostic boundary support + configurable task classes; structural backup always present",
        limits="Fixed importance heuristic; no fitting, no calibrated correctness, no new geometry or C changes")
    return maps, updated
