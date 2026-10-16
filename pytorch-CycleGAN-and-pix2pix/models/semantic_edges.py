"""Frozen semantic responses guide existing soft contours; not edge confidence.

This opt-in preview helper changes only C2c's fine/coarse importance map.
It does not register a new training mode or change existing C2 checkpoints.
"""

import numpy as np

from .saliency_edges import (
    make_adaptive_saliency_components,
    make_adaptive_saliency_edge,
    soft_multi_at_sigma,
    structure_texture_importance,
)


FMB_CLASS_NAMES = (
    "background", "road", "sidewalk", "building", "lamp", "sign", "vegetation",
    "sky", "person", "car", "truck", "bus", "motorcycle", "bicycle", "pole",
)
IMPORTANT_CLASSES = ("person", "car", "truck", "bus", "motorcycle", "bicycle", "building")
SETTINGS = {
    "high_thresholds": (0.12, 0.16, 0.20),
    "low_ratio": 0.5,
    "soft_width": 1.0,
    "sigmas": (0.7, 1.0, 1.6),
    "background_gain": 0.5,
    "importance_min": 0.18,
    "importance_max": 0.85,
    "texture_attenuation": 0.4,
    "fine_floor": 0.25,
}


def _unit_map(value, name, shape=None):
    values = np.asarray(value, dtype=np.float32)
    if values.ndim != 2 or not values.size or (shape is not None and values.shape != shape):
        raise ValueError(f"{name} must be a nonempty 2D map matching the TIR coordinates")
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError(f"{name} must contain finite values in [0,1]")
    return values


def semantic_importance(probabilities, class_names=FMB_CLASS_NAMES):
    """Return a bounded importance prior from original-coordinate class scores.

    All selected classes have equal importance; no manually tuned per-class
    gain, segmentation threshold, box input or morphology is used.
    """
    names = tuple(str(name) for name in class_names)
    if len(names) != len(FMB_CLASS_NAMES) or set(names) != set(FMB_CLASS_NAMES):
        raise ValueError("Expected the complete, unique 15-class FMB taxonomy")
    values = np.asarray(probabilities, dtype=np.float32)
    if values.ndim != 3 or values.shape[0] != len(names) or not all(values.shape[1:]):
        raise ValueError("Expected [15,H,W] original-coordinate semantic responses")
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Semantic responses must contain finite values in [0,1]")
    if not np.allclose(values.sum(0), 1.0, rtol=0, atol=3e-6):
        raise ValueError("Categorical semantic responses must sum to one")
    ids = [names.index(name) for name in IMPORTANT_CLASSES]
    support = np.clip(values[ids].sum(0), 0.0, 1.0).astype(np.float32)
    importance = SETTINGS["importance_min"] + (
        SETTINGS["importance_max"] - SETTINGS["importance_min"]
    ) * support
    return importance.astype(np.float32), support


def fuse_candidates(fine, coarse, clutter, importance):
    """Keep the same C2c fusion, texture penalty and nonzero fine-edge floor."""
    fine = _unit_map(fine, "fine")
    coarse = _unit_map(coarse, "coarse", fine.shape)
    clutter = _unit_map(clutter, "clutter", fine.shape)
    importance = _unit_map(importance, "importance", fine.shape)
    if np.any(importance < SETTINGS["importance_min"] - 1e-6) or np.any(
        importance > SETTINGS["importance_max"] + 1e-6
    ):
        raise ValueError("Importance must stay in C2c's [0.18,0.85] range")
    result = importance * fine + (1.0 - importance) * SETTINGS["background_gain"] * coarse
    result *= 1.0 - SETTINGS["texture_attenuation"] * clutter
    return np.clip(np.maximum(result, SETTINGS["fine_floor"] * fine), 0, 1).astype(np.float32)


def make_semantic_contours(image, probabilities, class_names=FMB_CLASS_NAMES):
    """Return unchanged C2b/C2c controls plus an R-only semantic candidate.

    Supply semantic responses inferred from the same full TIR image and
    restored to its coordinates. Review boxes never enter this function.
    """
    image = _unit_map(image, "TIR")
    importance, support = semantic_importance(probabilities, class_names)
    if importance.shape != image.shape:
        raise ValueError("Semantic responses must match the processed TIR coordinates")
    c2c, base_importance, fine, coarse = make_adaptive_saliency_edge(
        image, SETTINGS["high_thresholds"], SETTINGS["low_ratio"], SETTINGS["soft_width"],
        SETTINGS["sigmas"], SETTINGS["background_gain"], variant="structure_texture",
    )
    _, clutter = structure_texture_importance(image, fine, coarse)
    c2b = soft_multi_at_sigma(
        image, 1.0, SETTINGS["high_thresholds"], SETTINGS["low_ratio"], SETTINGS["soft_width"],
    )
    return dict(
        c2b=c2b, c2c=c2c, semantic512=fuse_candidates(fine, coarse, clutter, importance),
        semantic_importance=importance, semantic_support=support,
        base_importance=base_importance, fine=fine, coarse=coarse, clutter=clutter,
    )


def structure_semantic_importance(base_importance, semantic_weight, semantic_support,
                                 fine, coarse, clutter):
    """Add bounded structural support where semantics would suppress detail.

    The semantic-only route remains intact. Strong semantic support, dense
    texture and a coarse-only candidate close the structural correction.
    This is an importance heuristic, not calibrated edge confidence.
    """
    base = _unit_map(base_importance, "structure importance")
    semantic_weight = _unit_map(semantic_weight, "semantic importance", base.shape)
    support = _unit_map(semantic_support, "semantic support", base.shape)
    fine = _unit_map(fine, "fine", base.shape)
    coarse = _unit_map(coarse, "coarse", base.shape)
    clutter = _unit_map(clutter, "clutter", base.shape)
    for values in (base, semantic_weight):
        if np.any(values < SETTINGS["importance_min"] - 1e-6) or np.any(
            values > SETTINGS["importance_max"] + 1e-6
        ):
            raise ValueError("Both importance priors must stay in [0.18,0.85]")
    # Increasing R would reduce a coarse-only route. Restrict the correction
    # to positive fine-candidate advantage and normalize without a new cutoff.
    advantage = np.clip(
        (fine - SETTINGS["background_gain"] * coarse) / (fine + 1e-6), 0, 1,
    ).astype(np.float32)
    gate = ((1.0 - support) * (1.0 - clutter) * advantage).astype(np.float32)
    headroom = np.maximum(base - semantic_weight, 0.0)
    joint = semantic_weight + gate * headroom
    return joint.astype(np.float32), gate, advantage


def make_structure_semantic_condition(image, probabilities, class_names=FMB_CLASS_NAMES):
    """Return only joint512 for training, with unchanged candidates and fusion.

    Each C2c candidate is computed once; the structural importance and clutter
    are reused. C2b, the original C2c output and semantic-v1 preview maps are
    not generated on this path.
    """
    image = _unit_map(image, "TIR")
    importance, support = semantic_importance(probabilities, class_names)
    if importance.shape != image.shape:
        raise ValueError("Semantic responses must match the processed TIR coordinates")
    fine, coarse, base, clutter = make_adaptive_saliency_components(
        image, SETTINGS["high_thresholds"], SETTINGS["low_ratio"], SETTINGS["soft_width"],
        SETTINGS["sigmas"], SETTINGS["background_gain"], variant="structure_texture",
    )
    joint, _, _ = structure_semantic_importance(
        base, importance, support, fine, coarse, clutter,
    )
    return fuse_candidates(fine, coarse, clutter, joint)


def make_structure_semantic_contours(image, probabilities, class_names=FMB_CLASS_NAMES):
    """Opt-in structural protection; keep every semantic-v1 control unchanged."""
    maps = make_semantic_contours(image, probabilities, class_names)
    joint, gate, advantage = structure_semantic_importance(
        maps["base_importance"], maps["semantic_importance"], maps["semantic_support"],
        maps["fine"], maps["coarse"], maps["clutter"],
    )
    maps.update(
        joint512=fuse_candidates(maps["fine"], maps["coarse"], maps["clutter"], joint),
        joint_importance=joint, structure_protection_gate=gate,
        candidate_advantage=advantage,
    )
    return maps
