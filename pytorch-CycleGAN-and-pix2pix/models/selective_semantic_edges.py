"""Opt-in contour selection; the original joint512 algorithm stays unchanged.

Importance controls contribution, not correctness. This module only narrows or
attenuates existing responses; it neither completes gaps nor predicts confidence.
"""

import numpy as np

from .semantic_edges import (
    FMB_CLASS_NAMES, SETTINGS, _unit_map, fuse_candidates, semantic_importance,
    structure_semantic_importance,
)
from .saliency_edges import make_adaptive_saliency_components


VARIANTS = ("joint512", "selective_v1", "clarity_v1", "recall_v1")
SELECTIVE_SETTINGS = {
    "algorithm": "importance_gain_and_normal_ridge_v1",
    "gradient_sigma": 1.0,
    "ridge_distance_pixels": 1.0,
    "soft_width": SETTINGS["soft_width"],
    "gain_min": SETTINGS["fine_floor"],
    "output_floor": "original unmerged fine candidate times existing fine_floor",
    "candidate_settings": SETTINGS,
    "prefilter": "none; previous bilateral trial weakened small contours",
    "completion": False,
    "confidence": False,
}


def importance_gain(importance):
    """Separate overall contribution from the existing fine/coarse mixing R."""
    value = _unit_map(importance, "joint importance")
    lower, upper = SETTINGS["importance_min"], SETTINGS["importance_max"]
    if np.any(value < lower - 1e-6) or np.any(value > upper + 1e-6):
        raise ValueError("Joint importance must stay in the declared range")
    normalized = np.clip((value - lower) / (upper - lower), 0, 1)
    floor = SELECTIVE_SETTINGS["gain_min"]
    return (floor + (1 - floor) * normalized).astype(np.float32)


def consolidate_response(image, response):
    """Locate normal-direction ridges of scale votes, then re-encode softly.

    Canny already thins each individual detection. Here the response being
    consolidated is the *combined* scale/threshold vote. Keep its local ridges,
    thin connected plateaus and restore distance softness with the original
    width. Capping by the input response prevents invented response outside the
    existing candidate. Near-zero thermal gradients use the original response.
    """
    from scipy.ndimage import gaussian_filter, map_coordinates, sobel, distance_transform_edt
    from skimage.morphology import thin

    gray = _unit_map(image, "TIR")
    value = _unit_map(response, "candidate", gray.shape)
    if not value.any():
        return value.copy(), np.zeros(gray.shape, dtype=bool)
    smooth = gaussian_filter(gray, SELECTIVE_SETTINGS["gradient_sigma"])
    gx, gy = sobel(smooth, axis=1) / 8, sobel(smooth, axis=0) / 8
    magnitude = np.hypot(gx, gy)
    epsilon = np.finfo(np.float32).eps
    valid = magnitude > epsilon
    nx = np.divide(gx, magnitude, out=np.zeros_like(gx), where=valid)
    ny = np.divide(gy, magnitude, out=np.zeros_like(gy), where=valid)
    yy, xx = np.indices(gray.shape, dtype=np.float32)
    distance = SELECTIVE_SETTINGS["ridge_distance_pixels"]
    ahead = map_coordinates(value, [yy + distance * ny, xx + distance * nx], order=1, mode="nearest")
    behind = map_coordinates(value, [yy - distance * ny, xx - distance * nx], order=1, mode="nearest")
    ridge = thin(valid & (value > epsilon) & (value >= ahead) & (value >= behind))
    if not ridge.any():
        return value.copy(), ridge
    distances, indices = distance_transform_edt(~ridge, return_indices=True)
    peaks = value[indices[0], indices[1]]
    restored = peaks * np.exp(-.5 * (distances / SELECTIVE_SETTINGS["soft_width"]) ** 2)
    consolidated = np.minimum(value, restored).astype(np.float32)
    consolidated[~valid] = value[~valid]
    return consolidated, ridge


def selective_components(image, probabilities, class_names=FMB_CLASS_NAMES):
    """Return independently inspectable gain-only, ridge-only and joint controls."""
    image = _unit_map(image, "TIR")
    semantic, support = semantic_importance(probabilities, class_names)
    if semantic.shape != image.shape:
        raise ValueError("Semantic responses must match the processed TIR coordinates")
    fine, coarse, structure, clutter = make_adaptive_saliency_components(
        image, SETTINGS["high_thresholds"], SETTINGS["low_ratio"], SETTINGS["soft_width"],
        SETTINGS["sigmas"], SETTINGS["background_gain"], variant="structure_texture",
    )
    joint, _, _ = structure_semantic_importance(structure, semantic, support, fine, coarse, clutter)
    legacy = fuse_candidates(fine, coarse, clutter, joint)
    gain = importance_gain(joint)
    merged_fine, fine_ridges = consolidate_response(image, fine)
    merged_coarse, coarse_ridges = consolidate_response(image, coarse)
    # Hold R and clutter fixed across this ablation: only candidate geometry changes.
    merged = fuse_candidates(merged_fine, merged_coarse, clutter, joint)
    # R is importance, not proof of correctness. Preserve the OLD fine floor
    # from the unmerged candidate: a semantic miss must not multiply it away.
    floor = SETTINGS["fine_floor"] * fine
    selected = np.maximum(floor, gain * merged).astype(np.float32)
    return dict(joint512=legacy, gain_only=np.maximum(floor, gain * legacy).astype(np.float32),
                ridge_only=np.maximum(floor, merged).astype(np.float32),
                selective_v1=selected, importance_gain=gain, preserved_floor=floor,
                joint_importance=joint, semantic_support=support, clutter=clutter,
                fine=fine, coarse=coarse, merged_fine=merged_fine, merged_coarse=merged_coarse,
                fine_ridges=fine_ridges.astype(np.float32), coarse_ridges=coarse_ridges.astype(np.float32))


def make_selective_semantic_condition(image, probabilities, class_names=FMB_CLASS_NAMES):
    return selective_components(image, probabilities, class_names)["selective_v1"]
