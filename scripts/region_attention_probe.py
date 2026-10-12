"""Untrained, image-only regional attention probe; never imported by training.

Connected grayscale regions supply region/context contrast, not class labels,
object probabilities, edge reliability, or new edges. Parameters are fixed for
one feasibility experiment, not selected against annotation scores.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, find_objects, gaussian_filter
from skimage.segmentation import felzenszwalb


PARAMETERS = dict(segmentation_scale=100., segmentation_sigma=.8,
                  minimum_region_pixels=20, context_radius=8,
                  contrast_floor=.03, spatial_sigma=1.,
                  importance_min=.18, importance_max=.85, region_mix=.5)


def validate_map(array):
    array = np.asarray(array, dtype=np.float32)
    if array.ndim != 2 or not array.size:
        raise ValueError("Expected a nonempty 2D map")
    if not np.isfinite(array).all() or np.any((array < 0) | (array > 1)):
        raise ValueError("Expected finite values in [0,1]")
    return array


def region_attention(gray, old_importance):
    """Return R and explanatory maps, using ONLY TIR and the old image-only R.

For each connected segment, compare its median to its eight-pixel exterior
ring. Internal and context absolute deviations reduce the contrast score.
No hot-only preference, center prior, semantic masks, candidate brightness,
annotation boxes, checkpoints, or cross-frame fitting enter this function.
"""
    gray, old_importance = validate_map(gray), validate_map(old_importance)
    if gray.shape != old_importance.shape:
        raise ValueError("TIR and importance must have identical shapes")
    p = PARAMETERS
    labels = felzenszwalb(gray, scale=p["segmentation_scale"],
                         sigma=p["segmentation_sigma"],
                         min_size=p["minimum_region_pixels"], channel_axis=None)
    labels = labels.astype(np.int32)
    # find_objects expects labels starting at one.
    bounds = find_objects(labels + 1)
    count = len(bounds)
    medians = np.zeros(count, np.float32)
    scores = np.zeros(count, np.float32)
    internal = np.zeros(count, np.float32)
    contextual = np.zeros(count, np.float32)
    contrasts = np.zeros(count, np.float32)
    radius = p["context_radius"]
    for label, slices in enumerate(bounds):
        ys, xs = slices
        y0, y1 = max(0, ys.start-radius), min(gray.shape[0], ys.stop+radius)
        x0, x1 = max(0, xs.start-radius), min(gray.shape[1], xs.stop+radius)
        local_gray = gray[y0:y1, x0:x1]
        inside = labels[y0:y1, x0:x1] == label
        ring = binary_dilation(inside, iterations=radius) & ~inside
        values = local_gray[inside]
        center = np.median(values)
        medians[label] = center
        internal[label] = np.mean(np.abs(values-center))
        if not ring.any():
            continue
        outside = local_gray[ring]
        neighbor = np.median(outside)
        contextual[label] = np.mean(np.abs(outside-neighbor))
        contrast = abs(center-neighbor)
        contrasts[label] = contrast
        scores[label] = contrast / (contrast + internal[label] + contextual[label]
                                    + p["contrast_floor"])
    region_score = gaussian_filter(scores[labels], p["spatial_sigma"])
    regional_R = p["importance_min"] + (p["importance_max"]-p["importance_min"]) * region_score
    importance = ((1-p["region_mix"]) * old_importance + p["region_mix"] * regional_R).astype(np.float32)
    return importance, dict(labels=labels, region_median=medians[labels],
                            region_score=region_score.astype(np.float32),
                            regional_R=regional_R.astype(np.float32),
                            region_internal_deviation=internal[labels],
                            region_context_deviation=contextual[labels],
                            region_contrast=contrasts[labels])


def match_mean(baseline, target):
    """Diagnostic uniform gain (including dimming) matches a full-image mean."""
    baseline, target = validate_map(baseline), validate_map(target)
    if baseline.shape != target.shape:
        raise ValueError("Mean control needs matching maps")
    desired = target.mean(dtype=np.float64)
    if desired == 0:
        return np.zeros_like(baseline), 0.
    if desired > np.mean(baseline > 0) + 1e-8:
        raise ValueError("Target mean exceeds the baseline support capacity")
    response = lambda gain: np.minimum(baseline.astype(np.float64) * gain, 1).astype(np.float32)
    lo, hi = 0., 1.
    while response(hi).mean(dtype=np.float64) < desired and hi < 2**40:
        hi *= 2
    if response(hi).mean(dtype=np.float64) < desired - 1e-7:
        raise ValueError("Cannot match target mean")
    for _ in range(48):
        middle = (lo+hi)/2
        if response(middle).mean(dtype=np.float64) < desired:
            lo = middle
        else:
            hi = middle
    return response((lo+hi)/2), (lo+hi)/2
