"""Bounded structural importance recall on the unchanged clarity candidates.

This is neither person detection nor semantic correction. Only existing edges
can gain contribution; semantic P, colour guidance and reserved confidence stay
unchanged. Defaults of all older variants stay unchanged.
"""

import numpy as np

from .clarity_semantic_edges import CLARITY_SETTINGS, clarity_components
from .saliency_edges import (
    REGION_RECALL_SETTINGS, region_recall_importance, structure_texture_importance,
)
from .selective_semantic_edges import importance_gain
from .semantic_edges import FMB_CLASS_NAMES, SETTINGS, _unit_map, fuse_candidates


RECALL_SETTINGS = {
    "algorithm": "clarity_with_bounded_local_importance_recall_v1",
    "reference": CLARITY_SETTINGS,
    "region_recall": REGION_RECALL_SETTINGS,
    "clutter_gate": "1-D; D recomputed from unchanged raw fine/coarse candidates",
    "candidate_advantage": "current clarity fine - background_gain * current clarity coarse",
    "importance": "Rjoint + (1-D) * (Rlocal-Rjoint)",
    "gain": "same importance_gain evaluated at recalled importance",
    "floor": "unchanged clarity raw-fine ridge floor",
    "semantic_correction": False, "completion": False, "confidence": False,
}


def recall_importance(image, fine, coarse, base, clutter):
    """Increase R only on locally supported, positive fine-advantage pixels."""
    image = _unit_map(image, "TIR")
    base = _unit_map(base, "joint importance", image.shape)
    clutter = _unit_map(clutter, "clutter", image.shape)
    importance_gain(base)  # Enforce the declared importance range.
    local, debug = region_recall_importance(image, fine, coarse, base,
                                            background_gain=SETTINGS["background_gain"])
    corrected = (base + (1-clutter)*(local-base)).astype(np.float32)
    return corrected, debug


def recall_components(image, probabilities, class_names=FMB_CLASS_NAMES):
    """Keep geometry fixed, and expose R-only changes for reproducible review."""
    maps, metadata = clarity_components(image, probabilities, class_names, include_probes=False)
    # No additional segmentation or Canny pass. Reuse the raw candidates for D.
    structure, clutter = structure_texture_importance(image, maps["raw_fine"], maps["raw_coarse"])
    corrected, debug = recall_importance(image, maps["clarity_v1_fine"], maps["clarity_v1_coarse"],
                                         maps["joint_importance"], clutter)
    gain = importance_gain(corrected)
    fused = fuse_candidates(maps["clarity_v1_fine"], maps["clarity_v1_coarse"], clutter, corrected)
    # Positive current-candidate advantage makes F nondecreasing in R; g is
    # nondecreasing too. Do not max against an old full map to hide attenuation.
    maps.update(recall_v1=np.maximum(maps["preserved_ridge_floor"], gain*fused).astype(np.float32),
                recall_importance=corrected, recall_gain=gain, clutter=clutter,
                structure_importance=structure, **debug)
    return maps, metadata


def make_recall_semantic_condition(image, probabilities, class_names=FMB_CLASS_NAMES):
    return recall_components(image, probabilities, class_names)[0]["recall_v1"]
