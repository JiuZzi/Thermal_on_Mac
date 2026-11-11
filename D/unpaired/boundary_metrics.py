"""Match annotated target contours at a fixed tolerance in native test pixels."""

from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.spatial import cKDTree
from skimage.feature import canny

from scripts.audit_c1_edges import load_mask
from scripts.evaluate_distribution import load_gray
from .data import ConditionDataset


TOLERANCE = 2.0
CONDITION_THRESHOLD = 0.5
CANNY = dict(sigma=1.0, low_threshold=0.08, high_threshold=0.16)


def boundary_files(sources, gt_dir, roi_dir):
    labels = {path.stem: path for path in Path(gt_dir).glob("*.png")}
    rois = {path.stem: path for path in Path(roi_dir).glob("*.png")}
    if not labels or labels.keys() != rois.keys() or labels.keys() - {row["stem"] for row in sources}:
        raise ValueError("Provide a nonempty subset of test stems with one GT and one valid-ROI PNG per frame")
    return {stem: (labels[stem], rois[stem]) for stem in sorted(labels)}


def matched_counts(prediction, annotation, valid):
    predicted = np.argwhere(prediction & valid)
    reference = np.argwhere(annotation & valid)
    n_pred, n_gt = len(predicted), len(reference)
    matched = 0
    if n_pred and n_gt:
        edges = cKDTree(predicted).sparse_distance_matrix(cKDTree(reference), TOLERANCE, output_type="coo_matrix")
        matched = int(np.count_nonzero(maximum_bipartite_matching(edges.tocsr()) >= 0))
    return matched, n_pred, n_gt


def scores(counts):
    matched, predicted, annotated = map(int, counts)
    return dict(matched_pixels=matched, predicted_pixels=predicted, annotated_pixels=annotated,
                precision=matched / predicted if predicted else None,
                recall=matched / annotated if annotated else None,
                f1=2 * matched / (predicted + annotated) if predicted + annotated else None)


def evaluate_boundaries(prediction_dir, cache_dir, labels):
    dataset = ConditionDataset(cache_dir, "TIR", "test")
    by_stem = {row["stem"]: index for index, row in enumerate(dataset.frames)}
    totals = {name: np.zeros(3, dtype=np.int64) for name in ("condition_edge", "generated_rgb_canny")}
    frames = {}
    for stem, (gt_path, roi_path) in labels.items():
        sample = dataset[by_stem[stem]]
        size = tuple(sample["frame"]["native_size"])
        annotation, valid = (load_mask(path, "full", size) for path in (gt_path, roi_path))
        if not valid.any():
            raise ValueError(f"Empty annotated ROI: {roi_path}")
        condition = sample["x"][0, 1].numpy()
        gray = load_gray(prediction_dir / "images" / f"{stem}_fake_B.png", rgb_weights=True)
        if condition.shape != annotation.shape or gray.shape != annotation.shape:
            raise ValueError(f"Boundary scoring requires native, aligned coordinates for {stem}")
        maps = {"condition_edge": condition >= CONDITION_THRESHOLD, "generated_rgb_canny": canny(gray, **CANNY)}
        frames[stem] = {}
        for name, edge in maps.items():
            counts = matched_counts(edge, annotation, valid)
            frames[stem][name] = scores(counts)
            totals[name] += counts
    if not totals["condition_edge"][2]:
        raise ValueError("No independently annotated boundary pixels inside the supplied ROIs")
    return dict(tolerance_px=TOLERANCE, matching="one-to-one", pooling="micro over all annotated pixels",
                condition_threshold=CONDITION_THRESHOLD, rgb_canny=CANNY,
                overall={name: scores(counts) for name, counts in totals.items()}, frames=frames)
