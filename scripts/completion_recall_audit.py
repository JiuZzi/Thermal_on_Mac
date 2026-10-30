"""Review-only diagnostics; erased observed edges are NOT real-gap truth."""

import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.ndimage import convolve, distance_transform_edt
from skimage.draw import line


def select_erasure(ridges, observed, seed, length):
    """Seeded selection on the observed graph BEFORE running either algorithm.

    Follow degree-two chains with six observed points on each side. This is a
    controlled deletion of a measured ridge, not annotation of a natural gap.
    Selection uses neither method's success nor target classes/boxes.
    """
    degree = convolve(ridges.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant")-ridges
    coordinates = np.argwhere(ridges & (degree == 2) & (observed >= .03))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(coordinates))
    required = length+12
    for index in order[:2000]:
        start = tuple(int(v) for v in coordinates[index])
        path, seen = [start], {start}
        while len(path) < required:
            y, x = path[-1]
            neighbors = [(yy, xx) for yy in range(max(0,y-1),min(ridges.shape[0],y+2))
                         for xx in range(max(0,x-1),min(ridges.shape[1],x+2))
                         if (yy,xx) not in seen and ridges[yy,xx] and degree[yy,xx] == 2]
            if not neighbors:break
            next_point = sorted(neighbors)[0]
            path.append(next_point);seen.add(next_point)
        if len(path) != required:continue
        p=np.asarray(path)
        # Exclude tiny loops and strongly branching/jagged traces. No image
        # gradient criterion or algorithm result is used to cherry-pick success.
        if np.linalg.norm(p[-1]-p[0]) < .75*(len(p)-1):continue
        if any(min(point)<=1 or point[0]>=ridges.shape[0]-2 or point[1]>=ridges.shape[1]-2 for point in path):continue
        return [list(point) for point in path[6:6+length]]
    return None


def erasure_mask(shape, polyline, radius=1):
    trace=np.zeros(shape,bool)
    coords=np.asarray(polyline,dtype=int)
    trace[tuple(coords.T)]=True
    return distance_transform_edt(~trace)<=radius


def candidate_mask(metadata, shape, stage):
    mask=np.zeros(shape,bool)
    for c in metadata["candidates"]:
        if stage=="geometry" and not c["geometry_selected"]:continue
        if stage=="reliable" and not c["reliability_accepted"]:continue
        p=np.asarray(c["path_yx"],dtype=int)[1:-1]
        mask[tuple(p.T)]=True
    return mask


def coverage(truth, prediction, tolerance=1.):
    if not truth.any():return None
    if not prediction.any():return 0.
    return float((distance_transform_edt(~prediction)[truth]<=tolerance).mean())


def control_statistics(truth, erased, original_ridges, baseline_metadata, recalled_metadata,
                       baseline_increment, recalled_increment):
    """Known-ridge recovery and local spill, NEVER natural-boundary accuracy."""
    result={}
    for name,metadata,increment in (("baseline",baseline_metadata,baseline_increment),
                                     ("recall",recalled_metadata,recalled_increment)):
        for stage in ("all","geometry","reliable"):
            predicted=candidate_mask(metadata,truth.shape,stage)
            result[f"{name}_{stage}_coverage_1px"]=coverage(truth,predicted)
        visible=increment>=1/255
        result[f"{name}_visible_coverage_1px"]=coverage(truth,visible)
        # Only the erased patch has a supplied reference; other real additions
        # are unlabelled. Do not call all outside-patch additions false positives.
        reliable=candidate_mask(metadata,truth.shape,"reliable")
        result[f"{name}_off_reference_pixels_inside_erasure"]=int((reliable & erased & (distance_transform_edt(~original_ridges)>1)).sum())
        result[f"{name}_unlabelled_addition_pixels_outside_erasure"]=int((reliable & ~erased).sum())
    return result


def read_annotations(path, prefixes):
    if path is None:return []
    data=json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("format")!="completion_gap_annotations_v1" or not isinstance(data.get("gaps"),list):
        raise ValueError("Explicit completion_gap_annotations_v1 JSON required")
    seen=set()
    for gap in data["gaps"]:
        if not isinstance(gap.get("id"),str) or not gap["id"] or gap["id"] in seen:
            raise ValueError("Unique nonempty annotation IDs required")
        seen.add(gap["id"])
        if gap.get("prefix") not in prefixes or gap.get("label") not in ("visible_boundary","no_boundary","uncertain"):
            raise ValueError("Unknown frame or gap label")
        points=np.asarray(gap.get("polyline_yx"))
        if points.ndim!=2 or points.shape[1]!=2 or len(points)<2 or points.dtype.kind not in "iuf" or not np.isfinite(points).all():
            raise ValueError("At least two finite [y,x] annotation points required")
    return data["gaps"]


def annotation_statistics(gap, shape, old_metadata, new_metadata, old_increment, new_increment):
    points=np.asarray(gap["polyline_yx"],dtype=float)
    if (points<0).any() or (points>np.asarray(shape)-1).any():raise ValueError("Annotation outside input coordinates")
    trace=np.zeros(shape,bool)
    rounded=np.rint(points).astype(int)
    for a,b in zip(rounded[:-1],rounded[1:]):
        rr,cc=line(*a,*b);trace[rr,cc]=True
    row=dict(id=gap["id"],prefix=gap["prefix"],label=gap["label"],points=int(trace.sum()),notes=gap.get("notes",""))
    for name,metadata,increment in (("baseline",old_metadata,old_increment),("recall",new_metadata,new_increment)):
        for stage in ("all","geometry","reliable"):
            row[f"{name}_{stage}_coverage_1px"]=coverage(trace,candidate_mask(metadata,shape,stage))
        row[f"{name}_visible_coverage_1px"]=coverage(trace,increment>=1/255)
        stages=[row[f"{name}_{s}_coverage_1px"] for s in ("all","geometry","reliable","visible")]
        # A practical localization label with its threshold explicitly recorded;
        # continuous coverages are retained. Proximity cannot establish ownership.
        row[f"{name}_bottleneck_at_half_coverage"]=next((s for s,v in zip(
            ("no_candidate","geometry_conflict","C_rejected","accepted_but_weak"),stages) if v<.5),"accepted")
    return row


def aggregate_diagnostics(rows):
    counts=Counter()
    categories={key:Counter() for key in ("pair_rejections","geometry_rejections","C_rejection_reasons")}
    for row in rows:
        for key in ("endpoints","admissible_paths","geometry_selected","reliability_accepted"):
            counts[key]+=row[key]
        for key in categories:categories[key].update(row[key])
    return dict(**dict(counts),**{k:dict(v) for k,v in categories.items()},
        note="Reasons overlap; candidates and stage counts are not real missing-boundary recall")
