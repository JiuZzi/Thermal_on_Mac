"""Anchored weak-edge proposals, with the original matching/C/R rules frozen.

Only short, direction-compatible ridges measured in the input TIR are traced.
No class, box, image coordinate prior, fitted threshold or generated RGB enters
proposal generation. Existing geometry decisions (including rejected C paths)
stay locked, so this is not also a rematching experiment.
"""

import copy
from collections import Counter
from dataclasses import asdict, dataclass
import math

import numpy as np

from .clarity_semantic_edges import normal_ridges
from .contour_completion import (
    CompletionSettings, _DIRECTIONS, _LENGTHS, _STEPS, _TURN,
    _diagonals, _paint, _path_features, _render_paths, evidence_fields,
)
from .semantic_edges import _unit_map


@dataclass(frozen=True)
class RecallSettings:
    minimum_length_fraction: float = 3 / 288
    minimum_scale_agreement: float = 2 / 3
    beam_width: int = 4
    max_anchors: int = 4096
    max_states_per_anchor: int = 256

    def __post_init__(self):
        for name in ("minimum_length_fraction", "minimum_scale_agreement"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"Finite recall fraction inside (0,1] required: {name}")
        if self.minimum_length_fraction > .15:
            raise ValueError("Recall is restricted to local extensions")
        for name in ("beam_width", "max_anchors", "max_states_per_anchor"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"Positive integer recall budget required: {name}")


def _direction_quality(points):
    coords = np.asarray(points, dtype=np.float64)
    values = np.linalg.eigvalsh(np.cov(coords, rowvar=False, bias=True))
    return float((values[-1] - values[0]) / max(values.sum(), 1e-12))


def _validate_source(metadata, ridges):
    if metadata.get("format") != "local_contour_completion_v1":
        raise ValueError("An original local contour-completion record is required")
    settings = CompletionSettings(**metadata["settings"])
    endpoints = metadata.get("endpoints")
    candidates = metadata.get("candidates")
    if not isinstance(endpoints, list) or not isinstance(candidates, list):
        raise ValueError("Explicit endpoints and candidates required")
    if [e.get("id") for e in endpoints] != list(range(len(endpoints))):
        raise ValueError("Original endpoint IDs must be consecutive")
    if [c.get("id") for c in candidates] != list(range(len(candidates))):
        raise ValueError("Original candidate IDs must be consecutive")
    shape = np.asarray(ridges.shape)
    for e in endpoints:
        point = np.asarray(e.get("point"))
        tangent = np.asarray(e.get("tangent"), dtype=np.float64)
        if (point.shape != (2,) or point.dtype.kind not in "iu" or (point < 0).any()
                or (point >= shape).any() or not ridges[tuple(point)]):
            raise ValueError("Endpoint must be on the supplied original trace map")
        if tangent.shape != (2,) or not np.isfinite(tangent).all() or not np.isclose(np.linalg.norm(tangent), 1, atol=1e-5):
            raise ValueError("A finite unit endpoint tangent is required")
        for key in ("strength", "tangent_quality", "anchor_support"):
            v = e.get(key)
            if not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 + 1e-6:
                raise ValueError(f"Invalid endpoint evidence: {key}")
    # Reuse the same validation as R, without importing or changing its policy.
    from .completion_importance import _validate_candidate
    for c in candidates:
        _validate_candidate(c, ridges.shape)
        ids = c.get("endpoints")
        if not isinstance(ids, list) or len(ids) != 2 or len(set(ids)) != 2 or any(type(i) is not int or not 0 <= i < len(endpoints) for i in ids):
            raise ValueError("Candidate must reference two distinct source endpoints")
        if np.asarray(c["path_yx"])[[0, -1]].tolist() != [endpoints[i]["point"] for i in ids]:
            raise ValueError("Source path and endpoint coordinates disagree")
    if not np.isclose(metadata.get("gap_limit_pixels", -1), max(2., min(ridges.shape) * settings.max_gap_fraction)):
        raise ValueError("Source coordinate scale changed")
    return settings


def _weak_ridges(image, fields, settings, recall):
    # The magnitude denominator is the ORIGINAL noise-aware evidence formula;
    # there is no image-wise percentile, min/max rescale or lower C threshold.
    strengths = np.linalg.norm(fields["vectors"], axis=1)
    enough_scales = (strengths >= settings.support_threshold).mean(0) + 1e-12 >= recall.minimum_scale_agreement
    return normal_ridges(image, fields["strength"]) & enough_scales & (fields["strength"] >= settings.support_threshold)


def _trace(first, weak, blocked, endpoint_at, endpoints, locked, fields, settings, recall, maximum):
    """Bounded deterministic beam; every new pixel has measured ridge support."""
    start = tuple(first["point"])
    tangent = np.asarray(first["tangent"])
    cosine = math.cos(math.radians(settings.max_endpoint_angle))
    minimum = max(3., min(weak.shape) * recall.minimum_length_fraction)
    states = [(0., 0., [start], -1, None)]  # raw cost, arc length, path, heading, closing anchor
    terminals, stops = [], Counter()
    expanded = 0
    for _ in range(int(math.ceil(maximum)) + 1):
        following = []
        for cost, length, path, previous, closing in states:
            if closing is not None:
                terminals.append((cost, length, path, closing)); continue
            y, x = path[-1]
            choices = 0
            for direction, (dy, dx) in enumerate(_STEPS):
                if _DIRECTIONS[direction] @ tangent < cosine:
                    continue  # Outward cone, not an image-up/down or object-shape prior.
                turn = 0. if previous < 0 else float(_TURN[previous, direction])
                if turn > math.pi / 2 + 1e-8:
                    continue
                yy, xx = y + int(dy), x + int(dx)
                if not (0 < yy < weak.shape[0]-1 and 0 < xx < weak.shape[1]-1):
                    stops["image_border"] += 1; continue
                new_length = length + float(_LENGTHS[direction])
                if new_length > maximum + 1e-8:
                    stops["local_length_limit"] += 1; continue
                if (yy, xx) in path:
                    continue
                target = endpoint_at.get((yy, xx))
                if target is not None and (target in locked or target == first["id"]
                        or _DIRECTIONS[direction] @ (-np.asarray(endpoints[target]["tangent"])) < cosine):
                    target = None
                if blocked[yy, xx] and target is None:
                    stops["original_or_locked_boundary"] += 1; continue
                if dy and dx and blocked[y, xx] and blocked[yy, x]:
                    stops["diagonal_crossing"] += 1; continue
                if not weak[yy, xx]:
                    stops["no_measured_weak_ridge"] += 1; continue
                support = float(fields["oriented"][direction, yy, xx])
                if support < settings.support_threshold:
                    stops["gradient_direction_conflict"] += 1; continue
                raw_cost = cost + float(_LENGTHS[direction]) * (1 + settings.image_cost * (1-support)) + settings.bend_cost * (turn/(math.pi/2))**2
                following.append((raw_cost, new_length, path+[(yy, xx)], direction, target))
                choices += 1
            if not choices and length >= minimum and len(path) >= 5:
                terminals.append((cost, length, path, None))
            expanded += 1
            if expanded >= recall.max_states_per_anchor:
                stops["state_budget"] += 1
                break
        if expanded >= recall.max_states_per_anchor or not following:
            break
        # Do not bias toward a short prefix merely because its summed cost is
        # smaller. Stable lexical ties make runs independent of hash ordering.
        following.sort(key=lambda s: (s[0]/max(s[1], 1e-12), -s[1], s[2]))
        if len(following) > recall.beam_width:
            stops["beam_pruned"] += len(following)-recall.beam_width
        states = following[:recall.beam_width]
    # A final beam can terminate at the length budget rather than an absent
    # neighbor. Its last measured point remains a conservative unpainted anchor.
    for cost, length, path, _, closing in states:
        if length >= minimum and len(path) >= 5:
            terminals.append((cost, length, path, closing))
    unique = {}
    for item in terminals:
        cost, length, path, closing = item
        if length < minimum or len(path) < 5:
            continue
        chord = float(np.linalg.norm(np.asarray(path[-1])-path[0]))
        if chord <= math.sqrt(2) or length > settings.max_length_ratio * chord:
            continue
        key = tuple(path)
        unique[key] = (cost/chord, length, path, closing)
    # Prefixes and subpixel alternatives in one ridge are ONE proposal, not
    # independent competing hypotheses. Keep distinct terminal branches for C.
    chosen = []
    duplicate_radius = max(1., min(weak.shape) * 2 / 288)
    for item in sorted(unique.values(), key=lambda s: (-s[1], s[0], s[2])):
        if any(np.linalg.norm(np.asarray(item[2][-1])-other[2][-1]) <= duplicate_radius
               or item[2] == other[2][:len(item[2])] for other in chosen):
            continue
        chosen.append(item)
    return chosen, dict(stops), expanded


def _score(candidate, alternatives, settings):
    """EXACT original C formula and rejection thresholds, on new evidence."""
    margin = max(0., (min(alternatives)-candidate["cost"])/max(candidate["cost"], 1e-12)) if alternatives else 1.
    f = candidate["features"]
    f["alternative_margin"] = float(margin)
    score = (.40*f["mean_support"] + .25*f["support_coverage"]
             + .20*f["cross_scale_agreement"] + .15*f["tangent_quality"])
    score *= .5 + .5*min(1., margin/.2)
    candidate["reliability_score"] = float(np.clip(score, 0, 1))
    failures = candidate["reliability_rejections"]
    if score < settings.reliability_min_score: failures.append("low_score")
    if f["support_coverage"] < settings.reliability_min_coverage: failures.append("low_coverage")
    if f["unsupported_run"] > settings.reliability_max_unsupported_run: failures.append("unsupported_run")
    if margin < settings.reliability_min_margin: failures.append("ambiguous_match")


def recall_completion(image, observed, trace_ridges, metadata, *, settings=None):
    """Append observed weak continuations; preserve every old candidate/C field.

    Open extensions may end in a measured weak fragment. They do not predict an
    invisible object boundary, nor do they cross the old selected geometry.
    R remains a separate subsequent call to prioritize_completion().
    """
    recall = RecallSettings() if settings is None else settings
    if not isinstance(recall, RecallSettings):
        raise TypeError("Use a validated RecallSettings instance")
    image = _unit_map(image, "TIR")
    observed = _unit_map(observed, "observed", image.shape)
    ridges = np.asarray(trace_ridges)
    if ridges.dtype != np.bool_ or ridges.shape != image.shape or min(image.shape) < 8:
        raise ValueError("Aligned boolean original trace map, at least 8 pixels required")
    if "weak_recall" in metadata:
        raise ValueError("Do not append recall twice to the same source")
    original_settings = _validate_source(metadata, ridges)
    updated = copy.deepcopy(metadata)
    endpoints, candidates = updated["endpoints"], updated["candidates"]
    original_count = len(candidates)
    fields = evidence_fields(image)
    weak = _weak_ridges(image, fields, original_settings, recall)
    blocked = ridges.copy()
    painted = np.zeros_like(ridges)
    diagonal_cells, locked = {}, set()
    for c in candidates:
        if not c["geometry_selected"]: continue
        locked.update(c["endpoints"])
        path = np.asarray(c["path_yx"], dtype=int)
        blocked[tuple(path.T)] = True
        painted[tuple(path.T)] = True
        diagonal_cells.update(_diagonals(path))
    original_endpoints = len(endpoints)
    endpoint_at = {tuple(e["point"]):e["id"] for e in endpoints}
    available = sorted((e for e in endpoints if e["id"] not in locked), key=lambda e:(-e["strength"], e["id"]))
    budget_ids = {e["id"] for e in available[:recall.max_anchors]}
    ledger, totals = [], Counter()
    for first in endpoints[:original_endpoints]:
        row = dict(endpoint_id=first["id"], y=first["point"][0], x=first["point"][1],
                   anchor_strength=first["strength"], candidates=0, status="", expanded_states=0)
        if first["id"] in locked:
            row["status"] = "legacy_geometry_locked"
        elif first["id"] not in budget_ids:
            row["status"] = "anchor_budget"
        elif first["anchor_support"] < original_settings.geometry_min_support:
            row["status"] = "anchor_image_conflict"
        else:
            proposals, stops, expanded = _trace(first, weak, blocked, endpoint_at, endpoints,
                locked, fields, original_settings, recall, metadata["gap_limit_pixels"])
            row.update(expanded_states=expanded, stops=stops)
            for cost, length, path, target in proposals:
                if target is None:
                    chain = np.asarray(path[-max(3, int(math.ceil(min(image.shape)*original_settings.tangent_fraction))):])
                    direction = np.asarray(path[-2])-path[-1]
                    direction = direction/np.linalg.norm(direction)
                    second = dict(id=len(endpoints), point=list(path[-1]), tangent=direction.tolist(),
                        tangent_quality=_direction_quality(chain), strength=float(fields["strength"][path[-1]]),
                        source="measured_weak_terminal")
                    endpoints.append(second)
                else:
                    second = endpoints[target]
                p = np.asarray(path, dtype=np.int32)
                f = _path_features(p, first, second, fields, original_settings)
                candidates.append(dict(id=len(candidates), endpoints=[first["id"],second["id"]],
                    path_yx=p.tolist(), gap=float(np.linalg.norm(p[-1]-p[0])), length=float(length), cost=float(cost),
                    strength=float(min(first["strength"],second["strength"],f["mean_support"])),
                    features=f, geometry_selected=False, reliability_accepted=False,
                    geometry_rejection=None, reliability_rejections=[], source="anchored_weak_observation",
                    closing_original_endpoint=target is not None))
                row["candidates"] += 1
            row["status"] = "proposed" if proposals else "no_eligible_weak_continuation"
        totals[row["status"]] += 1
        ledger.append(row)
    new = candidates[original_count:]
    costs_by_endpoint = {}
    for c in candidates:
        for i in c["endpoints"]: costs_by_endpoint.setdefault(i, []).append((c["id"],c["cost"]))
    for c in new:
        alternatives = [cost for i in c["endpoints"] for j,cost in costs_by_endpoint[i] if j != c["id"]]
        _score(c, alternatives, original_settings)
    for c in sorted(new, key=lambda c:(c["cost"],c["id"])):
        path = np.asarray(c["path_yx"],dtype=int)
        if locked & set(c["endpoints"]):
            c["geometry_rejection"]="endpoint_already_used";continue
        diagonal = _diagonals(path)
        if painted[tuple(path.T)].any() or any(cell in diagonal_cells and diagonal_cells[cell] != slope for cell,slope in diagonal.items()):
            c["geometry_rejection"]="candidate_crossing";continue
        c["geometry_selected"]=True
        c["reliability_accepted"]=not c["reliability_rejections"]
        locked.update(c["endpoints"])
        painted[tuple(path.T)]=True;diagonal_cells.update(diagonal)
    old_geometry, old_reliable = _render_paths(image.shape, metadata["candidates"], original_settings.soft_width)
    geometry, reliable = _render_paths(image.shape, candidates, original_settings.soft_width)
    _, new_reliable = _render_paths(image.shape, new, original_settings.soft_width)
    new_mask, _, _ = _paint(image.shape, new, accepted_only=True)
    if (reliable < old_reliable).any() or (geometry < old_geometry).any():
        raise AssertionError("Recall must preserve the exact old geometry and C output")
    maps = dict(observed=observed.copy(), baseline=np.maximum(observed,old_reliable),
        recalled=np.maximum(observed,reliable), weak_ridges=weak, recall_mask=new_mask,
        extra_reliable=new_reliable)
    updated["weak_recall"] = dict(format="anchored_weak_recall_v1", settings=asdict(recall),
        original_candidates=original_count,original_endpoints=original_endpoints,
        new_candidates=len(new),new_geometry_selected=sum(c["geometry_selected"] for c in new),
        new_reliability_accepted=sum(c["reliability_accepted"] for c in new),
        endpoint_status_counts=dict(totals),endpoint_ledger=ledger,
        C_formula="EXACT original support/coverage/cross-scale/tangent + alternative margin; thresholds unchanged",
        matching="Original geometry locks preserved, including C-rejected paths; no alternative rematching",
        rendering="Same per-path Gaussian and C; measured weak terminal is an unpainted anchor",
        amplitude="min(original anchor E, terminal measured support, path mean directional support)",
        limits="Open observed continuations, not inferred invisible surfaces, semantic correction or calibrated confidence")
    for values in maps.values():
        if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            raise FloatingPointError("Invalid recalled contour")
    return maps, updated


def source_diagnostics(metadata):
    """Stage counts are NOT gap recall or correctness; reasons can overlap."""
    cs=metadata["candidates"]
    selected=[c for c in cs if c["geometry_selected"]]
    return dict(endpoints=len(metadata["endpoints"]),admissible_paths=len(cs),
        geometry_selected=len(selected),reliability_accepted=sum(c["reliability_accepted"] for c in selected),
        pair_rejections=dict(metadata.get("pair_rejections",{})),
        geometry_rejections=dict(Counter(c["geometry_rejection"] for c in cs if c["geometry_rejection"])),
        C_rejection_reasons=dict(Counter(r for c in selected if not c["reliability_accepted"] for r in c["reliability_rejections"])),
        interpretation="Descriptive stage counts, overlapping failures; no independent true-gap annotations")
