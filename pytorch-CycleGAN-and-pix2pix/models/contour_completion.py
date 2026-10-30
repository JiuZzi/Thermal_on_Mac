"""Opt-in, observation-constrained contour completion; no training integration.

All four comparisons preserve the same observed soft contour. Geometry proposes
local paths between existing fragments; the reliability arm only rejects or
attenuates those SAME selected paths. Scores are heuristics, not probabilities.
No class templates, target boxes, generated RGB, or dataset-specific filenames
enter the algorithm.
"""

from dataclasses import asdict, dataclass
import heapq
import math

import numpy as np
from scipy.ndimage import convolve, gaussian_filter, map_coordinates, sobel
from scipy.spatial import cKDTree
from skimage.morphology import binary_closing, thin

from .clarity_semantic_edges import noise_sigma, soften_ridges
from .semantic_edges import _unit_map


MODES = ("observed", "closing", "geometry", "reliable")


@dataclass(frozen=True)
class CompletionSettings:
    max_gap_fraction: float = .035
    tangent_fraction: float = .018
    trace_threshold: float = .03
    max_endpoint_angle: float = 65.
    max_length_ratio: float = 1.6
    image_cost: float = 2.
    bend_cost: float = .6
    support_threshold: float = .25
    geometry_min_support: float = .10
    reliability_min_score: float = .58
    reliability_min_coverage: float = .45
    reliability_max_unsupported_run: float = .65
    reliability_min_margin: float = .08
    soft_width: float = .5
    max_neighbors: int = 4
    max_searches: int = 512
    max_search_states: int = 6000

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f"Finite numeric completion setting required: {name}")
        for name in ("max_neighbors", "max_searches", "max_search_states"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"Positive integer required: {name}")
        for name in ("trace_threshold", "support_threshold", "geometry_min_support",
                     "reliability_min_score", "reliability_min_coverage",
                     "reliability_max_unsupported_run", "reliability_min_margin"):
            if not 0 < getattr(self, name) < 1:
                raise ValueError(f"Completion fraction must be inside (0,1): {name}")
        if not 0 < self.max_gap_fraction <= .15 or not 0 < self.tangent_fraction <= .1:
            raise ValueError("Local search/tangent fractions outside their supported range")
        if not 0 < self.max_endpoint_angle < 90 or not 1 <= self.max_length_ratio <= 3:
            raise ValueError("Invalid endpoint angle or detour ratio")
        if self.image_cost < 0 or self.bend_cost < 0 or self.soft_width <= 0:
            raise ValueError("Nonnegative costs and positive soft width required")


_STEPS = np.asarray(((-1, -1), (-1, 0), (-1, 1), (0, -1),
                     (0, 1), (1, -1), (1, 0), (1, 1)), dtype=np.int32)
_LENGTHS = np.linalg.norm(_STEPS, axis=1)
_DIRECTIONS = _STEPS / _LENGTHS[:, None]
_TURN = np.arccos(np.clip(_DIRECTIONS @ _DIRECTIONS.T, -1, 1))


def _anchored_thinning(mask, protected):
    """Zhang-Suen thinning deleting only newly filled pixels, never anchors.

    Thinning the entire closed mask and then subtracting the old ridges would
    mistake displaced medial axes for newly completed object boundaries.
    """
    result = np.asarray(mask, dtype=bool).copy()
    protected = np.asarray(protected, dtype=bool)
    changed = True
    while changed:
        changed = False
        for phase in (0, 1):
            padded = np.pad(result, 1)
            p = (padded[:-2, 1:-1], padded[:-2, 2:], padded[1:-1, 2:], padded[2:, 2:],
                 padded[2:, 1:-1], padded[2:, :-2], padded[1:-1, :-2], padded[:-2, :-2])
            neighbors = sum(v.astype(np.uint8) for v in p)
            transitions = sum((~p[i] & p[(i+1) % 8]).astype(np.uint8) for i in range(8))
            if phase == 0:
                corners = ~(p[0] & p[2] & p[4]) & ~(p[2] & p[4] & p[6])
            else:
                corners = ~(p[0] & p[2] & p[6]) & ~(p[0] & p[4] & p[6])
            remove = result & ~protected & (neighbors >= 2) & (neighbors <= 6) & (transitions == 1) & corners
            if remove.any():
                result[remove] = False
                changed = True
    return result


def evidence_fields(image):
    """Direction-dependent weak-gradient support at three generic scales.

    Values stay zero for flat input; no per-image min/max or percentile stretch.
    Gradient polarity is ignored. Cross-scale agreement is correlated evidence,
    not three independent sensor observations.
    """
    image = _unit_map(image, "TIR")
    scale = min(image.shape) / 288.
    sigma_noise = noise_sigma(image)
    oriented, strengths, vectors = [], [], []
    for sigma in (.7, 1., 1.6):
        smooth = gaussian_filter(image, max(.45, sigma * scale), mode="reflect")
        gy, gx = sobel(smooth, axis=0) / 8., sobel(smooth, axis=1) / 8.
        magnitude = np.hypot(gy, gx)
        strength = magnitude / (magnitude + .75 * sigma_noise + 1 / 255.)
        normals = np.stack((-_DIRECTIONS[:, 1], _DIRECTIONS[:, 0]), axis=1)
        alignment = np.abs(normals[:, 0, None, None] * gy + normals[:, 1, None, None] * gx)
        alignment /= np.maximum(magnitude, 1e-12)[None]
        oriented.append(np.clip(alignment * strength[None], 0, 1).astype(np.float32))
        strengths.append(strength.astype(np.float32))
        vectors.append((np.stack((gy, gx)) / (magnitude + .75 * sigma_noise + 1 / 255.)).astype(np.float32))
    scales = np.stack(oriented)
    return dict(oriented=scales.mean(0), scales=scales, vectors=np.stack(vectors),
                strength=np.stack(strengths).mean(0), noise_sigma=sigma_noise)


def _neighbors(point, ridges):
    y, x = point
    h, w = ridges.shape
    return [(y + int(dy), x + int(dx)) for dy, dx in _STEPS
            if 0 <= y + dy < h and 0 <= x + dx < w and ridges[y + dy, x + dx]]


def fragment_endpoints(ridges, observed, settings):
    """Estimate outward tangents from traced fragments, not one noisy pixel."""
    degree = convolve(ridges.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant")
    degree = degree - ridges.astype(np.uint8)
    points = np.argwhere(ridges & (degree == 1))
    trace_length = max(3, int(math.ceil(min(ridges.shape) * settings.tangent_fraction)))
    endpoints = []
    for point in points:
        start = tuple(int(v) for v in point)
        if min(start) <= 0 or start[0] >= ridges.shape[0] - 1 or start[1] >= ridges.shape[1] - 1:
            continue  # Image truncation is not a missing object boundary.
        chain, seen = [start], {start}
        while len(chain) < trace_length:
            following = [p for p in _neighbors(chain[-1], ridges) if p not in seen]
            if len(following) != 1:
                break
            chain.append(following[0]); seen.add(following[0])
        if len(chain) < 3:
            continue
        coords = np.asarray(chain, dtype=np.float64)
        covariance = np.cov(coords, rowvar=False, bias=True)
        values, vectors = np.linalg.eigh(covariance)
        tangent = vectors[:, -1]
        if tangent @ (coords[0] - coords[-1]) < 0:
            tangent = -tangent
        strength = float(np.median(observed[tuple(coords.astype(int).T)]))
        if strength < settings.trace_threshold:
            continue
        endpoints.append(dict(id=len(endpoints), point=start, tangent=tangent,
                              tangent_quality=float((values[-1] - values[0]) / max(values.sum(), 1e-12)),
                              strength=strength, trace_length=len(chain), trace_yx=chain))
    return endpoints


def _find_path(first, second, ridges, fields, settings, gap):
    """Bounded A* in (y,x,direction); costs are nonnegative and deterministic."""
    start, goal = first["point"], second["point"]
    h, w = ridges.shape
    margin = int(math.ceil(gap * .5)) + 1
    y0, x0 = max(0, min(start[0], goal[0]) - margin), max(0, min(start[1], goal[1]) - margin)
    y1, x1 = min(h, max(start[0], goal[0]) + margin + 1), min(w, max(start[1], goal[1]) + margin + 1)
    cosine = math.cos(math.radians(settings.max_endpoint_angle))
    initial = (*start, -1)
    best, parents = {initial: 0.}, {}
    heap = [(1., 0., initial)]
    expanded = 0

    def blocked(y, x):
        # Permit only the two anchors, never walk along another observed edge.
        return bool(ridges[y, x] and (y, x) not in (start, goal))

    while heap:
        _, cost, state = heapq.heappop(heap)
        if cost > best.get(state, math.inf):
            continue
        y, x, previous = state
        if (y, x) == goal and previous >= 0:
            if _DIRECTIONS[previous] @ (-second["tangent"]) < cosine:
                continue
            states = [state]
            while states[-1] != initial:
                states.append(parents[states[-1]])
            path = np.asarray([(s[0], s[1]) for s in states[::-1]], dtype=np.int32)
            length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
            if length > settings.max_length_ratio * gap:
                return None, "detour", expanded
            return (path, float(cost), length), None, expanded
        expanded += 1
        if expanded > settings.max_search_states:
            return None, "search_budget", expanded
        for direction, (dy, dx) in enumerate(_STEPS):
            if previous < 0 and _DIRECTIONS[direction] @ first["tangent"] < cosine:
                continue
            turn = 0. if previous < 0 else float(_TURN[previous, direction])
            if turn > math.pi / 2 + 1e-8:
                continue
            yy, xx = y + int(dy), x + int(dx)
            if not (y0 <= yy < y1 and x0 <= xx < x1) or blocked(yy, xx):
                continue
            if dy and dx and blocked(y, xx) and blocked(yy, x):
                continue  # Do not jump diagonally through an observed crossing.
            if math.hypot(yy - start[0], xx - start[1]) + math.hypot(yy - goal[0], xx - goal[1]) > settings.max_length_ratio * gap:
                continue
            support = float(fields["oriented"][direction, yy, xx])
            increment = (_LENGTHS[direction] * (1 + settings.image_cost * (1 - support))
                         + settings.bend_cost * (turn / (math.pi / 2)) ** 2) / gap
            new_cost = cost + increment
            new_state = (yy, xx, direction)
            if new_cost + 1e-12 < best.get(new_state, math.inf):
                best[new_state], parents[new_state] = new_cost, state
                heuristic = math.hypot(yy - goal[0], xx - goal[1]) / gap
                heapq.heappush(heap, (new_cost + heuristic, new_cost, new_state))
    return None, "no_path", expanded


def _longest_run(mask):
    longest = current = 0
    for value in mask:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def _path_features(path, first, second, fields, settings):
    # Score missing pixels ONLY; strong anchors must not certify an empty gap.
    interior = path[1:-1]
    # Neighborhood tangents prevent a zigzag from claiming support from a
    # single unrelated transverse edge. Use a fraction of this local gap.
    window = max(1, int(math.ceil(np.linalg.norm(path[-1] - path[0]) / 4)))
    indices = np.arange(1, len(path)-1)
    tangents = path[np.minimum(indices + window, len(path)-1)] - path[np.maximum(indices-window, 0)]
    tangents = tangents / np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-12)
    normals = np.stack((-tangents[:, 1], tangents[:, 0]), axis=1)
    y, x = interior.T
    scales = np.abs(fields["vectors"][:, 0, y, x] * normals[:, 0]
                    + fields["vectors"][:, 1, y, x] * normals[:, 1])
    support = scales.mean(0)
    cross_scale = (scales >= settings.support_threshold).mean(0)
    supported = support >= settings.support_threshold
    return dict(mean_support=float(support.mean()),
                support_coverage=float(supported.mean()),
                cross_scale_agreement=float(cross_scale.mean()),
                unsupported_run=float(_longest_run(~supported) / len(interior)),
                tangent_quality=min(first["tangent_quality"], second["tangent_quality"]))


def _diagonals(path):
    result = {}
    for a, b in zip(path[:-1], path[1:]):
        dy, dx = b - a
        if dy and dx:
            result[(int(min(a[0], b[0])), int(min(a[1], b[1])))] = int(dy * dx)
    return result


def _paint(shape, paths, *, weighted=False, accepted_only=False):
    mask = np.zeros(shape, bool)
    amplitude = np.zeros(shape, np.float32)
    score_map = np.zeros(shape, np.float32)
    for candidate in paths:
        if not candidate["geometry_selected"] or (accepted_only and not candidate["reliability_accepted"]):
            continue
        pixels = np.asarray(candidate["path_yx"], dtype=int)[1:-1]
        y, x = pixels.T
        mask[y, x] = True
        strength = candidate["strength"] * (candidate["reliability_score"] if weighted else 1.)
        amplitude[y, x] = np.maximum(amplitude[y, x], strength)
        score_map[y, x] = np.maximum(score_map[y, x], candidate["reliability_score"])
    return mask, amplitude, score_map


def _render_paths(shape, candidates, width):
    """Max of per-path Gaussians: rejecting a path can NEVER brighten a pixel.

    Rendering all ridges in one nearest-neighbor distance transform would not
    have that property when two paths have different amplitudes. Truncate only
    beyond eight sigma (relative tail < 1.3e-14) for bounded local computation.
    """
    geometry, reliable = np.zeros(shape, np.float32), np.zeros(shape, np.float32)
    padding = max(1, int(math.ceil(8 * width)))
    for candidate in candidates:
        if not candidate["geometry_selected"]:
            continue
        points = np.asarray(candidate["path_yx"], dtype=int)[1:-1]
        y0, x0 = np.maximum(0, points.min(0) - padding)
        y1, x1 = np.minimum(shape, points.max(0) + padding + 1)
        mask = np.zeros((y1-y0, x1-x0), bool)
        mask[tuple((points - (y0, x0)).T)] = True
        values = soften_ridges(mask, np.full(mask.shape, candidate["strength"], np.float32), width)
        geometry[y0:y1, x0:x1] = np.maximum(geometry[y0:y1, x0:x1], values)
        if candidate["reliability_accepted"]:
            weighted = values * candidate["reliability_score"]
            reliable[y0:y1, x0:x1] = np.maximum(reliable[y0:y1, x0:x1], weighted)
    return geometry, reliable


def completion_comparison(image, observed, *, trace_ridges=None, settings=None):
    """Return aligned four-arm float maps and explicit candidate provenance.

    The caller may provide unweighted candidate ridges from clarity extraction;
    otherwise the observed map is skeletonized at a recorded fixed threshold.
    P/Q are deliberately not used as hard edge truth in this first experiment.
    """
    settings = CompletionSettings() if settings is None else settings
    if not isinstance(settings, CompletionSettings):
        raise TypeError("Use a validated CompletionSettings instance")
    image = _unit_map(image, "TIR")
    observed = _unit_map(observed, "observed contour", image.shape)
    if min(image.shape) < 8:
        raise ValueError("Completion requires image dimensions of at least 8 pixels")
    if trace_ridges is None:
        ridges = thin(observed >= settings.trace_threshold)
    else:
        trace_ridges = np.asarray(trace_ridges)
        if trace_ridges.dtype != np.bool_ or trace_ridges.shape != image.shape:
            raise ValueError("Trace ridges must be aligned boolean centerlines")
        ridges = thin(trace_ridges.copy())
    # All arms use exactly the same supported trace map, including closing.
    from scipy.ndimage import maximum_filter, distance_transform_edt
    ridges &= maximum_filter(observed, size=3) >= settings.trace_threshold
    ridges[[0, -1], :] = False; ridges[:, [0, -1]] = False
    gap_limit = max(2., min(image.shape) * settings.max_gap_fraction)
    radius = max(1, int(math.floor(gap_limit / 2)))
    closed = binary_closing(ridges, footprint=np.ones((2 * radius + 1, 2 * radius + 1), bool))
    closing_added = _anchored_thinning(closed | ridges, ridges) & ~ridges
    closing_added[[0, -1], :] = False; closing_added[:, [0, -1]] = False
    if ridges.any():
        _, nearest = distance_transform_edt(~ridges, return_indices=True)
        closing_amplitude = observed[nearest[0], nearest[1]].astype(np.float32)
    else:
        closing_amplitude = np.zeros_like(observed)
    closing_edge = soften_ridges(closing_added, closing_amplitude, settings.soft_width)
    fields = evidence_fields(image)
    endpoints = fragment_endpoints(ridges, observed, settings)
    for endpoint in endpoints:
        chain = np.asarray(endpoint["trace_yx"], dtype=int)
        y, x = chain.T
        dy, dx = endpoint["tangent"]
        support = np.abs(-dx * fields["vectors"][:, 0, y, x]
                         + dy * fields["vectors"][:, 1, y, x])
        endpoint["anchor_support"] = float(support.mean())
    pairs = []
    reasons = {}
    def reject(reason):
        reasons[reason] = reasons.get(reason, 0) + 1

    if endpoints:
        coordinates = np.asarray([e["point"] for e in endpoints])
        tree = cKDTree(coordinates)
        neighbors = []
        for index, point in enumerate(coordinates):
            local = tree.query_ball_point(point, gap_limit)
            local = sorted((i for i in local if i != index),
                           key=lambda i: (float(np.linalg.norm(point - coordinates[i])), i))
            neighbors.append(set(local[:settings.max_neighbors]))
        cosine = math.cos(math.radians(settings.max_endpoint_angle))
        for first, second in sorted(tree.query_pairs(gap_limit)):
            if second not in neighbors[first] and first not in neighbors[second]:
                reject("neighbor_budget"); continue
            delta = coordinates[second] - coordinates[first]
            gap = float(np.linalg.norm(delta))
            if gap <= math.sqrt(2) + 1e-6:
                reject("adjacent"); continue
            direction = delta / gap
            a = float(direction @ endpoints[first]["tangent"])
            b = float(-direction @ endpoints[second]["tangent"])
            if min(a, b) < cosine:
                reject("endpoint_direction"); continue
            pairs.append((gap, first, second))
    pairs.sort()
    truncated = max(0, len(pairs) - settings.max_searches)
    candidates, total_expanded = [], 0
    for gap, first, second in pairs[:settings.max_searches]:
        if min(endpoints[first]["anchor_support"], endpoints[second]["anchor_support"]) < settings.geometry_min_support:
            reject("anchor_image_conflict"); continue
        result, reason, expanded = _find_path(endpoints[first], endpoints[second], ridges, fields, settings, gap)
        total_expanded += expanded
        if result is None:
            reject(reason); continue
        path, cost, length = result
        features = _path_features(path, endpoints[first], endpoints[second], fields, settings)
        if features["mean_support"] < settings.geometry_min_support:
            reject("insufficient_observation"); continue
        candidates.append(dict(id=len(candidates), endpoints=[first, second],
            path_yx=path.tolist(), gap=gap, length=length, cost=cost,
            strength=min(endpoints[first]["strength"], endpoints[second]["strength"]),
            features=features, geometry_selected=False, reliability_accepted=False,
            geometry_rejection=None, reliability_rejections=[]))

    # Ambiguity compares ALL admissible alternatives before greedy matching.
    for candidate in candidates:
        alternatives = [c["cost"] for c in candidates if c["id"] != candidate["id"]
                        and set(c["endpoints"]) & set(candidate["endpoints"])]
        margin = max(0., (min(alternatives) - candidate["cost"]) / max(candidate["cost"], 1e-12)) if alternatives else 1.
        f = candidate["features"]
        f["alternative_margin"] = float(margin)
        score = (.40 * f["mean_support"] + .25 * f["support_coverage"]
                 + .20 * f["cross_scale_agreement"] + .15 * f["tangent_quality"])
        score *= .5 + .5 * min(1., margin / .2)
        candidate["reliability_score"] = float(np.clip(score, 0, 1))
        failures = candidate["reliability_rejections"]
        if score < settings.reliability_min_score: failures.append("low_score")
        if f["support_coverage"] < settings.reliability_min_coverage: failures.append("low_coverage")
        if f["unsupported_run"] > settings.reliability_max_unsupported_run: failures.append("unsupported_run")
        if margin < settings.reliability_min_margin: failures.append("ambiguous_match")

    used, painted, diagonal_cells = set(), np.zeros_like(ridges), {}
    for candidate in sorted(candidates, key=lambda c: (c["cost"], c["id"])):
        path = np.asarray(candidate["path_yx"], dtype=int)
        diagonal = _diagonals(path)
        if used & set(candidate["endpoints"]):
            candidate["geometry_rejection"] = "endpoint_already_used"; continue
        if painted[tuple(path.T)].any() or any(cell in diagonal_cells and diagonal_cells[cell] != slope
                                              for cell, slope in diagonal.items()):
            candidate["geometry_rejection"] = "candidate_crossing"; continue
        candidate["geometry_selected"] = True
        candidate["reliability_accepted"] = not candidate["reliability_rejections"]
        used.update(candidate["endpoints"])
        painted[tuple(path.T)] = True; diagonal_cells.update(diagonal)

    geometry_mask, _, scores = _paint(image.shape, candidates)
    reliable_mask, _, _ = _paint(image.shape, candidates, accepted_only=True)
    geometry_edge, reliable_edge = _render_paths(image.shape, candidates, settings.soft_width)
    maps = dict(observed=observed.copy(), closing=np.maximum(observed, closing_edge),
        geometry=np.maximum(observed, geometry_edge), reliable=np.maximum(observed, reliable_edge),
        closing_added=closing_edge, geometry_added=geometry_edge, reliable_added=reliable_edge,
        closing_mask=closing_added, geometry_mask=geometry_mask, reliable_mask=reliable_mask,
        candidate_score=scores, trace_ridges=ridges, gradient_support=fields["strength"])
    for key, values in maps.items():
        if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
            raise FloatingPointError(f"Invalid completion output: {key}")
    endpoint_records = [dict(e, point=list(e["point"]), tangent=e["tangent"].tolist()) for e in endpoints]
    metadata = dict(format="local_contour_completion_v1", settings=asdict(settings),
        gap_limit_pixels=gap_limit, closing_square_radius=radius,
        closing_policy="Binary square closing, anchored thinning preserves every original ridge",
        endpoints=endpoint_records, candidates=candidates, pair_rejections=reasons,
        eligible_pairs=len(pairs), truncated_pairs=truncated, expanded_states=total_expanded,
        geometry_selected=sum(c["geometry_selected"] for c in candidates),
        reliability_accepted=sum(c["reliability_accepted"] for c in candidates),
        noise_sigma=fields["noise_sigma"],
        semantics="Only inherited through fixed observed contours; no hard P/Q veto",
        score_meaning="Uncalibrated structural reliability proxy; not correctness probability",
        soft_rendering="Per-path Gaussian max, eight-sigma truncation; identical width in every arm",
        matching="Deterministic greedy, at most one connection per endpoint; not globally optimal")
    return maps, metadata
