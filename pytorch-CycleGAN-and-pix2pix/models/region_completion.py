"""Non-learning completion controls on frozen fragment/weak proposal pools.

Exact geometry deduplication, held-out regional evidence, and joint selection
are separate arms. No class, box, filename, semantic map or generated RGB is
read here. All scores are uncalibrated diagnostics, never probabilities.
"""

import copy
from collections import Counter
from dataclasses import asdict, dataclass
import math

import numpy as np

from .clarity_semantic_edges import noise_sigma
from .completion_importance import _validate_candidate
from .completion_recall import _score, _validate_source
from .fragment_completion import _conflict, _occupy, _old_locks
from .semantic_edges import _unit_map


ARMS = ("dedup", "observed_control", "region", "weak_region", "joint")
R_KEYS = ("add_importance", "importance_gain", "importance_features")


@dataclass(frozen=True)
class RegionSettings:
    band_fraction: float = 3 / 288
    minimum_samples: int = 24
    minimum_side_fold_samples: int = 4
    minimum_region_support: float = .5
    minimum_tangent_quality: float = .65
    huber_noise_multiplier: float = 2.5
    complexity_weight: float = 1.
    batch_size: int = 256

    def __post_init__(self):
        for name in ("band_fraction", "minimum_region_support", "minimum_tangent_quality"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 < v <= 1:
                raise ValueError(f"Finite fraction in (0,1] required: {name}")
        if self.band_fraction > .05:
            raise ValueError("Local regional evidence required")
        for name in ("minimum_samples", "minimum_side_fold_samples", "batch_size"):
            v = getattr(self, name)
            if type(v) is not int or v < 1:
                raise ValueError(f"Positive integer required: {name}")
        if self.minimum_samples < 12 or self.minimum_side_fold_samples < 3:
            raise ValueError("Two planar fits require enough unique samples")
        for name in ("huber_noise_multiplier", "complexity_weight"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
                raise ValueError(f"Positive finite value required: {name}")


def path_key(path):
    p = tuple(tuple(int(v) for v in row) for row in path)
    return min(p, p[::-1])


def _without_R(candidate):
    return {k: v for k, v in candidate.items() if k not in R_KEYS}


def _validate_pool(pool, original, ridges, kind):
    n = len(original["candidates"])
    detail = pool.get("fragment_completion" if kind == "fragment" else "weak_recall", {})
    expected = "two_fragment_completion_v2" if kind == "fragment" else "anchored_weak_recall_v1"
    if detail.get("format") != expected or detail.get("original_candidates") != n:
        raise ValueError(f"A compatible {kind} proposal record is required")
    if pool.get("settings") != original["settings"] or pool.get("gap_limit_pixels") != original["gap_limit_pixels"]:
        raise ValueError("Proposal coordinate scale/settings differ")
    if [_without_R(c) for c in pool["candidates"][:n]] != [_without_R(c) for c in original["candidates"]]:
        raise ValueError("Proposal original candidates differ")
    m = len(original["endpoints"])
    if pool["endpoints"][:m] != original["endpoints"]:
        raise ValueError("Proposal original anchors differ")
    if [c.get("id") for c in pool["candidates"]] != list(range(len(pool["candidates"]))):
        raise ValueError("Consecutive proposal IDs required")
    for i, e in enumerate(pool["endpoints"]):
        p = np.asarray(e.get("point"))
        t = np.asarray(e.get("tangent"), dtype=float)
        if e.get("id") != i or p.shape != (2,) or p.dtype.kind not in "iu" or (p < 0).any() or (p >= np.asarray(ridges.shape)).any():
            raise ValueError("Invalid proposal anchor coordinates/identity")
        if t.shape != (2,) or not np.isfinite(t).all() or not np.isclose(np.linalg.norm(t), 1, atol=1e-5):
            raise ValueError("A finite unit anchor tangent is required")
        for name in ("strength", "tangent_quality"):
            v = e.get(name)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1:
                raise ValueError("Invalid anchor evidence")
    for c in pool["candidates"][n:]:
        p = _validate_candidate(c, ridges.shape)
        if len(np.unique(p, axis=0)) != len(p) or ridges[tuple(p[1:-1].T)].any():
            raise ValueError("New proposals must be simple and avoid original ridge interiors")
        ids = c.get("endpoints")
        if not isinstance(ids, list) or len(ids) != 2 or len(set(ids)) != 2 or any(type(i) is not int or not 0 <= i < len(pool["endpoints"]) for i in ids):
            raise ValueError("Two distinct proposal anchors required")
        if p[[0, -1]].tolist() != [pool["endpoints"][i]["point"] for i in ids]:
            raise ValueError("Path and anchor coordinates differ")
        for name in ("cost", "gap", "length"):
            if not isinstance(c.get(name), (int, float)) or not math.isfinite(c[name]) or c[name] <= 0:
                raise ValueError("Finite positive path costs/lengths required")
        if not np.isclose(c["gap"], np.linalg.norm(p[-1] - p[0]), atol=1e-5):
            raise ValueError("Proposal gap differs from coordinates")
        if not np.isclose(c["length"], np.linalg.norm(np.diff(p, axis=0), axis=1).sum(), atol=1e-5):
            raise ValueError("Proposal arc length differs from coordinates")
        if c["gap"] > original["gap_limit_pixels"] + 1e-5:
            raise ValueError("Proposal exceeds the original local gap limit")
        for name in ("mean_support", "support_coverage", "cross_scale_agreement", "unsupported_run", "tangent_quality"):
            v = c.get("features", {}).get(name)
            if not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1:
                raise ValueError("Invalid frozen path evidence")


def _samples(path, shape, radius):
    """Unique actual pixels in the interior's normal bands, not interpolates.

    Endpoints/centerline are excluded. Pixels assigned to both sides of a tight
    turn are unknown and omitted. Duplicate sampling never increases n.
    """
    p = np.asarray(path, dtype=float)
    middle = p[1:-1]
    d = p[2:] - p[:-2]
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    normal = np.stack((-d[:, 1], d[:, 0]), axis=1)
    offsets = np.r_[np.arange(-radius, 0), np.arange(1, radius + 1)]
    coords = np.rint(middle[:, None] + offsets[None, :, None] * normal[:, None]).astype(int)
    centers = {tuple(row) for row in p.astype(int)}
    sides = {}
    for point, offset in zip(coords.reshape(-1, 2), np.tile(offsets, len(middle))):
        q = tuple(point)
        if min(q) < 0 or q[0] >= shape[0] or q[1] >= shape[1] or q in centers:
            continue
        sign = int(np.sign(offset))
        sides[q] = sign if q not in sides else (sides[q] if sides[q] == sign else 0)
    keys = sorted(q for q, sign in sides.items() if sign)
    if not keys:
        return np.empty((0, 2), int), np.empty(0, int)
    return np.asarray(keys), np.asarray([sides[q] for q in keys])


def _fit(X, values, weights, delta):
    # The tiny numerical ridge is fixed, and is not a fitted statistical prior.
    active = weights.copy()
    beta = np.zeros((len(X), X.shape[-1]), float)
    for _ in range(2):
        gram = X.transpose(0, 2, 1) @ (active[..., None] * X)
        rhs = X.transpose(0, 2, 1) @ (active * values)[..., None]
        beta = np.linalg.solve(gram + 1e-8 * np.eye(X.shape[-1]), rhs)[..., 0]
        residual = np.abs(np.einsum("bni,bi->bn", X, beta) - values)
        active = weights * np.minimum(1., delta / np.maximum(residual, 1e-12))
    return beta


def region_evidence(image, paths, *, settings=None):
    """Held-out H0 plane vs H1 two planes, using identical unique pixels.

    Two checkerboard folds swap train/validation roles. H1 has three additional
    parameters and pays 3*noise^2*log(n)/n. The result is a bounded improvement
    proxy, not a likelihood ratio or calibrated correctness probability.
    """
    settings = RegionSettings() if settings is None else settings
    if not isinstance(settings, RegionSettings):
        raise TypeError("Use validated RegionSettings")
    image = _unit_map(image, "TIR")
    radius = max(2, int(math.ceil(min(image.shape) * settings.band_fraction)))
    sigma = max(float(noise_sigma(image)), 1 / 255)
    delta = settings.huber_noise_multiplier * sigma
    records = []
    for start in range(0, len(paths), settings.batch_size):
        batch = paths[start:start + settings.batch_size]
        samples = [_samples(p, image.shape, radius) for p in batch]
        width = max(1, max(len(p) for p, _ in samples))
        X = np.zeros((len(batch), width, 3), float)
        side = np.zeros((len(batch), width), int)
        values = np.zeros((len(batch), width), float)
        valid = np.zeros((len(batch), width), bool)
        parity = np.zeros((len(batch), width), int)
        n = np.asarray([len(p) for p, _ in samples])
        for i, (coords, signs) in enumerate(samples):
            if not len(coords):
                continue
            normalized = (coords - np.mean(batch[i], axis=0)) / max(radius, np.ptp(np.asarray(batch[i]), axis=0).max(), 1)
            X[i, :len(coords)] = np.c_[np.ones(len(coords)), normalized]
            values[i, :len(coords)] = image[tuple(coords.T)]
            side[i, :len(coords)] = signs
            parity[i, :len(coords)] = coords.sum(1) % 2
            valid[i, :len(coords)] = True
        H1 = np.concatenate((X * (side < 0)[..., None], X * (side > 0)[..., None]), axis=-1)
        loss0 = np.zeros(len(batch)); loss1 = np.zeros(len(batch))
        usable = n >= settings.minimum_samples
        for fold in (0, 1):
            train = valid & (parity == fold)
            held = valid & ~train
            for sign in (-1, 1):
                usable &= ((train & (side == sign)).sum(1) >= settings.minimum_side_fold_samples)
            gram = H1.transpose(0, 2, 1) @ (train[..., None] * H1)
            usable &= np.linalg.eigvalsh(gram)[:, 0] > 1e-8
            b0 = _fit(X, values, train.astype(float), delta)
            b1 = _fit(H1, values, train.astype(float), delta)
            for design, beta, total in ((X, b0, loss0), (H1, b1, loss1)):
                residual = np.abs(np.einsum("bni,bi->bn", design, beta) - values)
                huber = np.where(residual <= delta, residual**2, 2 * delta * residual - delta**2)
                total += (huber * held).sum(1)
        loss0 /= np.maximum(n, 1); loss1 /= np.maximum(n, 1)
        penalty = settings.complexity_weight * 3 * sigma**2 * np.log(np.maximum(n, 1)) / np.maximum(n, 1)
        improvement = loss0 - loss1 - penalty
        support = np.clip(improvement / (loss0 + sigma**2), 0, 1)
        support[~usable] = 0
        for i in range(len(batch)):
            records.append(dict(available=bool(usable[i]), unique_pixels=int(n[i]), band_radius=radius,
                support=float(support[i]), null_loss=float(loss0[i]), split_loss=float(loss1[i]),
                complexity_penalty=float(penalty[i]), signed_improvement=float(improvement[i]), noise_floor=sigma))
    return records


def _prepare(original, pools, ridges):
    endpoints = copy.deepcopy(original["endpoints"])
    original_paths = {path_key(c["path_yx"]) for c in original["candidates"]}
    groups = {}
    for kind, pool in pools.items():
        _validate_pool(pool, original, ridges, kind)
        mapping = {i: i for i in range(len(original["endpoints"]))}
        for e in pool["endpoints"][len(original["endpoints"]):]:
            mapping[e["id"]] = len(endpoints)
            endpoints.append(dict(copy.deepcopy(e), id=len(endpoints), proposal_pool=kind, proposal_endpoint_id=e["id"]))
        for c in pool["candidates"][len(original["candidates"]):]:
            key = path_key(c["path_yx"])
            member = copy.deepcopy(c)
            member.update(pool=kind, proposal_id=c["id"], endpoints=[mapping[i] for i in c["endpoints"]])
            groups.setdefault(key, []).append(member)
    for members in groups.values():
        members.sort(key=lambda c: (c["cost"], c["pool"], c["proposal_id"]))
    return endpoints, groups, original_paths


def _anchor_summary(members, endpoints):
    # A measured weak terminal is NOT a second inducer for invisible-gap inference.
    measured = [m for m in members if m.get("source") == "two_measured_fragments" or m.get("closing_original_endpoint") is True]
    if not measured:
        return dict(two_measured_anchors=False, support=0., alignment=0., quality=0.)
    c = measured[0]
    a, b = (endpoints[i] for i in c["endpoints"])
    direction = np.asarray(b["point"], float) - a["point"]
    direction /= np.linalg.norm(direction)
    # Closed weak paths use original anchors, which have explicit anchor support.
    support = min(float(a.get("anchor_support", 0)), float(b.get("anchor_support", 0)))
    alignment = min(float(direction @ a["tangent"]), float(-direction @ b["tangent"]))
    return dict(two_measured_anchors=True, support=float(np.clip(support, 0, 1)),
        alignment=float(np.clip(alignment, 0, 1)), quality=min(a["tangent_quality"], b["tangent_quality"]))


def _regional_unary(c, settings, original, allow_regions=True):
    f, region, anchors = c["features"], c["region_evidence"], c["anchor_evidence"]
    observed_score = .4*f["mean_support"] + .25*f["support_coverage"] + .2*f["cross_scale_agreement"] + .15*f["tangent_quality"]
    observed_ok = (observed_score >= original.reliability_min_score
        and f["support_coverage"] >= original.reliability_min_coverage
        and f["unsupported_run"] <= original.reliability_max_unsupported_run)
    gap_score = .4*region["support"] + .2*anchors["support"] + .2*anchors["alignment"] + .2*anchors["quality"]
    gap_ok = (allow_regions and anchors["two_measured_anchors"] and region["available"]
        and region["support"] >= settings.minimum_region_support
        and anchors["support"] >= original.geometry_min_support
        and anchors["quality"] >= settings.minimum_tangent_quality
        and anchors["alignment"] >= math.cos(math.radians(original.max_endpoint_angle))
        and gap_score >= original.reliability_min_score)
    # Explicit alternative decision rules; do not interpret max(scores) as a probability.
    route = "observed" if observed_ok else ("regional_gap" if gap_ok else "abstain")
    unary = observed_score if observed_ok else (gap_score if gap_ok else 0.)
    c.update(evidence_route=route, unary_score=float(unary),
        decision_utility=float(max(0., unary - original.reliability_min_score)),
        evidence_scores=dict(observed=float(observed_score), regional_gap=float(gap_score)))
    if route == "abstain":
        c["reliability_rejections"] = ["insufficient_observed_or_regional_evidence"]
        c["reliability_score"] = 0.


def completion_stages(image, observed, ridges, metadata, fragments, weak, *, settings=None):
    """Return three staged changes plus a weak-only common-rule control.

    The caller owns proposal search/cache provenance and separate R computation.
    Original observations, candidate C, and old geometry locks remain frozen.
    New exact-path groups retain every member's metadata in the returned audit.
    """
    settings = RegionSettings() if settings is None else settings
    if not isinstance(settings, RegionSettings):
        raise TypeError("Use validated RegionSettings")
    image = _unit_map(image, "TIR"); observed = _unit_map(observed, "observed", image.shape)
    ridges = np.asarray(ridges)
    if ridges.dtype != np.bool_ or ridges.shape != image.shape or min(image.shape) < 8:
        raise ValueError("Aligned boolean original ridges required")
    if "region_completion" in metadata:
        raise ValueError("Do not append selection stages twice")
    original = _validate_source(metadata, ridges)
    endpoints, groups, original_paths = _prepare(metadata, dict(fragment=fragments, weak=weak), ridges)
    keys = sorted(groups)
    regional = dict(zip(keys, region_evidence(image, keys, settings=settings)))
    outputs = {}; n = len(metadata["candidates"])
    raw_counts = Counter(m["pool"] for members in groups.values() for m in members)
    audit = []
    for key in keys:
        members = []
        for m in groups[key]:
            record = {k: v for k, v in m.items() if k != "path_yx"}
            record["path_reversed_from_canonical"] = tuple(map(tuple, m["path_yx"])) != key
            members.append(record)
        audit.append(dict(canonical_path_yx=[list(p) for p in key], members=members,
            duplicates_original_candidate=key in original_paths, region_evidence=regional[key]))
    for arm in ARMS:
        allowed = {"fragment"} if arm in ("dedup", "observed_control", "region") else ({"weak"} if arm == "weak_region" else {"fragment", "weak"})
        new = []
        for key in keys:
            members = [m for m in groups[key] if m["pool"] in allowed]
            if not members or key in original_paths:
                continue
            c = copy.deepcopy(members[0]); c["id"] = n + len(new)
            c.update(features=dict(c["features"]), geometry_selected=False, reliability_accepted=False,
                geometry_rejection=None, reliability_rejections=[],
                source_members=[dict(pool=m["pool"], id=m["proposal_id"]) for m in members],
                region_evidence=dict(regional[key]), anchor_evidence=_anchor_summary(members, endpoints))
            new.append(c)
        by_site = {}
        competitors = metadata["candidates"] + new if arm == "dedup" else new
        if arm != "dedup":
            for c in new:
                _regional_unary(c, settings, original, allow_regions=arm != "observed_control")
            competitors = [c for c in new if not c["reliability_rejections"]]
        for c in competitors:
            for p in (tuple(c["path_yx"][0]), tuple(c["path_yx"][-1])):
                by_site.setdefault(p, []).append(c)
        for c in new:
            alternatives = {a["id"]: a for p in (tuple(c["path_yx"][0]), tuple(c["path_yx"][-1]))
                for a in by_site.get(p, []) if a["id"] != c["id"] and path_key(a["path_yx"]) != path_key(c["path_yx"])}
            if arm == "dedup":
                _score(c, [a["cost"] for a in alternatives.values()], original)
            elif not c["reliability_rejections"]:
                # Utility of not connecting is zero. Compare only distinct,
                # evidence-eligible explanations, never duplicate observations.
                utility = c["decision_utility"]
                alternative = max([0.] + [a["decision_utility"] for a in alternatives.values()])
                margin = float(np.clip((utility-alternative)/max(utility, 1e-12), 0, 1))
                c["features"]["alternative_margin"] = margin
                score = c["unary_score"] * (.5 + .5 * min(1., margin/.2))
                c["reliability_score"] = float(score)
                if margin < original.reliability_min_margin: c["reliability_rejections"].append("ambiguous_connection")
                if score < original.reliability_min_score: c["reliability_rejections"].append("low_score")
        occupied, painted, diagonals = _old_locks(metadata, image.shape)
        # Preserve source-ID tie order in the isolated deduplication control.
        tie = lambda c: (c["pool"], c["proposal_id"])
        ordering = sorted(new, key=lambda c: (c["cost"], *tie(c))) if arm == "dedup" else sorted(new, key=lambda c: (-c["reliability_score"], c["cost"]/c["length"], *tie(c)))
        for c in ordering:
            if arm != "dedup" and c["reliability_rejections"]:
                c["geometry_rejection"] = "C_rejected_before_matching"
                continue
            p = np.asarray(c["path_yx"], int)
            conflict = _conflict(p, occupied, painted, diagonals)
            if conflict:
                c["geometry_rejection"] = conflict
                continue
            c["geometry_selected"] = True
            c["reliability_accepted"] = not c["reliability_rejections"]
            _occupy(p, occupied, painted, diagonals)
        updated = copy.deepcopy(metadata)
        updated["endpoints"] = endpoints
        updated["candidates"].extend(new)
        updated["geometry_selected"] = sum(c["geometry_selected"] for c in updated["candidates"])
        updated["reliability_accepted"] = sum(c["reliability_accepted"] for c in updated["candidates"])
        raw = sum(raw_counts[k] for k in allowed)
        unique = sum(any(m["pool"] in allowed for m in groups[key]) for key in keys)
        updated["region_completion"] = dict(format="regional_selection_v1", arm=arm, settings=asdict(settings),
            original_candidates=n, raw_new_records=raw, unique_new_geometries=unique,
            exact_duplicate_records=raw-unique, duplicates_original_excluded=unique-len(new),
            new_candidates=len(new), new_accepted=sum(c["reliability_accepted"] for c in new),
            accepted_routes=dict(Counter(c.get("evidence_route", "legacy_C") for c in new if c["reliability_accepted"])),
            parameters_learned=0, original_C_and_geometry_preserved=True,
            score_meaning="Uncalibrated evidence/decision utility; no correctness probability",
            matching="Legacy order in dedup arm; evidence-first in regional arms; original locks retained",
            null_hypothesis="One planar region, zero connection utility; no evidence means abstain")
        outputs[arm] = updated
    return outputs, dict(groups=audit, proposal_endpoints=endpoints, raw_pool_counts=dict(raw_counts),
        settings=asdict(settings), parameters_learned=0)
