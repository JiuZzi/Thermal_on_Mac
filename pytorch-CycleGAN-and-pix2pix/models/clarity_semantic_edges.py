"""Clarity experiments on candidates, with the existing semantic/R policy fixed.

This is not a FWGAN reproduction. NLM supplies a small, weight-free restoration
control; the Weber-range probe uses an explicitly bounded, positive response
instead of guessing the paper's unspecified logarithm/output normalization.
"""

import numpy as np

from .semantic_edges import FMB_CLASS_NAMES, SETTINGS, _unit_map, fuse_candidates
from .selective_semantic_edges import SELECTIVE_SETTINGS, selective_components


CLARITY_SETTINGS = {
    "algorithm": "noise_residual_hard_vote_ridges_v1",
    "noise_estimator": "MAD of disjoint normalized 2x2 Haar HH coefficients",
    "nlm_patch_size": 3, "nlm_patch_distance": 3, "nlm_h_sigma": .8,
    "denoised_fraction": .5, "unsharp_sigma": 1., "unsharp_amount": .5,
    "main_preprocessing": "residual NLM only; unsharp is an independent probe",
    "ridge_distance_pixels": 1., "soft_width": .5,
    "candidate_settings": SETTINGS,
    "reference_selection": SELECTIVE_SETTINGS,
    "weber_probe": "(local_max-local_min)/(local_max+1/255); 3x3; not FWGAN eq.8",
    "reference_policy": "same raw-TIR joint importance, clutter and gain; .25 raw-fine RIDGE floor with the new width",
    "floor_change": "keep peak strength at retained raw fine ridges; no promise of the old dense pixelwise floor",
    "completion": False, "confidence": False, "semantic_correction": False,
}


def noise_sigma(image):
    """AWGN proxy only; actual texture/sensor artifacts can bias the estimate."""
    image = _unit_map(image, "TIR")
    h, w = image.shape
    if h < 2 or w < 2:
        return 0.
    h, w = h // 2 * 2, w // 2 * 2
    a = image[:h, :w]
    hh = (a[::2, ::2] - a[::2, 1::2] - a[1::2, ::2] + a[1::2, 1::2]) / 2
    return float(np.median(np.abs(hh - np.median(hh))) / .6744897501960817)


def preprocess_candidates(image):
    """Denoise locally, retain half the original residual, then mildly sharpen."""
    from scipy.ndimage import gaussian_filter
    from skimage.restoration import denoise_nl_means

    image = _unit_map(image, "TIR")
    sigma = noise_sigma(image)
    if sigma <= np.finfo(np.float32).eps:
        denoised = image.copy()
    else:
        denoised = denoise_nl_means(image, patch_size=CLARITY_SETTINGS["nlm_patch_size"],
            patch_distance=CLARITY_SETTINGS["nlm_patch_distance"],
            h=CLARITY_SETTINGS["nlm_h_sigma"] * sigma, sigma=sigma, fast_mode=True,
            preserve_range=True, channel_axis=None).astype(np.float32)
    mixed = image + CLARITY_SETTINGS["denoised_fraction"] * (denoised - image)
    sharpened = mixed + CLARITY_SETTINGS["unsharp_amount"] * (
        mixed - gaussian_filter(mixed, CLARITY_SETTINGS["unsharp_sigma"]))
    return dict(denoised=denoised, restored=mixed.astype(np.float32),
                enhanced=np.clip(sharpened, 0, 1).astype(np.float32), noise_sigma=sigma)


def normal_ridges(image, response):
    """Thin response peaks in the thermal-gradient normal, without filling gaps."""
    from scipy.ndimage import gaussian_filter, map_coordinates, sobel
    from skimage.morphology import thin

    image = _unit_map(image, "TIR")
    response = _unit_map(response, "response", image.shape)
    smooth = gaussian_filter(image, 1.)
    gx, gy = sobel(smooth, axis=1), sobel(smooth, axis=0)
    magnitude = np.hypot(gx, gy)
    valid = magnitude > np.finfo(np.float32).eps
    nx = np.divide(gx, magnitude, out=np.zeros_like(gx), where=valid)
    ny = np.divide(gy, magnitude, out=np.zeros_like(gy), where=valid)
    yy, xx = np.indices(image.shape, dtype=np.float32)
    step = CLARITY_SETTINGS["ridge_distance_pixels"]
    ahead = map_coordinates(response, [yy + step * ny, xx + step * nx], order=1, mode="nearest")
    behind = map_coordinates(response, [yy - step * ny, xx - step * nx], order=1, mode="nearest")
    peaks = thin(valid & (response > 0) & (response >= ahead) & (response >= behind))
    # Match Canny's exclusion of the outermost image border.
    peaks[[0, -1], :] = False; peaks[:, [0, -1]] = False
    return peaks


def soften_ridges(ridges, amplitudes, width=None):
    from scipy.ndimage import distance_transform_edt

    amplitudes = _unit_map(amplitudes, "ridge amplitudes")
    if ridges.shape != amplitudes.shape or ridges.dtype != np.bool_:
        raise ValueError("Ridges must be an aligned boolean map")
    width = CLARITY_SETTINGS["soft_width"] if width is None else width
    if not np.isfinite(width) or width <= 0:
        raise ValueError("Positive finite soft width required")
    if not ridges.any():
        return np.zeros_like(amplitudes)
    distance, coordinates = distance_transform_edt(~ridges, return_indices=True)
    peaks = amplitudes[coordinates[0], coordinates[1]]
    return (peaks * np.exp(-.5 * (distance / width) ** 2)).astype(np.float32)


def canny_vote_candidates(image):
    """Consolidate HARD votes before the single distance-softening operation."""
    from skimage.feature import canny

    image = _unit_map(image, "TIR")
    votes = []
    for sigma in SETTINGS["sigmas"]:
        votes.append(np.mean([canny(image, sigma=sigma, low_threshold=high * SETTINGS["low_ratio"],
                                    high_threshold=high).astype(np.float32)
                              for high in SETTINGS["high_thresholds"]], axis=0, dtype=np.float32))
    output = {}
    for key, response in (("fine", (votes[0] + votes[1]) / 2), ("coarse", (votes[1] + votes[2]) / 2)):
        ridges = normal_ridges(image, response)
        output[key] = soften_ridges(ridges, response)
        output[key + "_ridges"] = ridges
    return output


def weber_range_candidates(image):
    """Diagnostic alternative: non-directional local range + ridge/hysteresis.

    C=(max-min)/(min+1/255), bounded by C/(1+C). Unlike the published logarithm
    this has a defined, zero response for a flat patch. No percentile contrast
    stretching, per-image target quota, box or semantic mask selects edges.
    """
    from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter, label

    image = _unit_map(image, "TIR")
    votes = []
    for sigma in SETTINGS["sigmas"]:
        smooth = gaussian_filter(image, sigma)
        high = maximum_filter(smooth, size=3, mode="reflect")
        low = minimum_filter(smooth, size=3, mode="reflect")
        response = ((high - low) / (high + 1 / 255)).astype(np.float32)
        ridges = normal_ridges(smooth, response)
        candidates = []
        for threshold in SETTINGS["high_thresholds"]:
            components, count = label(ridges & (response >= threshold * SETTINGS["low_ratio"]),
                                      structure=np.ones((3, 3), bool))
            connected = np.zeros(count + 1, bool)
            connected[components[ridges & (response >= threshold)]] = True
            connected[0] = False
            candidates.append(connected[components].astype(np.float32))
        votes.append(np.mean(candidates, axis=0, dtype=np.float32))
    output = {}
    for key, response in (("fine", (votes[0] + votes[1]) / 2), ("coarse", (votes[1] + votes[2]) / 2)):
        ridges = normal_ridges(image, response)
        output[key] = soften_ridges(ridges, response)
        output[key + "_ridges"] = ridges
    return output


def clarity_components(image, probabilities, class_names=FMB_CLASS_NAMES, *, include_probes=True):
    """Change candidate extraction while holding ALL reference weighting fixed."""
    image = _unit_map(image, "TIR")
    reference = selective_components(image, probabilities, class_names)
    processed = preprocess_candidates(image)
    maps = dict(joint512=reference["joint512"], selective_v1=reference["selective_v1"],
                joint_importance=reference["joint_importance"], importance_gain=reference["importance_gain"],
                raw_fine=reference["fine"], raw_coarse=reference["coarse"],
                denoised=processed["denoised"], restored=processed["restored"], enhanced=processed["enhanced"])
    # Isolate representation width from better extraction: this control only
    # re-encodes the previous output's ridges, with no restoration/new votes.
    old_ridges = normal_ridges(image, reference["selective_v1"])
    maps["narrow_only"] = soften_ridges(old_ridges, reference["selective_v1"])
    maps["narrow_only_ridges"] = old_ridges.astype(np.float32)
    raw_ridges = normal_ridges(image, reference["fine"])
    maps["preserved_ridge_floor"] = SETTINGS["fine_floor"] * soften_ridges(raw_ridges, reference["fine"])
    maps["raw_fine_ridges"] = raw_ridges.astype(np.float32)
    routes = {"clarity_v1": canny_vote_candidates(processed["restored"])}
    if include_probes:
        routes.update(vote_only=canny_vote_candidates(image),
                      sharpened_canny=canny_vote_candidates(processed["enhanced"]),
                      weber_raw=weber_range_candidates(image),
                      weber_enhanced=weber_range_candidates(processed["enhanced"]))
    for key, candidate in routes.items():
        fused = fuse_candidates(candidate["fine"], candidate["coarse"],
                                 reference["clutter"], reference["joint_importance"])
        maps[key] = np.maximum(maps["preserved_ridge_floor"], reference["importance_gain"] * fused).astype(np.float32)
        for scale in ("fine", "coarse"):
            maps[key + "_" + scale] = candidate[scale]
            maps[key + "_" + scale + "_ridges"] = candidate[scale + "_ridges"].astype(np.float32)
    return maps, {"noise_sigma": processed["noise_sigma"]}


def make_clarity_semantic_condition(image, probabilities, class_names=FMB_CLASS_NAMES):
    return clarity_components(image, probabilities, class_names, include_probes=False)[0]["clarity_v1"]
