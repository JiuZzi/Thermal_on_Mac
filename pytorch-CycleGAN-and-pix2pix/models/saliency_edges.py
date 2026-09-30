"""Position, content-adaptive, and detector-guided soft contour candidates.

The map returned by ``center_importance`` is only the former spatial prior.
The adaptive structure/texture map is not semantic recognition or confidence.
"""

import numpy as np


def center_importance(
    height: int,
    width: int,
    inner_weight: float,
    outer_weight: float,
    transition_fraction: float,
) -> np.ndarray:
    """Return a smooth middle-third importance prior at the input resolution."""
    transition = max(height * transition_fraction, 1.0)
    y = np.arange(height, dtype=np.float32) + 0.5

    def smoothstep(value: np.ndarray) -> np.ndarray:
        value = np.clip(value, 0.0, 1.0)
        return value * value * (3.0 - 2.0 * value)

    entering = smoothstep((y - (height / 3.0 - transition / 2.0)) / transition)
    leaving = smoothstep((y - (2.0 * height / 3.0 - transition / 2.0)) / transition)
    middle = entering * (1.0 - leaving)
    line = outer_weight + (inner_weight - outer_weight) * middle
    return np.broadcast_to(line[:, None], (height, width)).copy().astype(np.float32)


def soft_multi_at_sigma(
    image: np.ndarray,
    sigma: float,
    highs: tuple[float, ...],
    low_ratio: float,
    width: float,
) -> np.ndarray:
    """Match C2b's Canny thresholds and Gaussian distance softening."""
    from scipy.ndimage import distance_transform_edt
    from skimage.feature import canny

    maps = []
    for high in highs:
        hard = canny(
            image,
            sigma=sigma,
            low_threshold=high * low_ratio,
            high_threshold=high,
        )
        if hard.any():
            distance = distance_transform_edt(~hard)
            maps.append(np.exp(-0.5 * (distance / width) ** 2).astype(np.float32))
        else:
            maps.append(np.zeros(image.shape, dtype=np.float32))
    return np.mean(maps, axis=0, dtype=np.float32)


def make_saliency_edge(
    image: np.ndarray,
    highs: tuple[float, ...],
    low_ratio: float,
    width: float,
    sigmas: tuple[float, float, float],
    importance: np.ndarray,
    background_gain: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fuse fine/coarse candidates using an importance map, without thresholding it."""
    if image.shape != importance.shape:
        raise ValueError("Importance map must match the processed TIR image")
    if not np.isfinite(importance).all() or np.any((importance < 0) | (importance > 1)):
        raise ValueError("Importance map must contain finite values in [0, 1]")
    fine_map, middle_map, coarse_map = (
        soft_multi_at_sigma(image, sigma, highs, low_ratio, width)
        for sigma in sigmas
    )
    fine = (fine_map + middle_map) / 2.0
    coarse = (middle_map + coarse_map) / 2.0
    result = importance * fine + (1.0 - importance) * background_gain * coarse
    return result.astype(np.float32), fine.astype(np.float32), coarse.astype(np.float32)


def structure_texture_importance(
    image: np.ndarray,
    fine_edge: np.ndarray,
    coarse_edge: np.ndarray,
    *,
    legacy_binary_density: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Continuous fine-detail weight from local thermal structure and clutter.

    This is an importance prior, not edge confidence or an object mask. The
    nonzero floor prevents missing small objects merely because their local
    statistics resemble texture.
    """
    from scipy.ndimage import gaussian_filter, sobel

    if image.shape != fine_edge.shape or image.shape != coarse_edge.shape:
        raise ValueError("Image and edge candidates must have identical shapes")
    smooth = gaussian_filter(image, 1.0)
    gx = sobel(smooth, axis=1) / 8.0
    gy = sobel(smooth, axis=0) / 8.0
    jxx = gaussian_filter(gx * gx, 2.0)
    jxy = gaussian_filter(gx * gy, 2.0)
    jyy = gaussian_filter(gy * gy, 2.0)
    anisotropy = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy**2)
    coherence = anisotropy / (jxx + jyy + 1e-5)
    contrast = np.clip(np.sqrt(jxx + jyy) / 0.06, 0.0, 1.0)
    structure = coherence * contrast

    # The current C2c counts soft-edge strength continuously. The previous
    # binary threshold is retained only to reproduce the audit comparison.
    fine_values = (fine_edge > 0.45).astype(np.float32) if legacy_binary_density else fine_edge
    coarse_values = (coarse_edge > 0.45).astype(np.float32) if legacy_binary_density else coarse_edge
    fine_density = gaussian_filter(fine_values, 4.0)
    coarse_density = gaussian_filter(coarse_values, 4.0)
    excess_texture = np.clip((fine_density - coarse_density - 0.04) / 0.22, 0.0, 1.0)
    dense_texture = np.clip((fine_density - 0.18) / 0.28, 0.0, 1.0)
    clutter = np.clip(0.65 * dense_texture + 0.35 * excess_texture, 0.0, 1.0)
    clutter *= 1.0 - 0.65 * structure
    raw = 0.32 + 0.45 * structure + 0.12 * contrast - 0.35 * clutter
    importance = np.clip(gaussian_filter(raw, 2.0), 0.18, 0.85)
    return importance.astype(np.float32), gaussian_filter(clutter, 1.5).astype(np.float32)


def make_adaptive_saliency_edge(
    image: np.ndarray,
    highs: tuple[float, ...],
    low_ratio: float,
    width: float,
    sigmas: tuple[float, float, float],
    background_gain: float,
    variant: str = "structure_texture",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return soft edge, importance, fine, coarse for two C2c candidates."""
    if variant not in (
        "structure_texture", "edge_preserving", "structure_texture_legacy",
        "structure_texture_density_only",
    ):
        raise ValueError(f"Unknown adaptive edge variant: {variant}")
    if not (0 < background_gain <= 1):
        raise ValueError("Background gain must be in (0, 1]")
    if variant == "edge_preserving":
        from skimage.restoration import denoise_bilateral

        filtered = denoise_bilateral(
            image, sigma_color=0.045, sigma_spatial=2.0, channel_axis=None,
        ).astype(np.float32)
        candidate_image = 0.35 * image + 0.65 * filtered
    else:
        candidate_image = image
    fine_map, middle_map, coarse_map = (
        soft_multi_at_sigma(candidate_image, sigma, highs, low_ratio, width)
        for sigma in sigmas
    )
    fine = (fine_map + middle_map) / 2.0
    coarse = (middle_map + coarse_map) / 2.0
    importance, clutter = structure_texture_importance(
        candidate_image, fine, coarse,
        legacy_binary_density=(variant == "structure_texture_legacy"),
    )
    # Preserve a coarse structural route everywhere. Importance only decides
    # how much fine detail to add; it never sets all edges to zero.
    result = importance * fine + (1.0 - importance) * background_gain * coarse
    result *= 1.0 - 0.4 * clutter
    if variant in ("structure_texture", "edge_preserving"):
        # A fine-only edge previously fell to 0.18 * (1 - 0.4) = 0.108 of
        # its candidate strength. Keep at least 25% without inventing edges.
        result = np.maximum(result, 0.25 * fine)
    return (np.clip(result, 0.0, 1.0).astype(np.float32), importance,
            fine.astype(np.float32), coarse.astype(np.float32))


def long_line_importance(image: np.ndarray) -> tuple[np.ndarray, int]:
    """Long contours are a structure proxy; this does not identify buildings."""
    from PIL import Image, ImageDraw
    from scipy.ndimage import distance_transform_edt
    from skimage.feature import canny
    from skimage.transform import probabilistic_hough_line

    height, width = image.shape
    hard = canny(image, sigma=1.6, low_threshold=0.08, high_threshold=0.16)
    length = max(18, round(min(height, width) * 0.08))
    lines = probabilistic_hough_line(
        hard, threshold=10, line_length=length, line_gap=4,
        rng=np.random.default_rng(0),
    )
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for start, end in lines:
        draw.line((start, end), fill=255, width=2)
    inside = np.asarray(mask, dtype=np.uint8) > 0
    if not inside.any():
        return np.zeros(image.shape, dtype=np.float32), 0
    distance = distance_transform_edt(~inside)
    return np.exp(-0.5 * (distance / 3.0) ** 2).astype(np.float32), len(lines)


def object_line_contour(
    image: np.ndarray,
    object_importance: np.ndarray,
    line_importance: np.ndarray,
    background_gain: float,
    highs: tuple[float, ...] = (0.12, 0.16, 0.20),
    low_ratio: float = 0.5,
    width: float = 1.0,
    sigmas: tuple[float, float, float] = (0.7, 1.0, 1.6),
) -> np.ndarray:
    """Keep fine candidates near detected objects and coarse long structures."""
    from scipy.ndimage import maximum_filter

    if image.shape != object_importance.shape or image.shape != line_importance.shape:
        raise ValueError("Object and line maps must match the processed TIR image")
    if not (0 < background_gain <= 1):
        raise ValueError("Background gain must be in (0, 1]")
    if any(not np.isfinite(values).all() or np.any((values < 0) | (values > 1))
           for values in (object_importance, line_importance)):
        raise ValueError("Object and line maps must contain finite values in [0, 1]")
    fine, middle, coarse = (
        soft_multi_at_sigma(image, sigma, highs, low_ratio, width)
        for sigma in sigmas
    )
    target_candidate = (fine + middle) / 2.0
    # Fine edges without nearby mid-scale support remain possible but dimmer.
    # This is part of edge extraction, not the reserved confidence channel.
    support = maximum_filter(middle, size=3)
    target_candidate *= 0.65 + 0.35 * support
    background_candidate = (middle + coarse) / 2.0
    background_candidate *= background_gain + (1.0 - background_gain) * line_importance
    result = object_importance * target_candidate + (1.0 - object_importance) * background_candidate
    return np.clip(result, 0.0, 1.0).astype(np.float32)
