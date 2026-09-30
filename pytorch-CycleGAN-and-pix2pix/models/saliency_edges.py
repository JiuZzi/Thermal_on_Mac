"""Deterministic, position-guided soft contour candidates for C2c previews.

The map returned by ``center_importance`` is only a spatial prior. It does
not recognize people, vehicles, buildings, or foliage. A semantic predictor
can later supply its own [0, 1] map through ``saliency_override``.
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
