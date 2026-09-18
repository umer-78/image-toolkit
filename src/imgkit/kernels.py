"""The kernels, and what each one is for."""

from __future__ import annotations

import numpy as np


def box(size: int = 3) -> np.ndarray:
    """A plain average. Cheap, separable, and it blurs edges as hard as anything else."""
    _odd(size)
    return np.ones((size, size), dtype=np.float64) / (size * size)


def gaussian(sigma: float = 1.0, size: int | None = None) -> np.ndarray:
    """A 2D Gaussian, normalised to sum to one.

    The default radius is three sigma: beyond that the weights are under 1% of
    the peak and truncating there costs nothing visible. Too small a radius is a
    common and invisible bug — the kernel is renormalised so the image does not
    darken, which hides the fact that it is no longer a Gaussian.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if size is None:
        size = int(2 * np.ceil(3 * sigma) + 1)
    _odd(size)

    half = size // 2
    axis = np.arange(-half, half + 1, dtype=np.float64)
    line = np.exp(-(axis ** 2) / (2 * sigma ** 2))
    kernel = np.outer(line, line)
    return kernel / kernel.sum()


def gaussian_1d(sigma: float = 1.0, size: int | None = None) -> np.ndarray:
    """One half of the separable Gaussian."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if size is None:
        size = int(2 * np.ceil(3 * sigma) + 1)
    _odd(size)
    half = size // 2
    axis = np.arange(-half, half + 1, dtype=np.float64)
    line = np.exp(-(axis ** 2) / (2 * sigma ** 2))
    return line / line.sum()


def sobel_x() -> np.ndarray:
    """Horizontal gradient. Positive where the image gets brighter to the right."""
    return np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)


def sobel_y() -> np.ndarray:
    """Vertical gradient. Positive where the image gets brighter downward."""
    return np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)


def prewitt_x() -> np.ndarray:
    return np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64)


def prewitt_y() -> np.ndarray:
    return np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float64)


def laplacian(diagonal: bool = False) -> np.ndarray:
    """Second derivative: responds to where the gradient changes, not where it is high."""
    if diagonal:
        return np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float64)
    return np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)


def sharpen(strength: float = 1.0) -> np.ndarray:
    """Identity plus a scaled Laplacian: adds back what a blur would remove."""
    return np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float64) - strength * laplacian()


def emboss() -> np.ndarray:
    """A directional derivative with the mid-grey offset applied afterwards."""
    return np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float64)


def _odd(size: int) -> None:
    if size < 1 or size % 2 == 0:
        raise ValueError(f"the kernel size must be a positive odd number, got {size}")


NAMED = {
    "box": box,
    "gaussian": gaussian,
    "sobel-x": sobel_x,
    "sobel-y": sobel_y,
    "prewitt-x": prewitt_x,
    "prewitt-y": prewitt_y,
    "laplacian": laplacian,
    "sharpen": sharpen,
    "emboss": emboss,
}
