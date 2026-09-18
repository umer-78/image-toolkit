"""Whole operations built on top of the kernels."""

from __future__ import annotations

import numpy as np

from .convolve import convolve_separable, correlate, pad
from .kernels import gaussian_1d, laplacian, sobel_x, sobel_y

# Rec. 709 luminance weights. A plain (r+g+b)/3 is the usual shortcut and it is
# wrong in a way you can see: green contributes most of perceived brightness, so
# an even average turns a green field and a blue sky into the same grey.
LUMA = np.array([0.2126, 0.7152, 0.0722])


def to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.float64)
    if image.shape[2] < 3:
        return image[:, :, 0].astype(np.float64)
    return image[:, :, :3].astype(np.float64) @ LUMA


def blur(image: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """Gaussian blur, run as two 1D passes."""
    line = gaussian_1d(sigma)
    return convolve_separable(image, line, line)


def median_filter(image: np.ndarray, size: int = 3) -> np.ndarray:
    """Replace each pixel with the median of its neighbourhood.

    The right tool for salt-and-pepper noise, and the reason is worth stating: a
    single white pixel drags a mean with it, so a blur spreads the speck into a
    grey smudge. A median ignores it entirely as long as it is outnumbered.
    """
    if size < 1 or size % 2 == 0:
        raise ValueError("the window must be a positive odd number")

    working = image.astype(np.float64)
    single = working.ndim == 2
    if single:
        working = working[:, :, None]

    half = size // 2
    padded = pad(working, half, half, "reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (size, size), axis=(0, 1))
    out = np.median(windows, axis=(3, 4))
    return out[:, :, 0] if single else out


def gradients(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sobel magnitude and direction (radians) of a grayscale image."""
    gray = to_grayscale(image)
    gx = correlate(gray, sobel_x())
    gy = correlate(gray, sobel_y())
    return np.hypot(gx, gy), np.arctan2(gy, gx)


def sharpen_image(image: np.ndarray, strength: float = 1.0) -> np.ndarray:
    return np.clip(image.astype(np.float64) - strength * correlate(image, laplacian()), 0, 255)


def otsu_threshold(gray: np.ndarray, bins: int = 256) -> float:
    """Otsu's method: the cut that best separates the histogram into two groups.

    It maximises the variance *between* the two groups, which is the same as
    minimising the variance within them but needs only running sums — one pass
    over the histogram rather than a search. No parameter to pick and nothing to
    tune, which is why it beats a hard-coded 128 on any image that is not evenly
    exposed.
    """
    values = gray.ravel()
    if values.size == 0:
        raise ValueError("an empty image has no threshold")

    counts, edges = np.histogram(values, bins=bins, range=(values.min(), values.max() + 1e-9))
    centres = (edges[:-1] + edges[1:]) / 2
    total = counts.sum()
    if total == 0 or counts.max() == total:
        return float(values.min())

    weight_low = np.cumsum(counts)
    weight_high = total - weight_low
    sum_low = np.cumsum(counts * centres)
    sum_total = sum_low[-1]

    usable = (weight_low > 0) & (weight_high > 0)
    mean_low = np.zeros_like(sum_low, dtype=np.float64)
    mean_high = np.zeros_like(sum_low, dtype=np.float64)
    mean_low[usable] = sum_low[usable] / weight_low[usable]
    mean_high[usable] = (sum_total - sum_low[usable]) / weight_high[usable]

    between = np.zeros_like(mean_low)
    between[usable] = (weight_low[usable] * weight_high[usable]
                       * (mean_low[usable] - mean_high[usable]) ** 2)

    # On an image with two clean peaks, every cut in the empty valley between
    # them scores identically, and argmax would return the first — a threshold
    # sitting right against the darker peak, which the first speck of noise
    # pushes over. Taking the middle of the tied range puts it in the centre of
    # the valley, where it belongs.
    best = between.max()
    tied = np.flatnonzero(between >= best - 1e-12)
    return float(centres[tied[len(tied) // 2]])


def threshold(gray: np.ndarray, level: float | None = None) -> np.ndarray:
    """Binarise. With no level given, Otsu picks one."""
    level = otsu_threshold(gray) if level is None else level
    return (gray > level).astype(np.float64) * 255


def equalise(gray: np.ndarray, bins: int = 256) -> np.ndarray:
    """Histogram equalisation: spread the tones across the whole range."""
    values = gray.astype(np.float64)
    counts, edges = np.histogram(values, bins=bins, range=(0, 256))
    cumulative = np.cumsum(counts).astype(np.float64)
    if cumulative[-1] == 0:
        return values
    cumulative = (cumulative - cumulative.min()) / (cumulative[-1] - cumulative.min() or 1)
    centres = (edges[:-1] + edges[1:]) / 2
    return np.interp(values, centres, cumulative * 255)


def non_maximum_suppression(magnitude: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Thin an edge map to single-pixel ridges.

    A Sobel magnitude marks a wide band either side of every edge. Keeping only
    the pixels that are a local maximum *along the gradient direction* is what
    turns that band into a line. Skipping this step is why homemade edge
    detectors produce fat, smeared outlines.
    """
    angle = np.rad2deg(direction) % 180
    out = np.zeros_like(magnitude)
    padded = pad(magnitude, 1, 1, "constant")

    # Four directions, each comparing against the two neighbours along it.
    offsets = {0: ((0, -1), (0, 1)), 45: ((-1, 1), (1, -1)),
               90: ((-1, 0), (1, 0)), 135: ((-1, -1), (1, 1))}
    bucket = np.select(
        [(angle < 22.5) | (angle >= 157.5), (angle < 67.5), (angle < 112.5)],
        [0, 45, 90], default=135)

    height, width = magnitude.shape
    rows, columns = np.mgrid[0:height, 0:width]
    for key, ((dr1, dc1), (dr2, dc2)) in offsets.items():
        here = bucket == key
        first = padded[rows + 1 + dr1, columns + 1 + dc1]
        second = padded[rows + 1 + dr2, columns + 1 + dc2]
        keep = here & (magnitude >= first) & (magnitude >= second)
        out[keep] = magnitude[keep]
    return out


def hysteresis(thin: np.ndarray, low: float, high: float) -> np.ndarray:
    """Keep strong edges, and weak ones only where they touch a strong one.

    A single threshold either breaks long edges into dashes or lets noise
    through. Two thresholds plus connectivity is what holds a faint but real
    contour together while dropping isolated specks.
    """
    if low > high:
        raise ValueError("the low threshold must not exceed the high one")

    strong = thin >= high
    weak = (thin >= low) & ~strong

    kept = strong.copy()
    changed = True
    while changed:
        grown = kept.copy()
        neighbours = np.zeros_like(kept)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == dc == 0:
                    continue
                neighbours |= np.roll(np.roll(kept, dr, axis=0), dc, axis=1)
        grown |= weak & neighbours
        changed = bool((grown != kept).any())
        kept = grown

    return kept.astype(np.float64) * 255


def edges(image: np.ndarray, sigma: float = 1.4, low: float = 0.1, high: float = 0.25) -> np.ndarray:
    """Blur, take gradients, thin them, then link with two thresholds.

    The thresholds are fractions of the strongest gradient in this image, not
    absolute numbers: an absolute threshold that works on a bright photograph
    finds nothing at all in a dim one.
    """
    gray = to_grayscale(image)
    smoothed = blur(gray, sigma)
    magnitude, direction = gradients(smoothed)
    thin = non_maximum_suppression(magnitude, direction)
    peak = thin.max() or 1.0
    return hysteresis(thin, low * peak, high * peak)
