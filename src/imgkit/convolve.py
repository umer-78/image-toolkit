"""2D convolution, written out.

The distinction this module insists on is convolution versus correlation. Almost
every "convolution" in image-processing code is really correlation: the kernel is
slid over the image and multiplied in place, with no flip. For a symmetric kernel
— a box blur, a Gaussian — the two are identical and nobody notices. For an
asymmetric one, a Sobel operator above all, the result has the opposite sign, and
a gradient that points the wrong way is a bug that survives for years because the
magnitude still looks right.

Both are available here, `convolve` flips and `correlate` does not, and a test
pins the sign difference on a Sobel kernel.
"""

from __future__ import annotations

import numpy as np

PadMode = str
MODES = ("reflect", "edge", "constant", "wrap")


def pad(image: np.ndarray, rows: int, columns: int, mode: PadMode = "reflect",
        value: float = 0.0) -> np.ndarray:
    """Extend the border so every output pixel has a full neighbourhood.

    The default is reflection rather than zeros: padding with zeros puts a hard
    black edge just outside the picture, and every edge detector then reports a
    strong edge all the way around the frame.
    """
    if mode not in MODES:
        raise ValueError(f"unknown padding mode {mode!r} — one of {', '.join(MODES)}")
    if rows < 0 or columns < 0:
        raise ValueError("padding cannot be negative")

    widths = ((rows, rows), (columns, columns)) + ((0, 0),) * (image.ndim - 2)
    if mode == "constant":
        return np.pad(image, widths, mode="constant", constant_values=value)
    return np.pad(image, widths, mode=mode)


def correlate(image: np.ndarray, kernel: np.ndarray, mode: PadMode = "reflect") -> np.ndarray:
    """Slide the kernel over the image without flipping it."""
    if kernel.ndim != 2:
        raise ValueError("the kernel must be two-dimensional")
    if kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
        raise ValueError(f"the kernel must have odd sides, got {kernel.shape}")

    working = image.astype(np.float64)
    single = working.ndim == 2
    if single:
        working = working[:, :, None]

    kh, kw = kernel.shape
    padded = pad(working, kh // 2, kw // 2, mode)

    # One strided view of every kh x kw neighbourhood, then a single tensordot.
    # The explicit four-deep Python loop is the same arithmetic and about two
    # hundred times slower, which makes the difference between a second and three
    # minutes on a photograph.
    height, width, channels = working.shape
    windows = np.lib.stride_tricks.sliding_window_view(padded, (kh, kw), axis=(0, 1))
    out = np.tensordot(windows, kernel, axes=([3, 4], [0, 1]))
    out = out.reshape(height, width, channels)

    return out[:, :, 0] if single else out


def convolve(image: np.ndarray, kernel: np.ndarray, mode: PadMode = "reflect") -> np.ndarray:
    """True convolution: the kernel is flipped in both axes first."""
    return correlate(image, kernel[::-1, ::-1], mode)


def convolve_separable(image: np.ndarray, row: np.ndarray, column: np.ndarray,
                       mode: PadMode = "reflect") -> np.ndarray:
    """Two 1D passes instead of one 2D pass.

    When a 2D kernel is an outer product — a Gaussian is, a box blur is — running
    the two 1D halves in sequence gives an identical result for O(k) work per
    pixel instead of O(k**2). At a 15-pixel radius that is 31 multiplications
    instead of 961. A test checks the outputs match to floating-point precision,
    because an optimisation that changes the answer is not an optimisation.
    """
    if row.ndim != 1 or column.ndim != 1:
        raise ValueError("both halves must be one-dimensional")
    horizontal = correlate(image, row.reshape(1, -1), mode)
    return correlate(horizontal, column.reshape(-1, 1), mode)


def is_separable(kernel: np.ndarray, tolerance: float = 1e-10) -> bool:
    """Can this kernel be split into two 1D passes? True when it is rank one."""
    return bool(np.linalg.matrix_rank(kernel, tol=tolerance) <= 1)


def separate(kernel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a rank-one kernel into its row and column halves."""
    if not is_separable(kernel):
        raise ValueError("this kernel is not separable — its rank is above one")
    u, s, vt = np.linalg.svd(kernel)
    scale = np.sqrt(s[0])
    return vt[0] * scale, u[:, 0] * scale
