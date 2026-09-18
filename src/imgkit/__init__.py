"""Image processing from scratch: convolution, filters, thresholding and edge detection."""

from .convolve import convolve, convolve_separable, correlate, is_separable, pad, separate
from .filters import (
    blur,
    edges,
    equalise,
    gradients,
    hysteresis,
    median_filter,
    non_maximum_suppression,
    otsu_threshold,
    sharpen_image,
    threshold,
    to_grayscale,
)
from .imageio import read, stack, write
from .kernels import NAMED, box, emboss, gaussian, gaussian_1d, laplacian, sharpen, sobel_x, sobel_y

__all__ = [
    "NAMED", "blur", "box", "convolve", "convolve_separable", "correlate", "edges",
    "emboss", "equalise", "gaussian", "gaussian_1d", "gradients", "hysteresis",
    "is_separable", "laplacian", "median_filter", "non_maximum_suppression",
    "otsu_threshold", "pad", "read", "separate", "sharpen", "sharpen_image", "sobel_x",
    "sobel_y", "stack", "threshold", "to_grayscale", "write",
]
__version__ = "1.0.0"
