import numpy as np
import pytest

from imgkit import (
    convolve,
    convolve_separable,
    correlate,
    gaussian,
    gaussian_1d,
    is_separable,
    pad,
    separate,
    sobel_x,
)


def ramp(size: int = 7) -> np.ndarray:
    return np.arange(size * size, dtype=np.float64).reshape(size, size)


def test_the_identity_kernel_returns_the_image_unchanged():
    image = ramp()
    identity = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float64)

    assert np.allclose(correlate(image, identity), image)
    assert np.allclose(convolve(image, identity), image)


def test_convolution_flips_the_kernel_and_correlation_does_not():
    """The distinction almost every 'convolution' in image code gets wrong.

    For a symmetric kernel the two are identical and nobody notices. For a Sobel
    operator the result has the opposite sign, and a gradient pointing the wrong
    way survives for years because the magnitude still looks right.
    """
    image = ramp()

    assert np.allclose(correlate(image, sobel_x()), -convolve(image, sobel_x()))
    assert not np.allclose(correlate(image, sobel_x()), convolve(image, sobel_x()))


def test_the_two_agree_whenever_the_kernel_is_symmetric():
    image = ramp()
    symmetric = gaussian(1.0)

    assert np.allclose(correlate(image, symmetric), convolve(image, symmetric))


def test_a_blur_kernel_preserves_the_average_brightness():
    image = ramp(21)
    blurred = correlate(image, gaussian(2.0))

    assert blurred.mean() == pytest.approx(image.mean(), rel=0.02)


def test_separable_matches_the_full_two_dimensional_pass():
    """An optimisation that changes the answer is not an optimisation."""
    image = ramp(41)
    line = gaussian_1d(3.0)

    full = correlate(image, gaussian(3.0))
    split = convolve_separable(image, line, line)

    assert np.abs(full - split).max() < 1e-10


def test_a_gaussian_is_separable_and_a_sobel_is_not():
    assert is_separable(gaussian(1.5))
    assert not is_separable(np.array([[1, 0], [0, 1]], dtype=np.float64))


def test_separate_recovers_halves_whose_outer_product_is_the_kernel():
    kernel = gaussian(2.0)
    row, column = separate(kernel)

    assert np.allclose(np.outer(column, row), kernel)


def test_separating_an_inseparable_kernel_is_refused():
    with pytest.raises(ValueError, match="rank is above one"):
        separate(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64))


def test_even_sided_kernels_are_refused():
    with pytest.raises(ValueError, match="odd sides"):
        correlate(ramp(), np.ones((2, 2)))


def test_padding_reflects_by_default_rather_than_filling_with_black():
    """Zeros put a hard black edge just outside the picture, and every edge
    detector then reports a strong edge all the way around the frame."""
    image = np.full((5, 5), 100.0)

    reflected = pad(image, 1, 1, "reflect")
    zeroed = pad(image, 1, 1, "constant")

    assert reflected[0, 0] == 100.0
    assert zeroed[0, 0] == 0.0


def test_padding_keeps_colour_channels_intact():
    image = np.zeros((4, 4, 3))
    assert pad(image, 2, 2).shape == (8, 8, 3)


def test_an_unknown_padding_mode_lists_the_real_ones():
    with pytest.raises(ValueError, match="unknown padding mode"):
        pad(np.zeros((3, 3)), 1, 1, "invent")


def test_colour_images_keep_their_shape_and_are_filtered_per_channel():
    image = np.zeros((9, 9, 3))
    image[:, :, 0] = 100.0

    blurred = correlate(image, gaussian(1.0))

    assert blurred.shape == image.shape
    assert blurred[:, :, 0].mean() > 50
    assert blurred[:, :, 1].max() == 0
