from pathlib import Path

import numpy as np
import pytest

from imgkit import (
    blur,
    edges,
    equalise,
    gradients,
    hysteresis,
    median_filter,
    non_maximum_suppression,
    otsu_threshold,
    read,
    sobel_x,
    sobel_y,
    threshold,
    to_grayscale,
    write,
)

DATA = Path(__file__).resolve().parent.parent / "data"


def square(size: int = 60, value: float = 200.0) -> np.ndarray:
    image = np.zeros((size, size))
    image[size // 3:2 * size // 3, size // 3:2 * size // 3] = value
    return image


# ------------------------------------------------------------------ grayscale

def test_grayscale_weights_green_most_as_the_eye_does():
    """A plain (r+g+b)/3 turns a green field and a blue sky into the same grey."""
    green = np.full((2, 2, 3), 0.0)
    green[:, :, 1] = 255
    blue = np.full((2, 2, 3), 0.0)
    blue[:, :, 2] = 255

    assert to_grayscale(green).mean() > to_grayscale(blue).mean() * 5


def test_grayscale_of_a_grey_image_is_that_grey():
    assert to_grayscale(np.full((3, 3, 3), 128.0)) == pytest.approx(np.full((3, 3), 128.0))


# ------------------------------------------------------------------ smoothing

def test_blurring_reduces_the_spread_of_values():
    rng = np.random.default_rng(0)
    noisy = 128 + rng.normal(0, 30, (64, 64))

    assert blur(noisy, 2.0).std() < noisy.std() / 2


def test_the_median_removes_a_speck_that_a_blur_only_smears():
    """A single white pixel drags a mean with it; a median ignores it entirely
    as long as it is outnumbered."""
    image = np.full((21, 21), 50.0)
    image[10, 10] = 255.0

    assert median_filter(image, 3)[10, 10] == 50.0
    assert blur(image, 1.0)[10, 10] > 50.0


def test_the_median_leaves_a_straight_edge_where_it_is():
    image = np.zeros((20, 20))
    image[:, 10:] = 200.0

    filtered = median_filter(image, 3)

    assert filtered[10, 9] == 0.0 and filtered[10, 10] == 200.0


def test_an_even_median_window_is_refused():
    with pytest.raises(ValueError, match="odd"):
        median_filter(np.zeros((5, 5)), 4)


# ------------------------------------------------------------------ gradients

def test_the_horizontal_operator_sees_a_vertical_edge_and_not_a_horizontal_one():
    vertical = np.zeros((20, 20))
    vertical[:, 10:] = 200.0
    horizontal = vertical.T

    from imgkit import correlate
    assert np.abs(correlate(vertical, sobel_x())).max() > 100
    assert np.abs(correlate(vertical, sobel_y())).max() < 1e-9
    assert np.abs(correlate(horizontal, sobel_y())).max() > 100


def test_gradient_direction_points_across_the_edge():
    image = np.zeros((20, 20))
    image[:, 10:] = 200.0

    magnitude, direction = gradients(image)
    peak = np.unravel_index(np.argmax(magnitude), magnitude.shape)

    assert np.rad2deg(direction[peak]) == pytest.approx(0, abs=1)


def test_a_flat_image_has_no_gradient():
    magnitude, _ = gradients(np.full((16, 16), 90.0))
    assert magnitude.max() < 1e-9


# ------------------------------------------------------------------ thresholds

def test_otsu_lands_between_the_two_peaks_not_against_one_of_them():
    """Every cut in an empty valley scores identically; argmax would return the
    first, a threshold the first speck of noise pushes over."""
    image = square()
    assert 80 < otsu_threshold(image) < 120


def test_otsu_on_a_uniform_gradient_finds_the_middle():
    gradient = np.tile(np.linspace(0, 255, 64), (64, 1))
    assert otsu_threshold(gradient) == pytest.approx(127.5, abs=2)


def test_otsu_binarises_a_noisy_image_correctly():
    rng = np.random.default_rng(3)
    image = square()
    noisy = image + rng.normal(0, 12, image.shape)

    binary = threshold(noisy)
    agreement = ((binary > 0) == (image > 0)).mean()

    assert agreement > 0.98


def test_an_image_of_one_value_has_no_threshold_to_find():
    assert otsu_threshold(np.full((8, 8), 42.0)) == pytest.approx(42.0, abs=1)


def test_an_empty_image_is_refused():
    with pytest.raises(ValueError, match="empty image"):
        otsu_threshold(np.array([]))


def test_equalisation_spreads_a_narrow_range_across_the_whole_one():
    narrow = np.clip(np.random.default_rng(1).normal(120, 8, (64, 64)), 0, 255)
    spread = equalise(narrow)

    assert narrow.max() - narrow.min() < 80
    assert spread.max() - spread.min() > 200


# ------------------------------------------------------------------ edges

def test_suppression_thins_a_wide_gradient_band_to_a_ridge():
    """Skipping this step is why homemade edge detectors produce fat outlines."""
    image = np.zeros((40, 40))
    image[:, 20:] = 200.0

    magnitude, direction = gradients(blur(image, 1.0))
    thin = non_maximum_suppression(magnitude, direction)

    assert (thin > 1).sum() < (magnitude > 1).sum() / 2


def test_suppression_keeps_the_strongest_pixel_of_each_ridge():
    image = np.zeros((40, 40))
    image[:, 20:] = 200.0

    magnitude, direction = gradients(image)
    thin = non_maximum_suppression(magnitude, direction)

    assert thin.max() == magnitude.max()


def test_hysteresis_keeps_a_weak_edge_attached_to_a_strong_one():
    thin = np.zeros((5, 9))
    thin[2, 0] = 100.0                 # strong
    thin[2, 1:5] = 40.0                # weak, connected
    thin[2, 7] = 40.0                  # weak, isolated

    kept = hysteresis(thin, low=30, high=90)

    assert kept[2, 4] == 255, "the connected weak edge should survive"
    assert kept[2, 7] == 0, "the isolated speck should not"


def test_hysteresis_rejects_thresholds_the_wrong_way_round():
    with pytest.raises(ValueError, match="must not exceed"):
        hysteresis(np.zeros((3, 3)), low=10, high=5)


def test_edges_outline_a_square_without_filling_it_in():
    image = square(60)
    outline = edges(image)

    assert outline[30, 30] == 0, "the middle of the square is not an edge"
    assert outline[20, 25] > 0 or outline[20, 26] > 0, "its top border is"
    assert (outline > 0).sum() < image.size * 0.1, "an outline, not a blob"


def test_edges_find_nothing_in_a_flat_image():
    assert edges(np.full((40, 40), 120.0)).sum() == 0


def test_edge_thresholds_scale_with_the_image_so_a_dim_one_still_works():
    """An absolute threshold that works on a bright photograph finds nothing in
    a dim one."""
    bright = square(60, value=240)
    dim = square(60, value=30)

    assert (edges(bright) > 0).sum() == pytest.approx((edges(dim) > 0).sum(), rel=0.2)


# ------------------------------------------------------------------ files

def test_the_sample_images_load_at_the_size_they_were_written(tmp_path):
    for name in ("shapes.png", "shapes-noisy.png", "low-contrast.png"):
        image = read(DATA / name)
        assert image.shape[:2] == (320, 320)


def test_writing_clips_rather_than_rescaling(tmp_path):
    """A sharpen that pushes a highlight to 260 should lose that highlight, not
    quietly darken the whole picture to make room for it."""
    image = np.full((4, 4), 250.0)
    image[0, 0] = 400.0

    path = write(image, tmp_path / "clipped.png")
    back = read(path, grayscale=True)

    assert back[0, 0] == 255
    assert back[1, 1] == 250


def test_a_missing_file_is_reported_by_name(tmp_path):
    with pytest.raises(FileNotFoundError, match="no such image"):
        read(tmp_path / "absent.png")


def test_a_round_trip_through_disk_preserves_the_pixels(tmp_path):
    rng = np.random.default_rng(5)
    image = rng.integers(0, 256, (16, 16, 3)).astype(np.float64)

    back = read(write(image, tmp_path / "rt.png"))

    assert np.array_equal(back, image)


def test_the_noisy_sample_really_is_noisier_than_the_clean_one():
    clean = to_grayscale(read(DATA / "shapes.png"))
    noisy = to_grayscale(read(DATA / "shapes-noisy.png"))

    assert np.abs(np.diff(noisy, axis=1)).mean() > np.abs(np.diff(clean, axis=1)).mean() * 3


def test_the_median_filter_cleans_the_noisy_sample_toward_the_clean_one():
    clean = to_grayscale(read(DATA / "shapes.png"))
    noisy = to_grayscale(read(DATA / "shapes-noisy.png"))

    before = np.abs(noisy - clean).mean()
    after = np.abs(median_filter(noisy, 3) - clean).mean()

    assert after < before / 3


def test_the_low_contrast_sample_needs_equalisation():
    image = to_grayscale(read(DATA / "low-contrast.png"))

    assert image.max() - image.min() < 120
    assert equalise(image).max() - equalise(image).min() > 200
