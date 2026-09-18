"""Writes the sample images in data/.

Generated rather than photographed: every pixel is known, so a test can assert
what a filter should do to it, and there is no licence attached to anything in
the repository.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from imgkit.imageio import write  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data"
SEED = 20260918


def shapes(size: int = 320) -> np.ndarray:
    """Hard edges at known coordinates: a square, a disc, a triangle, a gradient."""
    image = np.full((size, size, 3), 24.0)
    y, x = np.mgrid[0:size, 0:size]

    image[40:140, 40:140] = (210, 190, 80)                       # square

    disc = (x - 230) ** 2 + (y - 90) ** 2 <= 55 ** 2
    image[disc] = (90, 160, 220)                                 # circle

    triangle = (y > 190) & (y < 290) & (np.abs(x - 110) < (y - 190) * 0.8)
    image[triangle] = (220, 100, 110)                            # triangle

    ramp = np.linspace(20, 235, 120)                             # gradient patch
    image[200:280, 190:310] = ramp[None, :, None]
    return image


def noisy(size: int = 320, salt: float = 0.02) -> np.ndarray:
    """The same scene with salt-and-pepper noise, for the median filter."""
    rng = np.random.default_rng(SEED)
    image = shapes(size).copy()

    picked = rng.random((size, size)) < salt
    image[picked] = 255.0
    picked = rng.random((size, size)) < salt
    image[picked] = 0.0
    return image


def low_contrast(size: int = 320) -> np.ndarray:
    """Everything squeezed into a narrow band of greys, for equalisation."""
    rng = np.random.default_rng(SEED + 1)
    y, x = np.mgrid[0:size, 0:size]
    base = 110 + 18 * np.sin(2 * np.pi * x / 90) * np.cos(2 * np.pi * y / 130)
    base += rng.normal(0, 2.5, base.shape)
    return np.clip(base, 0, 255)


def main() -> None:
    DATA.mkdir(exist_ok=True)
    for name, image in (("shapes.png", shapes()),
                        ("shapes-noisy.png", noisy()),
                        ("low-contrast.png", low_contrast())):
        path = write(image, DATA / name)
        print(f"{name}: {image.shape[1]}x{image.shape[0]}  {path.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
