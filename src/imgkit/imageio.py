"""Reading and writing images. Pillow decodes the file; everything else is ours."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read(path: str | Path, *, grayscale: bool = False) -> np.ndarray:
    """Load an image as float64 in 0–255, shaped (h, w) or (h, w, 3)."""
    from PIL import Image

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no such image: {path}")

    with Image.open(path) as handle:
        image = handle.convert("L" if grayscale else "RGB")
        return np.asarray(image, dtype=np.float64)


def write(image: np.ndarray, path: str | Path) -> Path:
    """Save an array, clipping to 0–255 and rounding to bytes.

    Clipping rather than rescaling: a sharpen that pushes a highlight to 260
    should lose that highlight, not quietly darken the entire picture to make
    room for it.
    """
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = np.clip(np.asarray(image, dtype=np.float64), 0, 255).round().astype(np.uint8)
    Image.fromarray(data, mode="L" if data.ndim == 2 else "RGB").save(path)
    return path


def stack(images: list[np.ndarray], *, columns: int = 2, gap: int = 8,
          background: float = 32.0) -> np.ndarray:
    """Tile images into one contact sheet, for comparing filters side by side."""
    if not images:
        raise ValueError("nothing to stack")

    prepared = [img if img.ndim == 3 else np.repeat(img[:, :, None], 3, axis=2)
                for img in images]
    height = max(img.shape[0] for img in prepared)
    width = max(img.shape[1] for img in prepared)
    rows = (len(prepared) + columns - 1) // columns

    sheet = np.full((rows * height + (rows + 1) * gap,
                     columns * width + (columns + 1) * gap, 3), background, dtype=np.float64)

    for index, img in enumerate(prepared):
        r, c = divmod(index, columns)
        top = gap + r * (height + gap)
        left = gap + c * (width + gap)
        sheet[top:top + img.shape[0], left:left + img.shape[1]] = img

    return sheet
