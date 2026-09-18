"""imgkit: apply a filter to an image, or build a contact sheet of them all."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

from . import __version__
from .convolve import convolve_separable, correlate
from .filters import (
    blur,
    edges,
    equalise,
    gradients,
    median_filter,
    otsu_threshold,
    sharpen_image,
    threshold,
    to_grayscale,
)
from .imageio import read, stack, write
from .kernels import NAMED, emboss, gaussian, gaussian_1d


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="imgkit", description=__doc__)
    parser.add_argument("--version", action="version", version=f"imgkit {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    info = sub.add_parser("info", help="size, range and histogram shape")
    info.add_argument("image", type=Path)

    apply_ = sub.add_parser("apply", help="run one filter")
    apply_.add_argument("image", type=Path)
    apply_.add_argument("filter", choices=["blur", "median", "sharpen", "edges", "threshold",
                                           "equalise", "gray", "gradient", *sorted(NAMED)])
    apply_.add_argument("-o", "--out", type=Path, default=Path("out.png"))
    apply_.add_argument("--sigma", type=float, default=1.5)
    apply_.add_argument("--size", type=int, default=3)

    sheet = sub.add_parser("sheet", help="one image through every filter, tiled")
    sheet.add_argument("image", type=Path)
    sheet.add_argument("-o", "--out", type=Path, default=Path("reports/contact-sheet.png"))

    bench = sub.add_parser("bench", help="separable versus full 2D convolution")
    bench.add_argument("image", type=Path)
    bench.add_argument("--sigma", type=float, default=4.0)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Wraps the real work so that piping into `head` — which closes
    the pipe early — ends quietly instead of printing a BrokenPipeError."""
    try:
        return _run(argv)
    except BrokenPipeError:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        return 0
    except KeyboardInterrupt:
        print(file=sys.stderr)
        return 130
    except (ValueError, FileNotFoundError) as error:
        print(f"imgkit: {error}", file=sys.stderr)
        return 2


def _apply(image: np.ndarray, name: str, sigma: float, size: int) -> np.ndarray:
    gray = to_grayscale(image)
    if name == "blur":
        return blur(image, sigma)
    if name == "median":
        return median_filter(image, size)
    if name == "sharpen":
        return sharpen_image(image)
    if name == "edges":
        return edges(image)
    if name == "threshold":
        return threshold(gray)
    if name == "equalise":
        return equalise(gray)
    if name == "gray":
        return gray
    if name == "gradient":
        magnitude, _ = gradients(image)
        return magnitude / (magnitude.max() or 1) * 255
    if name == "emboss":
        return correlate(gray, emboss()) + 128
    kernel = NAMED[name]()
    result = correlate(image if name in {"box", "gaussian", "sharpen"} else gray, kernel)
    return np.abs(result) if kernel.sum() == 0 else result


def _run(argv: list[str] | None) -> int:
    args = build_parser().parse_args(argv)
    image = read(args.image)

    if args.cmd == "info":
        gray = to_grayscale(image)
        print(f"{args.image.name}: {image.shape[1]}x{image.shape[0]}, "
              f"{image.shape[2] if image.ndim == 3 else 1} channel(s)")
        print(f"  range {gray.min():.1f} to {gray.max():.1f}   mean {gray.mean():.1f}   "
              f"std {gray.std():.1f}")
        print(f"  otsu threshold {otsu_threshold(gray):.1f}")
        counts, _ = np.histogram(gray, bins=8, range=(0, 256))
        widest = counts.max() or 1
        for i, count in enumerate(counts):
            print(f"  {i * 32:>3}-{i * 32 + 31:<3} {'#' * int(count * 40 / widest):<40} {count:>7,}")
        return 0

    if args.cmd == "apply":
        out = write(_apply(image, args.filter, args.sigma, args.size), args.out)
        print(f"wrote {out}")
        return 0

    if args.cmd == "sheet":
        names = ["gray", "blur", "median", "sharpen", "gradient", "edges", "threshold", "equalise"]
        panels = [image] + [_apply(image, name, args.sigma if hasattr(args, "sigma") else 1.5, 3)
                            for name in names]
        out = write(stack(panels, columns=3), args.out)
        print(f"wrote {out}  (original, {', '.join(names)})")
        return 0

    gray = to_grayscale(image)
    kernel = gaussian(args.sigma)
    line = gaussian_1d(args.sigma)

    start = time.perf_counter()
    full = correlate(gray, kernel)
    full_seconds = time.perf_counter() - start

    start = time.perf_counter()
    split = convolve_separable(gray, line, line)
    split_seconds = time.perf_counter() - start

    print(f"{gray.shape[1]}x{gray.shape[0]} image, sigma {args.sigma}, "
          f"{kernel.shape[0]}x{kernel.shape[1]} kernel")
    print(f"  full 2D     {full_seconds * 1000:8.1f} ms   {kernel.size} multiplies per pixel")
    print(f"  separable   {split_seconds * 1000:8.1f} ms   {2 * line.size} multiplies per pixel")
    print(f"  speed-up    {full_seconds / split_seconds:8.1f}x")
    print(f"  largest difference between the two results: "
          f"{np.abs(full - split).max():.2e}")
    return 0
