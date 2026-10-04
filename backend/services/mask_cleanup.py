"""Small, dependency-light binary-mask morphology helpers.

The operations are deliberately opt-in: zero-sized kernels return the input mask
unchanged so existing model baselines remain exactly reproducible.
"""
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

ALLOWED_KERNEL_SIZES = (0, 3, 5, 7)


def _validate_kernel(size: int, name: str) -> int:
    if isinstance(size, bool) or size not in ALLOWED_KERNEL_SIZES:
        raise ValueError(f"{name} must be one of {ALLOWED_KERNEL_SIZES} pixels.")
    return int(size)


def clean_binary_mask(
    mask: Any,
    opening_px: int = 0,
    closing_px: int = 0,
) -> np.ndarray:
    """Apply optional binary opening then closing with square structuring elements.

    Opening (erosion then dilation) removes small isolated foreground specks.
    Closing (dilation then erosion) fills tiny holes and joins very small gaps.
    These are geometric heuristics, not learned corrections; they can also remove
    small objects or merge nearby ones. Defaults are zero/no operation.
    """
    opening_px = _validate_kernel(opening_px, "opening_px")
    closing_px = _validate_kernel(closing_px, "closing_px")
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError("Binary mask must be a two-dimensional array.")
    binary = (array > 0).astype(np.uint8)
    if opening_px == 0 and closing_px == 0:
        return binary.copy()

    image = Image.fromarray(binary * 255)
    if opening_px:
        image = image.filter(ImageFilter.MinFilter(opening_px))
        image = image.filter(ImageFilter.MaxFilter(opening_px))
    if closing_px:
        image = image.filter(ImageFilter.MaxFilter(closing_px))
        image = image.filter(ImageFilter.MinFilter(closing_px))
    return (np.asarray(image) >= 128).astype(np.uint8)
