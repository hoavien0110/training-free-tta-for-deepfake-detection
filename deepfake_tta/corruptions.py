from __future__ import annotations

from pathlib import Path
from typing import Literal

import cv2
import numpy as np

PostprocessType = Literal[
    "gaussian_blur",
    "resize",
    "color_saturation",
    "color_contrast",
]

BLUR_KERNELS = {1: 5, 2: 9, 3: 13, 4: 17, 5: 21}
RESIZE_INTERMEDIATE = {1: 128, 2: 85, 3: 64, 4: 51, 5: 41}
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALL_METHODS: tuple[PostprocessType, ...] = (
    "gaussian_blur",
    "resize",
    "color_saturation",
    "color_contrast",
)


def validate_level(level: int) -> None:
    if level not in {1, 2, 3, 4, 5}:
        raise ValueError("level must be one of {1, 2, 3, 4, 5}")


def validate_levels(levels: list[int]) -> None:
    if not levels:
        raise ValueError("levels must not be empty")
    for level in levels:
        validate_level(level)


def gaussian_blur(img_bgr: np.ndarray, level: int) -> np.ndarray:
    validate_level(level)
    kernel = BLUR_KERNELS[level]
    return cv2.GaussianBlur(img_bgr, (kernel, kernel), sigmaX=0)


def resize_degrade(
    img_bgr: np.ndarray,
    level: int,
    output_size: tuple[int, int] = (256, 256),
) -> np.ndarray:
    validate_level(level)
    intermediate = RESIZE_INTERMEDIATE[level]
    small = cv2.resize(img_bgr, (intermediate, intermediate), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, output_size, interpolation=cv2.INTER_LINEAR)


def color_saturation(img_bgr: np.ndarray, level: int) -> np.ndarray:
    validate_level(level)
    img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    y, cr, cb = img_ycrcb[:, :, 0], img_ycrcb[:, :, 1], img_ycrcb[:, :, 2]
    factor = float(level)
    cr = 128.0 + (cr - 128.0) * factor
    cb = 128.0 + (cb - 128.0) * factor
    out = np.stack([y, cr, cb], axis=-1)
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2BGR)


def color_contrast(img_bgr: np.ndarray, level: int) -> np.ndarray:
    validate_level(level)
    img = img_bgr.astype(np.float32)
    mean = img.mean(axis=(0, 1), keepdims=True)
    out = mean + (img - mean) * float(level)
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_postprocessing(
    img_bgr: np.ndarray,
    method: PostprocessType,
    level: int,
    output_size: tuple[int, int] = (256, 256),
) -> np.ndarray:
    if method == "gaussian_blur":
        return gaussian_blur(img_bgr, level)
    if method == "resize":
        return resize_degrade(img_bgr, level, output_size=output_size)
    if method == "color_saturation":
        return color_saturation(img_bgr, level)
    if method == "color_contrast":
        return color_contrast(img_bgr, level)
    raise ValueError(f"Unknown method: {method}")


def iter_image_files(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    )


def process_folder(
    input_dir: str | Path,
    output_root: str | Path,
    dataset_name: str,
    methods: list[PostprocessType] | None = None,
    output_size: tuple[int, int] = (256, 256),
    levels: list[int] | None = None,
) -> None:
    input_path = Path(input_dir)
    output_path = Path(output_root)

    if not input_path.is_dir():
        raise ValueError(f"Input folder does not exist or is not a directory: {input_path}")

    methods = methods or list(ALL_METHODS)
    levels = levels or [1, 2, 3, 4, 5]
    validate_levels(levels)
    output_path.mkdir(parents=True, exist_ok=True)

    image_files = iter_image_files(input_path)
    if not image_files:
        raise ValueError(f"No image files found in: {input_path}")

    for img_file in image_files:
        img = cv2.imread(str(img_file))
        if img is None:
            print(f"[WARN] Cannot read image: {img_file}")
            continue

        img = cv2.resize(img, output_size, interpolation=cv2.INTER_LINEAR)
        rel_path = img_file.relative_to(input_path)

        for method in methods:
            for level in levels:
                out = apply_postprocessing(img, method, level, output_size=output_size)
                save_path = output_path / method / f"level_{level}" / dataset_name / rel_path
                save_path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(save_path), out):
                    print(f"[WARN] Failed to save image: {save_path}")

    print(f"Done. Results saved to: {output_path}")
