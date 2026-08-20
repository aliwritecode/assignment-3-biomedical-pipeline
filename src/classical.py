"""
Task 2 classical pipeline: Otsu → morphology → connected components → regionprops.

Nuclei are bright on a dark field, so the binary foreground is `gray > otsu`.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops_table
from skimage.morphology import closing, disk, opening, remove_small_objects

from . import config


def otsu_segment(
    gray: np.ndarray,
    min_area: int = config.MIN_OBJECT_AREA,
    opening_radius: int = 1,
    closing_radius: int = 2,
) -> tuple[np.ndarray, float]:
    """
    Return a boolean mask and the Otsu threshold (on the 0–1 scale of `gray`).

    `gray` may be uint8 0–255 or float 0–1.
    """
    g = gray.astype(np.float64)
    if g.max() > 1.5:
        g = g / 255.0
    t = float(threshold_otsu(g))
    binary = g > t
    if opening_radius > 0:
        binary = opening(binary, disk(opening_radius))
    if closing_radius > 0:
        binary = closing(binary, disk(closing_radius))
    # skimage 0.26: max_size removes objects with area <= that value
    binary = remove_small_objects(binary.astype(bool), max_size=min_area)
    return binary.astype(bool), t


def label_mask(binary: np.ndarray) -> np.ndarray:
    """4-connected labels; background = 0."""
    return label(binary, connectivity=1)


def region_table(
    binary: np.ndarray,
    intensity: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Per-object table from skimage.measure.regionprops_table."""
    lab = label_mask(binary)
    if lab.max() == 0:
        return pd.DataFrame(
            columns=[
                "label",
                "area",
                "eccentricity",
                "solidity",
                "mean_intensity",
                "equivalent_diameter_area",
                "perimeter",
                "axis_major_length",
                "axis_minor_length",
            ]
        )
    g = intensity
    if g is not None and g.max() > 1.5:
        g = g.astype(np.float64) / 255.0
    props = regionprops_table(
        lab,
        intensity_image=g,
        properties=[
            "label",
            "area",
            "eccentricity",
            "solidity",
            "mean_intensity",
            "equivalent_diameter_area",
            "perimeter",
            "axis_major_length",
            "axis_minor_length",
        ],
    )
    return pd.DataFrame(props)


def density_class(n_objects: int, mean_solidity: float | None = None) -> str:
    """
    Map object count onto the dataset's density vocabulary.

    'clustered' is reserved for moderate counts whose blobs look merged
    (low mean solidity) — a heuristic, not a diagnosis.
    """
    if n_objects <= 0:
        return "uncertain"
    if n_objects <= config.SPARSE_MAX:
        return "sparse"
    if mean_solidity is not None and n_objects <= config.NORMAL_MAX and mean_solidity < 0.85:
        return "clustered"
    if n_objects <= config.NORMAL_MAX:
        return "normal"
    return "dense"


def summarise_table(df: pd.DataFrame, extra: Optional[dict] = None) -> str:
    """Turn a regionprops table into a short natural-language / numeric brief."""
    n = int(len(df))
    lines = [f"n_objects: {n}"]
    if n == 0:
        lines.append("No connected components remained after Otsu + morphological cleanup.")
    else:
        for col in ("area", "eccentricity", "solidity", "mean_intensity", "equivalent_diameter_area"):
            if col not in df.columns:
                continue
            s = df[col].astype(float)
            lines.append(
                f"{col}: mean={s.mean():.4f}, std={s.std(ddof=0):.4f}, "
                f"min={s.min():.4f}, max={s.max():.4f}, median={s.median():.4f}"
            )
        lines.append(f"total_area_px: {float(df['area'].sum()):.1f}")
        lines.append(f"area_fraction: {float(df['area'].sum()) / (config.IMG_SIZE ** 2):.4f}")
        lines.append(f"density_class_heuristic: {density_class(n, float(df['solidity'].mean()))}")
    if extra:
        for k, v in extra.items():
            lines.append(f"{k}: {v}")
    return "\n".join(lines)


def dice_iou(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> tuple[float, float]:
    """Pixel Dice and IoU between two boolean masks."""
    p = pred.astype(bool)
    t = gt.astype(bool)
    inter = float(np.logical_and(p, t).sum())
    ps, ts = float(p.sum()), float(t.sum())
    dice = (2.0 * inter + eps) / (ps + ts + eps)
    union = ps + ts - inter
    iou = (inter + eps) / (union + eps)
    return float(dice), float(iou)
