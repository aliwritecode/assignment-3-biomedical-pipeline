"""
Task 1 data preparation: grayscale conversion, 256×256 resize, EDA figures.

The supplied nuclei images are already 256×256 RGB. We still run the conversion
so the pipeline is honest about the assignment spec and so downstream stages
all consume the same processed files.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from skimage.color import rgb2gray
from skimage.io import imread, imsave
from skimage.transform import resize

from . import config


SPLITS = ("train", "val", "test")


def to_gray_uint8(rgb: np.ndarray) -> np.ndarray:
    """RGB/RGBA uint8 → 256×256 grayscale uint8."""
    if rgb.ndim == 3:
        gray = rgb2gray(rgb[..., :3])
    else:
        gray = rgb.astype(np.float64)
        if gray.max() > 1.5:
            gray = gray / 255.0
    gray = resize(
        gray,
        (config.IMG_SIZE, config.IMG_SIZE),
        order=1,
        anti_aliasing=True,
        preserve_range=True,
    )
    gray = np.clip(gray, 0.0, 1.0)
    return (gray * 255).astype(np.uint8)


def prepare_split(split: str) -> list[str]:
    """Write processed grayscale images (and masks, if present) for one split."""
    src_img = config.DATA_RAW / split / "images"
    src_mask = config.DATA_RAW / split / "masks"
    dst_img = config.DATA_PROCESSED / split / "images"
    dst_mask = config.DATA_PROCESSED / split / "masks"
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_mask.mkdir(parents=True, exist_ok=True)

    ids = sorted(p.stem for p in src_img.glob("*.png"))
    for stem in ids:
        rgb = imread(src_img / f"{stem}.png")
        gray = to_gray_uint8(rgb)
        imsave(dst_img / f"{stem}.png", gray, check_contrast=False)
        mask_path = src_mask / f"{stem}.png"
        if mask_path.exists():
            mask = imread(mask_path)
            if mask.ndim == 3:
                mask = mask[..., 0]
            mask = resize(
                mask > 0,
                (config.IMG_SIZE, config.IMG_SIZE),
                order=0,
                anti_aliasing=False,
                preserve_range=True,
            )
            imsave(dst_mask / f"{stem}.png", (mask.astype(np.uint8) * 255), check_contrast=False)
    return ids


def prepare_all() -> dict:
    """Process train/val/test and return per-split id lists."""
    config.ensure_dirs()
    out = {}
    for split in SPLITS:
        out[split] = prepare_split(split)
    return out


def load_gray(split: str, image_id: str) -> np.ndarray:
    return imread(config.DATA_PROCESSED / split / "images" / f"{image_id}.png")


def load_mask(split: str, image_id: str) -> np.ndarray:
    return imread(config.DATA_PROCESSED / split / "masks" / f"{image_id}.png") > 0


def load_rgb(split: str, image_id: str) -> np.ndarray:
    return imread(config.DATA_RAW / split / "images" / f"{image_id}.png")


def plot_eda(n_samples: int = 8) -> Path:
    """Sample RGB + grayscale montage and a train-set intensity histogram."""
    meta = pd.read_csv(config.DATA_RAW / "metadata.csv")
    train_ids = sorted(p.stem for p in (config.DATA_PROCESSED / "train" / "images").glob("*.png"))
    rng = np.random.default_rng(config.SEED)
    sample_ids = list(rng.choice(train_ids, size=min(n_samples, len(train_ids)), replace=False))

    fig = plt.figure(figsize=(12, 8.5))
    gs = fig.add_gridspec(3, n_samples, height_ratios=[1, 1, 1.3])

    for i, stem in enumerate(sample_ids):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(load_rgb("train", stem))
        dens = meta.loc[meta.image_id == stem, "density"].values
        ax.set_title(f"{stem}\n{dens[0] if len(dens) else ''}", fontsize=8)
        ax.axis("off")
        ax2 = fig.add_subplot(gs[1, i])
        ax2.imshow(load_gray("train", stem), cmap="gray")
        ax2.axis("off")
        if i == 0:
            ax.set_ylabel("RGB (raw)", fontsize=9)
            ax2.set_ylabel("Gray 256×256", fontsize=9)

    # pooled histogram of processed train intensities
    axh = fig.add_subplot(gs[2, :])
    pixels = []
    for stem in train_ids:
        pixels.append(load_gray("train", stem).ravel())
    pixels = np.concatenate(pixels)
    axh.hist(pixels, bins=64, color="steelblue", edgecolor="white", density=True)
    axh.set_xlabel("Pixel intensity (0–255)")
    axh.set_ylabel("Density")
    axh.set_title(
        f"Train intensity histogram (n={len(train_ids)} images, "
        f"{len(pixels):,} pixels; mean={pixels.mean():.1f}, std={pixels.std():.1f})"
    )
    axh.spines["top"].set_visible(False)
    axh.spines["right"].set_visible(False)

    fig.suptitle("Task 1 — EDA: synthetic DAPI-like nuclei (train split)", fontsize=12)
    fig.tight_layout()
    out = config.FIG_DIR / "task1_eda.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def dataset_overview() -> dict:
    meta = pd.read_csv(config.DATA_RAW / "metadata.csv")
    return {
        "n_train": int((meta.split == "train").sum()),
        "n_val": int((meta.split == "val").sum()),
        "n_test": int((meta.split == "test").sum()),
        "density_counts": {
            f"{split}/{dens}": int(n)
            for (split, dens), n in meta.groupby(["split", "density"]).size().items()
        },
        "n_objects_range": [int(meta.n_objects.min()), int(meta.n_objects.max())],
        "image_size": [config.IMG_SIZE, config.IMG_SIZE],
        "modality": "synthetic fluorescence microscopy (DAPI-like stained nuclei)",
    }
