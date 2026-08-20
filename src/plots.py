"""Figure helpers used by run_assignment.py so the report has consistent panels."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config


def _save(fig, name: str) -> Path:
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = config.FIG_DIR / name
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return out


def plot_classical_panel(gray, binary, labeled, title: str, name: str) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.1))
    axes[0].imshow(gray, cmap="gray")
    axes[0].set_title("Grayscale input")
    axes[1].imshow(binary, cmap="gray")
    axes[1].set_title("Otsu + morphology")
    axes[2].imshow(labeled, cmap="nipy_spectral")
    axes[2].set_title(f"Connected components (n={int(labeled.max())})")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return _save(fig, name)


def plot_val_triplets(rows: list[dict], name: str) -> Path:
    """rows: dicts with keys image, gt, pred, image_id, dice."""
    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(8.4, 2.6 * n))
    if n == 1:
        axes = np.array([axes])
    col_titles = ["Input (gray)", "Ground-truth mask", "U-Net prediction"]
    for r, row in enumerate(rows):
        axes[r, 0].imshow(row["image"], cmap="gray")
        axes[r, 1].imshow(row["gt"], cmap="gray")
        axes[r, 2].imshow(row["pred"], cmap="gray")
        axes[r, 0].set_ylabel(f"{row['image_id']}\nDice={row['dice']:.3f}", fontsize=8)
        for c in range(3):
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(col_titles[c])
    fig.suptitle("Task 3 — validation: input / ground truth / prediction", fontsize=11)
    fig.tight_layout()
    return _save(fig, name)


def plot_curves(histories: dict[str, dict], name: str) -> Path:
    """Overlay train/val loss and val Dice for each loss variant."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for loss_name, hist in histories.items():
        epochs = range(1, len(hist["train_loss"]) + 1)
        axes[0].plot(epochs, hist["train_loss"], label=f"{loss_name} train")
        axes[0].plot(epochs, hist["val_loss"], linestyle="--", label=f"{loss_name} val")
        axes[1].plot(epochs, hist["val_dice"], label=f"{loss_name} Dice")
        axes[1].plot(epochs, hist["val_iou"], linestyle="--", label=f"{loss_name} IoU")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=7)
    axes[1].set_title("Validation Dice / IoU")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0, 1.02)
    axes[1].legend(fontsize=7)
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Task 3 — U-Net training curves (loss ablation)", fontsize=11)
    fig.tight_layout()
    return _save(fig, name)


def plot_otsu_vs_unet(gray, gt, otsu, unet, ids_title: str, name: str) -> Path:
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.8))
    panels = [
        (gray, "Input"),
        (gt, "Ground truth"),
        (otsu, "Otsu + morph"),
        (unet, "U-Net"),
    ]
    for ax, (im, title) in zip(axes, panels):
        ax.imshow(im, cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(ids_title, fontsize=10)
    fig.tight_layout()
    return _save(fig, name)


def plot_robustness(panels: list[tuple[np.ndarray, str]], name: str, title: str) -> Path:
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 2.8))
    if n == 1:
        axes = [axes]
    for ax, (im, lab) in zip(axes, panels):
        cmap = "gray"
        ax.imshow(im, cmap=cmap)
        ax.set_title(lab, fontsize=8)
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return _save(fig, name)


def plot_metrics_table(df: pd.DataFrame, name: str) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 1.2 + 0.35 * len(df)))
    ax.axis("off")
    tbl = ax.table(
        cellText=df.values,
        colLabels=list(df.columns),
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.1, 1.4)
    ax.set_title("Evaluation metrics (validation split)", fontsize=11, pad=12)
    fig.tight_layout()
    return _save(fig, name)
