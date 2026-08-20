"""
U-Net dataset, training loop, and Dice/IoU evaluation.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from . import config
from .unet_model import SmallUNet, get_loss


def set_seed(seed: int = config.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class NucleiDataset(Dataset):
    """Loads processed grayscale PNGs and binary masks as (1, H, W) float tensors."""

    def __init__(self, split: str, augment: bool = False):
        self.img_dir = config.DATA_PROCESSED / split / "images"
        self.mask_dir = config.DATA_PROCESSED / split / "masks"
        self.ids = sorted(p.stem for p in self.img_dir.glob("*.png"))
        self.augment = augment

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        stem = self.ids[idx]
        from skimage.io import imread

        img = imread(self.img_dir / f"{stem}.png").astype(np.float32) / 255.0
        mask = (imread(self.mask_dir / f"{stem}.png") > 0).astype(np.float32)
        if self.augment:
            if random.random() < 0.5:
                img = np.ascontiguousarray(np.fliplr(img))
                mask = np.ascontiguousarray(np.fliplr(mask))
            if random.random() < 0.5:
                img = np.ascontiguousarray(np.flipud(img))
                mask = np.ascontiguousarray(np.flipud(mask))
        img_t = torch.from_numpy(img).unsqueeze(0)
        mask_t = torch.from_numpy(mask).unsqueeze(0)
        return img_t, mask_t, stem


@torch.no_grad()
def batch_dice_iou(logits: torch.Tensor, targets: torch.Tensor, thresh: float = 0.5):
    """Mean Dice / IoU over the batch (pixel-wise, per image then averaged)."""
    probs = torch.sigmoid(logits)
    preds = (probs > thresh).float()
    dims = (1, 2, 3)
    inter = (preds * targets).sum(dim=dims)
    pred_sum = preds.sum(dim=dims)
    tgt_sum = targets.sum(dim=dims)
    dice = (2 * inter + 1e-6) / (pred_sum + tgt_sum + 1e-6)
    union = pred_sum + tgt_sum - inter
    iou = (inter + 1e-6) / (union + 1e-6)
    return float(dice.mean().cpu()), float(iou.mean().cpu())


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device, criterion=None) -> dict:
    model.eval()
    dices, ious, losses = [], [], []
    per_image = []
    for imgs, masks, stems in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)
        logits = model(imgs)
        if criterion is not None:
            losses.append(float(criterion(logits, masks).cpu()))
        # per-image metrics
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        for i, stem in enumerate(stems):
            p = preds[i]
            t = masks[i]
            inter = float((p * t).sum().cpu())
            ps, ts = float(p.sum().cpu()), float(t.sum().cpu())
            dice = (2 * inter + 1e-6) / (ps + ts + 1e-6)
            iou = (inter + 1e-6) / (ps + ts - inter + 1e-6)
            dices.append(dice)
            ious.append(iou)
            per_image.append({"image_id": stem, "dice": dice, "iou": iou})
    out = {
        "mean_dice": float(np.mean(dices)) if dices else 0.0,
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "std_dice": float(np.std(dices)) if dices else 0.0,
        "std_iou": float(np.std(ious)) if ious else 0.0,
        "per_image": per_image,
    }
    if losses:
        out["loss"] = float(np.mean(losses))
    return out


def train_unet(
    loss_name: str = "bce_dice",
    epochs: int = config.EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    lr: float = config.LR,
    device=None,
) -> dict:
    """
    Train one SmallUNet. Saves best-by-val-Dice weights and a history JSON.

    Returns history plus the path of the best checkpoint.
    """
    set_seed(config.SEED)
    device = device or config.get_device()
    train_ds = NucleiDataset("train", augment=True)
    val_ds = NucleiDataset("val", augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = SmallUNet(in_channels=1, out_channels=1, base=config.UNET_BASE).to(device)
    criterion = get_loss(loss_name)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": [], "val_dice": [], "val_iou": []}
    best_dice, best_state = -1.0, None

    for epoch in range(1, epochs + 1):
        model.train()
        running = []
        for imgs, masks, _ in tqdm(train_loader, desc=f"{loss_name} ep{epoch}/{epochs}", leave=False):
            imgs, masks = imgs.to(device), masks.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            opt.step()
            running.append(float(loss.detach().cpu()))
        val = evaluate(model, val_loader, device, criterion=criterion)
        history["train_loss"].append(float(np.mean(running)))
        history["val_loss"].append(val.get("loss", float("nan")))
        history["val_dice"].append(val["mean_dice"])
        history["val_iou"].append(val["mean_iou"])
        if val["mean_dice"] > best_dice:
            best_dice = val["mean_dice"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(
            f"  [{loss_name}] epoch {epoch:02d}  "
            f"train_loss={history['train_loss'][-1]:.4f}  "
            f"val_loss={history['val_loss'][-1]:.4f}  "
            f"val_dice={val['mean_dice']:.4f}  val_iou={val['mean_iou']:.4f}"
        )

    ckpt = config.MODEL_DIR / f"unet_{loss_name}.pt"
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state if best_state is not None else model.state_dict(),
            "loss_name": loss_name,
            "base": config.UNET_BASE,
            "best_val_dice": best_dice,
            "history": history,
        },
        ckpt,
    )
    hist_path = config.JSON_DIR / f"unet_{loss_name}_history.json"
    hist_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    # reload best weights for a clean val pass
    model.load_state_dict(best_state if best_state is not None else model.state_dict())
    final_val = evaluate(model, val_loader, device, criterion=criterion)
    return {
        "loss_name": loss_name,
        "checkpoint": str(ckpt),
        "history": history,
        "best_val_dice": best_dice,
        "final_val": {k: v for k, v in final_val.items() if k != "per_image"},
        "per_image_val": final_val["per_image"],
    }


def load_model(ckpt_path: Path, device=None) -> SmallUNet:
    device = device or config.get_device()
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = SmallUNet(in_channels=1, out_channels=1, base=blob.get("base", config.UNET_BASE))
    model.load_state_dict(blob["state_dict"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_mask(model: SmallUNet, gray: np.ndarray, device=None, thresh: float = 0.5) -> np.ndarray:
    """gray uint8/float H×W → boolean mask."""
    device = device or config.get_device()
    g = gray.astype(np.float32)
    if g.max() > 1.5:
        g = g / 255.0
    x = torch.from_numpy(g)[None, None].to(device)
    logits = model(x)
    pred = (torch.sigmoid(logits)[0, 0] > thresh).cpu().numpy()
    return pred.astype(bool)
