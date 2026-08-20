#!/usr/bin/env python3
"""
One-shot extras that do not retrain the U-Net:

1. Second VLM (llava) on the same representative image / structured prompt.
2. Same VLM prompt on the original RGB image (colour cue vs grayscale).
3. Side-by-side Otsu vs U-Net on the low-contrast corruption (true Otsu win).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src import ollama_client as oc  # noqa: E402
from src.classical import dice_iou, otsu_segment  # noqa: E402
from src.data_prep import load_gray, load_mask, load_rgb, to_gray_uint8  # noqa: E402
from src.plots import plot_otsu_vs_unet  # noqa: E402
from src.prompts import VLM_OPTIMISED  # noqa: E402
from src.train_eval import load_model, predict_mask  # noqa: E402
from skimage.io import imread  # noqa: E402


def main():
    config.ensure_dirs()
    image_id = config.REPRESENTATIVE_ID
    gray_path = config.DATA_PROCESSED / "train" / "images" / f"{image_id}.png"
    rgb_path = config.DATA_RAW / "train" / "images" / f"{image_id}.png"
    gray_b64 = oc.encode_image(gray_path)
    rgb_b64 = oc.encode_image(rgb_path)

    out = {"image_id": image_id}

    # --- second VLM ---
    second = "llava"
    try:
        print(f"structured prompt on {second} (grayscale) …")
        raw = oc.generate(VLM_OPTIMISED, second, images=[gray_b64], temperature=0.2, json_mode=True)
        out["llava_grayscale"] = {"raw": raw, "json": oc.extract_json(raw)}
        oc.unload(second)
    except Exception as e:  # noqa: BLE001
        out["llava_grayscale"] = {"error": str(e)}

    # --- colour vs gray on gemma3 ---
    vlm = "gemma3:4b"
    try:
        print(f"structured prompt on {vlm} (original RGB) …")
        raw = oc.generate(VLM_OPTIMISED, vlm, images=[rgb_b64], temperature=0.2, json_mode=True)
        out["gemma3_rgb"] = {"raw": raw, "json": oc.extract_json(raw)}
        oc.unload(vlm)
    except Exception as e:  # noqa: BLE001
        out["gemma3_rgb"] = {"error": str(e)}

    path = config.JSON_DIR / "extra_vlm_comparison.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("saved", path)

    # --- Otsu win on low-contrast corruption ---
    device = config.get_device()
    model = load_model(config.MODEL_DIR / "unet_bce_dice.pt", device)
    stem = "test_000"
    gt = load_mask("test", stem)
    corrupt = imread(config.DATA_RAW / "test_corrupted" / "images" / f"{stem}_lowcontrast.png")
    gray_c = to_gray_uint8(corrupt) if corrupt.ndim == 3 else corrupt
    otsu_m, _ = otsu_segment(gray_c)
    pred = predict_mask(model, gray_c, device)
    od, _ = dice_iou(otsu_m, gt)
    ud, _ = dice_iou(pred, gt)
    plot_otsu_vs_unet(
        gray_c,
        gt,
        otsu_m,
        pred,
        ids_title=f"{stem} low-contrast: Otsu Dice={od:.3f}  U-Net Dice={ud:.3f} (Otsu better)",
        name="extra_otsu_better_lowcontrast.png",
    )
    print(f"low-contrast panel: Otsu={od:.3f} U-Net={ud:.3f}")


if __name__ == "__main__":
    main()
