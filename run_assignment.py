#!/usr/bin/env python3
"""
run_assignment.py
=================
End-to-end runner for Assignment 3 (nuclei fluorescence-microscopy pipeline).

What it does
------------
Task 1  Grayscale + 256×256, EDA figures, naive vs structured VLM, non-identical runs
Task 2  Otsu / morphology / regionprops, numbers-first LLM (no image)
Task 3  Train SmallUNet with BCE, Dice, and BCE+Dice; val Dice/IoU; triplet panels
Task 4  Unseen test images: U-Net → regionprops → JSON + narrative → CSV
Extra   Robustness (blur / low-contrast) and loss ablation (included in Task 3)

Usage
-----
    python run_assignment.py              # everything
    python run_assignment.py --skip-llm   # figures + U-Net only (no Ollama)
    python run_assignment.py --skip-train # reuse outputs/models/*.pt
    python run_assignment.py --epochs 20

All LLMs talk to a local Ollama daemon. Educational use only — not clinically cleared.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.util import random_noise

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src import ollama_client as oc  # noqa: E402
from src.classical import (  # noqa: E402
    dice_iou,
    label_mask,
    otsu_segment,
    region_table,
    summarise_table,
)
from src.data_prep import (  # noqa: E402
    dataset_overview,
    load_gray,
    load_mask,
    load_rgb,
    plot_eda,
    prepare_all,
)
from src.hybrid import aggregate_records, features_from_mask, llm_record_and_narrative  # noqa: E402
from src.plots import (  # noqa: E402
    plot_classical_panel,
    plot_curves,
    plot_metrics_table,
    plot_otsu_vs_unet,
    plot_robustness,
    plot_val_triplets,
)
from src.prompts import CLASSICAL_LLM_PROMPT, VLM_NAIVE, VLM_OPTIMISED, dump_prompt_record  # noqa: E402
from src.train_eval import evaluate, load_model, predict_mask, train_unet  # noqa: E402
from src.unet_model import get_loss  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from src.train_eval import NucleiDataset  # noqa: E402


def _dump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


# ---------------------------------------------------------------------------
# Task 1
# ---------------------------------------------------------------------------

def task1_prepare() -> dict:
    print("\n=== Task 1: data preparation and EDA ===")
    ids = prepare_all()
    overview = dataset_overview()
    fig = plot_eda()
    overview["eda_figure"] = str(fig)
    overview["processed_counts"] = {k: len(v) for k, v in ids.items()}
    _dump(overview, config.JSON_DIR / "dataset_overview.json")
    dump_prompt_record(config.PROMPT_DIR / "optimised_prompts.md")
    print(f"  processed {overview['processed_counts']}")
    print(f"  EDA figure → {fig}")
    return overview


def task1_vlm(skip_llm: bool) -> dict:
    print("\n=== Task 1: multimodal LLM (llama3.2-vision) ===")
    image_id = config.REPRESENTATIVE_ID
    gray_path = config.DATA_PROCESSED / "train" / "images" / f"{image_id}.png"
    b64 = oc.encode_image(gray_path)
    out = {"image_id": image_id, "image_path": str(gray_path), "requested_model": config.VLM_MODEL}

    if skip_llm:
        out["skipped"] = True
        _dump(out, config.JSON_DIR / "task1_vlm.json")
        return out

    vlm = oc.resolve_vlm()
    out["model"] = vlm

    print(f"  naive prompt on {vlm} …")
    naive = oc.generate(
        VLM_NAIVE, vlm, images=[b64], temperature=0.7, json_mode=False
    )
    print("  structured prompt …")
    structured = oc.generate(
        VLM_OPTIMISED, vlm, images=[b64], temperature=0.2, json_mode=True
    )
    print("  three repeated structured runs (temperature=0.8) to show non-identity …")
    repeats = []
    for i in range(3):
        repeats.append(
            oc.generate(
                VLM_OPTIMISED,
                vlm,
                images=[b64],
                temperature=0.8,
                json_mode=True,
            )
        )

    parsed_struct, parsed_repeats = None, []
    try:
        parsed_struct = oc.extract_json(structured)
    except Exception as e:  # noqa: BLE001
        parsed_struct = {"parse_error": str(e), "raw": structured}
    for r in repeats:
        try:
            parsed_repeats.append(oc.extract_json(r))
        except Exception:
            parsed_repeats.append({"raw": r})

    identical = len({json.dumps(p, sort_keys=True) for p in parsed_repeats}) == 1
    out.update(
        {
            "naive_prompt": VLM_NAIVE,
            "optimised_prompt": VLM_OPTIMISED,
            "naive_response": naive,
            "structured_raw": structured,
            "structured_json": parsed_struct,
            "repeated_runs_raw": repeats,
            "repeated_runs_json": parsed_repeats,
            "repeated_runs_identical": identical,
        }
    )
    _dump(out, config.JSON_DIR / "task1_vlm.json")
    # also a human-readable comparison
    (config.JSON_DIR / "task1_naive_vs_structured.txt").write_text(
        "NAIVE PROMPT\n"
        + VLM_NAIVE
        + "\n\nNAIVE RESPONSE\n"
        + naive
        + "\n\nSTRUCTURED PROMPT\n"
        + VLM_OPTIMISED
        + "\n\nSTRUCTURED RESPONSE\n"
        + structured
        + "\n\nREPEATED RUNS IDENTICAL? "
        + str(identical)
        + "\n",
        encoding="utf-8",
    )
    oc.unload(vlm)

    # Extra credit: second vision model on the same image + same structured prompt
    second = next((m for m in config.VLM_FALLBACKS if m != vlm), None)
    installed = oc.listed_models()
    if second and second not in installed and second.split(":")[0] not in installed:
        second = None
    if second:
        try:
            print(f"  extra: second VLM ({second}) structured prompt …")
            raw2 = oc.generate(
                VLM_OPTIMISED, second, images=[b64], temperature=0.2, json_mode=True
            )
            try:
                parsed2 = oc.extract_json(raw2)
            except Exception:
                parsed2 = {"raw": raw2}
            out["second_vlm"] = {"model": second, "structured_json": parsed2, "raw": raw2}
            oc.unload(second)
        except Exception as e:  # noqa: BLE001
            out["second_vlm"] = {"model": second, "error": str(e)}
        _dump(out, config.JSON_DIR / "task1_vlm.json")

    print(f"  naive vs structured saved; repeated_runs_identical={identical}")
    return out


# ---------------------------------------------------------------------------
# Task 2
# ---------------------------------------------------------------------------

def task2_classical_and_llm(skip_llm: bool) -> dict:
    print("\n=== Task 2: classical features + numbers-first LLM ===")
    image_id = config.REPRESENTATIVE_ID
    gray = load_gray("train", image_id)
    gt = load_mask("train", image_id)
    binary, thresh = otsu_segment(gray)
    labeled = label_mask(binary)
    table = region_table(binary, intensity=gray)
    csv_path = config.CSV_DIR / f"{image_id}_regionprops.csv"
    table.to_csv(csv_path, index=False)
    d, iou = dice_iou(binary, gt)
    extra = {"otsu_threshold": round(thresh, 4), "dice_vs_gt": round(d, 4), "iou_vs_gt": round(iou, 4)}
    summary = summarise_table(table, extra=extra)
    plot_classical_panel(
        gray,
        binary,
        labeled,
        title=f"Task 2 — {image_id}  (Otsu t={thresh:.3f}, Dice={d:.3f})",
        name="task2_otsu_panel.png",
    )
    (config.JSON_DIR / "task2_numeric_summary.txt").write_text(summary, encoding="utf-8")

    result = {
        "image_id": image_id,
        "otsu_threshold": thresh,
        "n_objects": int(len(table)),
        "dice_vs_gt": d,
        "iou_vs_gt": iou,
        "numeric_summary": summary,
        "regionprops_csv": str(csv_path),
    }

    if skip_llm:
        result["skipped_llm"] = True
        _dump(result, config.JSON_DIR / "task2_classical.json")
        return result

    prompt = CLASSICAL_LLM_PROMPT.format(summary=summary)
    print("  numbers-first LLM (llama3.1:8b, no image) …")
    raw = oc.generate(prompt, config.LLM_MODEL, temperature=0.2, json_mode=False)
    try:
        parsed = oc.extract_json(raw)
    except Exception as e:  # noqa: BLE001
        parsed = {"parse_error": str(e)}
    paragraph = oc.extract_paragraph(raw)
    result.update(
        {
            "prompt": prompt,
            "llm_raw": raw,
            "llm_paragraph": paragraph,
            "llm_json": parsed,
        }
    )
    _dump(result, config.JSON_DIR / "task2_classical.json")
    oc.unload(config.LLM_MODEL)
    print(f"  objects={result['n_objects']}  Dice(Otsu vs GT)={d:.3f}")
    return result


# ---------------------------------------------------------------------------
# Task 3
# ---------------------------------------------------------------------------

def _otsu_val_metrics() -> list[dict]:
    rows = []
    for p in sorted((config.DATA_PROCESSED / "val" / "images").glob("*.png")):
        gray = imread(p)
        gt = load_mask("val", p.stem)
        binary, _ = otsu_segment(gray)
        d, iou = dice_iou(binary, gt)
        rows.append({"image_id": p.stem, "dice": d, "iou": iou, "method": "otsu"})
    return rows


def task3_unet(skip_train: bool, epochs: int) -> dict:
    print("\n=== Task 3: U-Net training and evaluation ===")
    device = config.get_device()
    print(f"  device={device}  epochs={epochs}")
    losses = ["bce", "dice", "bce_dice"]
    results = {}
    histories = {}

    for loss_name in losses:
        ckpt = config.MODEL_DIR / f"unet_{loss_name}.pt"
        if skip_train and ckpt.exists():
            print(f"  loading existing {ckpt.name}")
            model = load_model(ckpt, device)
            hist_path = config.JSON_DIR / f"unet_{loss_name}_history.json"
            hist = json.loads(hist_path.read_text()) if hist_path.exists() else {}
            val_loader = DataLoader(NucleiDataset("val"), batch_size=config.BATCH_SIZE, shuffle=False)
            val = evaluate(model, val_loader, device, criterion=get_loss(loss_name))
            results[loss_name] = {
                "loss_name": loss_name,
                "checkpoint": str(ckpt),
                "history": hist,
                "final_val": {k: v for k, v in val.items() if k != "per_image"},
                "per_image_val": val["per_image"],
            }
        else:
            results[loss_name] = train_unet(loss_name=loss_name, epochs=epochs, device=device)
        histories[loss_name] = results[loss_name]["history"]

    # pick best by validation Dice
    best_name = max(results, key=lambda k: results[k]["final_val"]["mean_dice"])
    best_ckpt = results[best_name]["checkpoint"]
    print(f"  best loss = {best_name}  Dice={results[best_name]['final_val']['mean_dice']:.4f}")

    # Otsu baseline on the same val split
    otsu_rows = _otsu_val_metrics()
    otsu_dice = float(np.mean([r["dice"] for r in otsu_rows]))
    otsu_iou = float(np.mean([r["iou"] for r in otsu_rows]))

    metrics_tbl = pd.DataFrame(
        [
            {"method": "Otsu + morphology", "val_Dice": round(otsu_dice, 4), "val_IoU": round(otsu_iou, 4)},
            *[
                {
                    "method": f"U-Net ({name})",
                    "val_Dice": round(results[name]["final_val"]["mean_dice"], 4),
                    "val_IoU": round(results[name]["final_val"]["mean_iou"], 4),
                }
                for name in losses
            ],
        ]
    )
    metrics_tbl.to_csv(config.CSV_DIR / "val_metrics_comparison.csv", index=False)
    plot_metrics_table(metrics_tbl, "task3_metrics_table.png")
    if any(histories[n] for n in losses):
        plot_curves({k: v for k, v in histories.items() if v}, "task3_curves.png")

    # three validation triplets from the best model (low / mid / high Dice)
    model = load_model(best_ckpt, device)
    per = sorted(results[best_name]["per_image_val"], key=lambda r: r["dice"])
    pick = [per[0], per[len(per) // 2], per[-1]]  # worst, median, best
    rows = []
    for item in pick:
        gray = load_gray("val", item["image_id"])
        gt = load_mask("val", item["image_id"])
        pred = predict_mask(model, gray, device)
        rows.append({"image": gray, "gt": gt, "pred": pred, "image_id": item["image_id"], "dice": item["dice"]})
    plot_val_triplets(rows, "task3_val_triplets.png")

    # one image where each method did better
    unet_map = {r["image_id"]: r["dice"] for r in results[best_name]["per_image_val"]}
    otsu_map = {r["image_id"]: r["dice"] for r in otsu_rows}
    deltas = {i: unet_map[i] - otsu_map[i] for i in unet_map}
    unet_wins = max(deltas, key=deltas.get)
    otsu_wins = min(deltas, key=deltas.get)
    for stem, tag in ((unet_wins, "unet_better"), (otsu_wins, "otsu_better")):
        gray = load_gray("val", stem)
        gt = load_mask("val", stem)
        otsu_m, _ = otsu_segment(gray)
        pred = predict_mask(model, gray, device)
        ud, _ = dice_iou(pred, gt)
        od, _ = dice_iou(otsu_m, gt)
        plot_otsu_vs_unet(
            gray,
            gt,
            otsu_m,
            pred,
            ids_title=f"{stem}: Otsu Dice={od:.3f}  U-Net Dice={ud:.3f}  ({tag})",
            name=f"task3_{tag}.png",
        )

    summary = {
        "best_loss": best_name,
        "best_checkpoint": best_ckpt,
        "metrics_table": metrics_tbl.to_dict(orient="records"),
        "otsu_val_dice": otsu_dice,
        "otsu_val_iou": otsu_iou,
        "unet_better_example": unet_wins,
        "otsu_better_example": otsu_wins,
        "delta_unet_minus_otsu": {k: round(v, 4) for k, v in deltas.items()},
        "per_image_best_unet": results[best_name]["per_image_val"],
        "triplet_ids": [r["image_id"] for r in pick],
    }
    _dump(summary, config.JSON_DIR / "task3_summary.json")
    return summary


# ---------------------------------------------------------------------------
# Task 4
# ---------------------------------------------------------------------------

def task4_hybrid(best_ckpt: str, skip_llm: bool) -> pd.DataFrame:
    print("\n=== Task 4: hybrid pipeline on unseen test images ===")
    device = config.get_device()
    model = load_model(best_ckpt, device)
    records = []
    test_ids = sorted(p.stem for p in (config.DATA_PROCESSED / "test" / "images").glob("*.png"))

    for i, image_id in enumerate(test_ids, 1):
        gray = load_gray("test", image_id)
        pred = predict_mask(model, gray, device)
        gt = load_mask("test", image_id)
        d, iou = dice_iou(pred, gt)
        table, computed, summary = features_from_mask(gray, pred)
        table.to_csv(config.CSV_DIR / "test_regionprops" / f"{image_id}.csv", index=False)
        print(f"  [{i}/{len(test_ids)}] {image_id}  n={computed['n_objects']}  Dice={d:.3f}")
        if skip_llm:
            rec = {
                "image_id": image_id,
                "n_objects": computed["n_objects"],
                "mean_area": computed["mean_area"],
                "density_class": computed["density_class_computed"],
                "quality_flag": computed.get("quality_flag_computed", "ok"),
                "narrative": "",
            }
            rec.update(computed)
            rec["dice_vs_gt"] = d
            rec["iou_vs_gt"] = iou
            raw = ""
        else:
            rec, narrative, raw = llm_record_and_narrative(image_id, summary, computed)
            rec["dice_vs_gt"] = d
            rec["iou_vs_gt"] = iou
            rec["narrative"] = narrative
        rec["unet_dice_vs_gt"] = d
        rec["unet_iou_vs_gt"] = iou
        records.append(rec)
        _dump({"record": rec, "llm_raw": raw, "numeric_summary": summary}, config.JSON_DIR / "test" / f"{image_id}.json")

    df = aggregate_records(records, config.CSV_DIR / "test_pipeline_records.csv")
    print(f"  CSV → {config.CSV_DIR / 'test_pipeline_records.csv'}")
    if not skip_llm:
        oc.unload(config.LLM_MODEL)
    return df


# ---------------------------------------------------------------------------
# Extra credit: robustness
# ---------------------------------------------------------------------------

def extra_robustness(best_ckpt: str, skip_llm: bool) -> dict:
    """
    Trace a clean image vs provided blur / low-contrast corruptions, and also
    a locally generated noisy copy, through mask → features → narrative.
    """
    print("\n=== Extra: robustness (corruption propagation) ===")
    device = config.get_device()
    model = load_model(best_ckpt, device)
    image_id = "test_000"
    gray_clean = load_gray("test", image_id)
    gt = load_mask("test", image_id)

    corrupt_dir = config.DATA_RAW / "test_corrupted" / "images"
    variants = {
        "clean": gray_clean,
        "blur": imread(corrupt_dir / f"{image_id}_blur.png"),
        "lowcontrast": imread(corrupt_dir / f"{image_id}_lowcontrast.png"),
    }
    # local extra: heavy Gaussian noise on the processed grayscale
    rng = np.random.default_rng(config.SEED)
    noisy = gray_clean.astype(np.float64) / 255.0
    noisy = random_noise(noisy, mode="gaussian", var=0.04, rng=rng)
    variants["noise"] = (np.clip(noisy, 0, 1) * 255).astype(np.uint8)

    rows = []
    panels_img = []
    panels_mask = []
    for name, img in variants.items():
        if img.ndim == 3:
            from src.data_prep import to_gray_uint8

            img = to_gray_uint8(img)
        pred = predict_mask(model, img, device)
        otsu_m, _ = otsu_segment(img)
        d_u, _ = dice_iou(pred, gt)
        d_o, _ = dice_iou(otsu_m, gt)
        table, computed, summary = features_from_mask(img, pred)
        rec = {
            "variant": name,
            "n_objects": computed["n_objects"],
            "mean_area": computed["mean_area"],
            "area_fraction": computed["area_fraction"],
            "unet_dice_vs_clean_gt": round(d_u, 4),
            "otsu_dice_vs_clean_gt": round(d_o, 4),
            "mean_intensity": computed["mean_intensity"],
        }
        if not skip_llm:
            llm_rec, narrative, _ = llm_record_and_narrative(f"{image_id}_{name}", summary, computed)
            rec["density_class"] = llm_rec["density_class"]
            rec["quality_flag"] = llm_rec["quality_flag"]
            rec["narrative"] = narrative
        rows.append(rec)
        panels_img.append((img, f"{name}\ninput"))
        panels_mask.append((pred, f"{name}\nU-Net n={computed['n_objects']} Dice={d_u:.2f}"))
        print(f"  {name:12s}  n={computed['n_objects']:3d}  U-Net Dice={d_u:.3f}  Otsu Dice={d_o:.3f}")

    plot_robustness(panels_img, "extra_robustness_inputs.png", f"Robustness — {image_id} inputs")
    plot_robustness(panels_mask, "extra_robustness_masks.png", f"Robustness — {image_id} U-Net masks")
    df = pd.DataFrame(rows)
    df.to_csv(config.CSV_DIR / "robustness_trace.csv", index=False)
    _dump({"image_id": image_id, "rows": rows}, config.JSON_DIR / "extra_robustness.json")
    if not skip_llm:
        oc.unload(config.LLM_MODEL)
    return {"image_id": image_id, "rows": rows}


# ---------------------------------------------------------------------------
# Notes file (not the 4-page report — numbers + prompts for the write-up)
# ---------------------------------------------------------------------------

def write_notes(task1, task2, task3, test_df: pd.DataFrame) -> None:
    lines = [
        "# Materials for the 4-page report (auto-generated)",
        "",
        "This file is NOT the submission report. It dumps prompts, metrics, and",
        "example records so the report can be written from actual run outputs.",
        "",
        "## Prompts",
        "See outputs/prompts/optimised_prompts.md",
        "",
        "## Task 3 metrics",
        json.dumps(task3.get("metrics_table"), indent=2),
        "",
        f"Best U-Net loss: {task3.get('best_loss')}",
        f"U-Net-better example image: {task3.get('unet_better_example')}",
        f"Otsu-better example image: {task3.get('otsu_better_example')}",
        "",
        "## Task 4 test CSV head",
        test_df.head(12).to_string(index=False) if test_df is not None else "(skipped)",
        "",
        "## Task 1 structured JSON",
        json.dumps(task1.get("structured_json"), indent=2, default=str),
        "",
        "## Task 2 LLM JSON / paragraph",
        json.dumps(task2.get("llm_json"), indent=2, default=str),
        "",
        task2.get("llm_paragraph") or "",
        "",
        "## Repeated VLM runs identical?",
        str(task1.get("repeated_runs_identical")),
    ]
    (config.OUTPUTS / "REPORT_NOTES.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser(description="Assignment 3 hybrid nuclei pipeline")
    p.add_argument("--skip-llm", action="store_true", help="Skip all Ollama calls")
    p.add_argument("--skip-train", action="store_true", help="Reuse saved U-Net weights")
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--no-robustness", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 200, "font.size": 9})
    config.ensure_dirs()
    (config.CSV_DIR / "test_regionprops").mkdir(parents=True, exist_ok=True)

    t1 = task1_prepare()
    t3 = task3_unet(skip_train=args.skip_train, epochs=args.epochs)
    t1v = task1_vlm(skip_llm=args.skip_llm)
    t2 = task2_classical_and_llm(skip_llm=args.skip_llm)
    t4 = task4_hybrid(t3["best_checkpoint"], skip_llm=args.skip_llm)
    extra = None
    if not args.no_robustness:
        extra = extra_robustness(t3["best_checkpoint"], skip_llm=args.skip_llm)
    write_notes({**t1, **t1v}, t2, t3, t4)
    print("\nDone. Figures →", config.FIG_DIR)
    print("CSV     →", config.CSV_DIR)
    print("JSON    →", config.JSON_DIR)
    print("Models  →", config.MODEL_DIR)
    if extra:
        print("Robustness traced on", extra["image_id"])


if __name__ == "__main__":
    main()
