# Assignment 3 — Hybrid biomedical image-analysis pipeline

Local pipeline for **synthetic DAPI-like fluorescence nuclei** (assigned modality):

`raw image → grayscale 256×256 → (VLM description | Otsu features | U-Net mask) → regionprops → structured JSON → short narrative`

All large language models run **locally via Ollama**. Outputs are for **educational use only** — none of the models are clinically cleared.

## Dataset

Source: [Nickolay-K/Assingnment-3-dataset](https://github.com/Nickolay-K/Assingnment-3-dataset) (`nuclei_dataset.zip`).

| Split | Images | Masks |
|-------|--------|--------|
| train | 80 | yes |
| val   | 20 | yes |
| test  | 12 | yes (held out as “unseen”) |

Images are 256×256 RGB (blue-stained nuclei on a dark field). Density regimes: sparse / normal / dense / clustered.

## Setup

```bash
# Python 3.12 recommended
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Ollama must be running.
#
# Assignment spec: llama3.2-vision. On Ollama ≥0.30 this may fail with
# `unknown model architecture: 'mllama'`. Module lead (Nickolay) confirmed
# approved alternatives: Qwen2.5-VL / Qwen3-VL / ministral-3:14b, or
# llama3.2-vision in Colab (Lab 2). This repo uses gemma3:4b (+ llava for
# a two-VLM comparison) with the same structured prompts.
#
#   ollama pull llama3.2-vision   # spec model; may not load locally
#   ollama pull gemma3:4b         # local VLM used for Task 1 outputs
#   ollama pull llava             # second VLM (extension comparison)
#   ollama pull llama3.1:8b       # Task 2 / 4 text LLM
```

## Run

```bash
source .venv/bin/activate
python run_assignment.py                 # full pipeline (U-Net + Ollama)
python run_assignment.py --epochs 20     # shorter training
python run_assignment.py --skip-llm      # figures + U-Net only
python run_assignment.py --skip-train    # reuse outputs/models/*.pt
```

Order of work: data prep → U-Net training (MPS/CUDA/CPU) → VLM → numbers-first LLM → hybrid test CSV → robustness.

## Repository layout

```
src/config.py          paths, seeds, model names
src/prompts.py         naive + optimised prompts (copied into the report)
src/ollama_client.py   local Ollama generate / JSON parse
src/data_prep.py       grayscale, resize, EDA
src/classical.py       Otsu, morphology, regionprops, Dice/IoU
src/unet_model.py      SmallUNet + BCE / Dice / BCE+Dice losses
src/train_eval.py      training loop and validation
src/hybrid.py          mask → features → JSON (numbers are source of truth)
src/plots.py           report figures
run_assignment.py      orchestrates Tasks 1–4 + extras
```

## Outputs

| Path | Contents |
|------|----------|
| `outputs/figures/` | EDA, Otsu panel, U-Net triplets, curves, robustness |
| `outputs/csv/test_pipeline_records.csv` | aggregated Task 4 JSON records |
| `outputs/csv/val_metrics_comparison.csv` | Otsu vs U-Net (three losses) |
| `outputs/json/` | VLM/LLM raw + parsed records |
| `outputs/prompts/optimised_prompts.md` | prompts required in the report |
| `outputs/models/unet_*.pt` | best-by-val-Dice checkpoints |
| `outputs/REPORT_NOTES.md` | numbers dumped for the write-up |

## Design notes (auditability)

- **VLM (Task 1)** is anchored as *descriptive, not diagnostic*, forced to JSON, and allowed to answer `"uncertain"`.
- **Numbers-first LLM (Task 2)** never sees the image — only a regionprops summary.
- **Hybrid JSON (Task 4)** copies `n_objects` and `mean_area` from Python after the LLM returns, so a hallucinated count cannot enter the CSV.
- **Loss ablation:** BCE vs Dice vs BCE+Dice; the best validation Dice checkpoint is used for the test pipeline.
- **Robustness extra:** clean vs provided blur / low-contrast (`test_corrupted/`) plus added Gaussian noise, traced through mask → table → narrative.
- **Second VLM extra:** same structured prompt on `gemma3:4b` vs `llava` (`python scripts/extra_vlm_and_otsu_win.py`).

Re-run extras only:

```bash
python scripts/extra_vlm_and_otsu_win.py
```

## Disclaimer

Not for clinical use. Hallucinations in a medical context can cause harm.
