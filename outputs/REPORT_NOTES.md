# Materials for the 4-page report (auto-generated)

This is **not** the submission PDF. It dumps prompts, numbers, figure paths, and draft answers
so the report can be written from actual run outputs. Copy figures from `outputs/figures/`.

Educational use only. Models are not clinically cleared.

## What was assigned

- **Modality:** synthetic fluorescence microscopy (DAPI-like stained nuclei), 256×256.
- **Splits:** 80 train / 20 val / 12 unseen test. Density regimes: sparse / normal / dense / clustered.
- **Local LLMs via Ollama.** Spec model `llama3.2-vision` could not load on Ollama 0.32.14
  (`unknown model architecture: 'mllama'` — [ollama#16490](https://github.com/ollama/ollama/issues/16490)).
  Task 1 therefore used **gemma3:4b** (same prompts). Extra: **llava** as a second VLM.
  Text steps used **llama3.1:8b**.

## Prompts (must appear in the report)

Full text: `outputs/prompts/optimised_prompts.md`

- Task 1 naive: `Describe this medical image.`
- Task 1 structured: descriptive-not-diagnostic, JSON keys
  `modality, tissue_type, notable_features, image_quality`, `"uncertain"` allowed.
- Task 2 / 4: numbers only; model never sees the image; JSON after a PARAGRAPH block.

## Suggested figures (keep the PDF to 4 pages)

| File | Use |
|------|-----|
| `outputs/figures/task1_eda.png` | sample RGB + gray + intensity histogram |
| `outputs/figures/task2_otsu_panel.png` | Otsu + connected components on `train_001` |
| `outputs/figures/task3_val_triplets.png` | val_005 / val_016 / val_018 input–GT–pred |
| `outputs/figures/task3_curves.png` | loss + Dice/IoU ablation |
| `outputs/figures/task3_metrics_table.png` | Otsu vs three U-Net losses |
| `outputs/figures/task3_unet_better.png` | val_005 (clustered): U-Net Dice 0.994 vs Otsu 0.970 |
| `outputs/figures/extra_otsu_better_lowcontrast.png` | **true Otsu win** on low-contrast `test_000` |
| `outputs/figures/extra_robustness_masks.png` | corruption → mask |

## Task 1 — VLM

Representative image: **train_001** (GT: normal, 29 nuclei).

**Naive (gemma3:4b, gray):** free-text microscopy description that then *diagnoses* (“cysts / cells / calcifications”) and defers to a pathologist — exactly the behaviour the structured prompt is designed to block.

**Structured JSON (gray):**
```json
{"modality":"microscopy","tissue_type":"unknown","notable_features":["multiple circular and oval shapes","varying sizes","distributed","grayscale tones"],"image_quality":"acceptable"}
```
Tissue type is `"unknown"` (the prompt allowed this) because grayscale drops the DAPI-blue cue.

**Same prompt on original RGB:** `tissue_type: "cells"`, `notable_features` includes `"objects are blue"`. Colour helps; still not “nuclei / fluorescence”.

**Repeated runs (T=0.8), not identical:** wording of `notable_features` changes (`structures` vs `shapes`; `gray scale` vs `dark background`). Schema is stable.

**llava (extra, same gray image, same prompt):** hallucinates `"modality": "MRI"`, `"tissue_type": "Bone"`, `"Bone marrow"`. Gemma3 is more useful *and* more trustworthy here.

## Task 2 — numbers-first

Otsu t=0.133 on `train_001`. After opening/closing/`remove_small_objects`:

- n_objects = **23** (GT instance count = **29** — touching nuclei merge)
- Dice vs GT mask = **0.968**, IoU = **0.937**
- mean area 244 px, mean eccentricity 0.61, mean solidity 0.93, area_fraction 0.086 (GT 0.086)

**LLM JSON (never saw the image):** n_objects=23, density_class=normal, shape_regularity=uncertain, quality_flag=ok.
Paragraph restates the numbers; it does not invent a diagnosis.

## Task 3 — U-Net (30 epochs, MPS, base=32, Adam 1e-3)

Validation (best checkpoint per loss):

| method | val Dice | val IoU |
|--------|----------|---------|
| Otsu + morphology | 0.9761 | 0.9533 |
| U-Net BCE | 0.9961 | 0.9922 |
| U-Net Dice | 0.9956 | 0.9912 |
| U-Net BCE+Dice | **0.9961** | **0.9923** |

Best loss for the hybrid pipeline: **BCE+Dice** (ties BCE on Dice, slightly higher IoU; Dice-only was jumpy in the first 10 epochs).

Per-image val Dice is uniformly high (~0.994–0.998). Worst val image **val_005** (clustered, 48 GT instances). Errors are thin boundary pixels / merged clumps, not missed fields.

On **clean** val images U-Net beats Otsu on every case (smallest gap val_011, +0.014 Dice). The assignment question still needs an Otsu-better *example*: use **low-contrast `test_000`**, where Otsu Dice=0.928 and U-Net Dice=0.046 (U-Net collapsed to ~empty / tiny speckle). That is a distribution-shift failure: the net never saw low-contrast training images.

## Task 4 — hybrid test CSV

`outputs/csv/test_pipeline_records.csv`

Pixel overlap vs GT masks is excellent (Dice 0.995–0.997) and **area_fraction matches GT to ~0.001**. Connected-component **counts under-count** on dense/clustered fields because the U-Net is *semantic* (one blob for touching nuclei):

- sparse `test_000` / `test_006`: 8 vs 8
- dense `test_010`: 38 vs 78 GT instances
- clustered `test_005`: 16 vs 49

JSON `n_objects` and `mean_area` are written from Python after the LLM returns (`llm_n_objects_raw` is kept for audit). `quality_flag` is also overwritten from a numeric rule (empty / area_fraction≥0.8 / n≥200 → `review`).

Example test_000 record:
```json
{"image_id":"test_000","n_objects":8,"mean_area":191.12,"density_class":"sparse","quality_flag":"ok"}
```

## Extra — robustness (`test_000`)

| variant | n_objects | U-Net Dice vs clean GT | Otsu Dice | quality_flag |
|---------|-----------|------------------------|-----------|--------------|
| clean | 8 | 0.996 | 0.983 | ok |
| blur | 7 | 0.644 | 0.703 | ok |
| lowcontrast | 1 (area_fraction=1.0 collapsed) | 0.046 | **0.928** | review |
| noise | 1483 | 0.204 | 0.566 | review |

**Earliest detectable stage**

- **Low contrast:** detectable in the *image* (mean intensity jumps 0.22 → 0.42) before a useful mask exists. U-Net mask is already unusable; Otsu still works. LLM narrative is misleading unless JSON is gated on `quality_flag`.
- **Blur:** first obvious at the *mask* (objects swell/merge, n 8→7, Dice 0.64). Feature table mean_area 191→462.
- **Noise:** first obvious at the *feature table* (n explodes 8→1483, mean_area 3 px). Mask looks like salt.

## Draft answers to the five report questions

1. **Useful vs trustworthy.** Direct VLM is more *fluent* (mentions round bodies on a dark field) but less trustworthy: naive prompt diagnoses; llava invented MRI/bone; gray gemma3 would not commit to nuclei. Numbers-first is more *useful for audit* (count, size, density tied to regionprops) and more trustworthy because every claim can be checked against the table. Trade-off: it cannot name “fluorescence / DAPI” because it never sees colour.

2. **U-Net vs Otsu.** On this clean synthetic modality, **yes**: val Dice 0.996 vs 0.976, and every val image favours U-Net (e.g. clustered **val_005**). **Otsu is better** on low-contrast **test_000** (0.928 vs 0.046) — a global threshold still separates the bimodal histogram; the CNN overfits the training contrast.

3. **Dice / IoU.** Dice = 2|P∩G|/(|P|+|G|); IoU = |P∩G|/|P∪G|. Val Dice 0.996 / IoU 0.992 means almost every predicted nucleus pixel matches the GT mask. Remaining mistakes: clustered/touching nuclei (semantic merge) and a handful of boundary pixels. Instance count is *not* what Dice measures — that is why dense test images look “perfect” on Dice and wrong on n_objects.

4. **Where the LLM can hallucinate.** (i) Task 1 pixels → free text (naive diagnoses; llava’s MRI). (ii) Task 2/4 qualitative labels (`density_class`, `quality_flag`, narrative) even when numbers are copied. (iii) Robustness: LLM said `quality_flag=ok` on a collapsed mask until we overwrote it. Mitigations: descriptive-not-diagnostic prompt; `"uncertain"` allowed; JSON schema; **Python owns n_objects, mean_area, quality_flag**; temperature 0.2 on production calls; never send the image to the numbers-first model. JSON as source of truth means the CSV remains checkable after a fluent-but-wrong paragraph.

5. **Clinical trust.** No part of this system is clinically trustworthy as-is: 112 synthetic images, no real stain/scanner shift, semantic not instance model, local VLMs hallucinate, none are cleared. Pixel Dice on in-distribution test is not a clinical claim. **Single change that would most improve trustworthiness:** keep the structured JSON as the record, but compute (don’t narrate) object counts from *instance* labels (watershed / instance U-Net / StarDist) **and** add a numeric `quality_flag` that blocks the narrative when area_fraction or n_objects leave a pre-registered range — plus real multi-site data and a human-in-the-loop review queue. A larger foundation model without those gates would not fix auditability.

## References (≤0.5 page)

- Ronneberger O, Fischer P, Brox T. U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI 2015.
- Otsu N. A threshold selection method from gray-level histograms. IEEE Trans. Syst. Man Cybern. 1979.
- Caicedo JC et al. Nucleus segmentation across imaging experiments: the 2018 Data Science Bowl. Nat Methods 2019.
- Dice LR. Measures of the amount of ecologic association between species. Ecology 1945.
- Liu H et al. Visual Instruction Tuning (LLaVA). NeurIPS 2023.
- Team G et al. Gemma 3 technical report. 2025.
- Ollama. llama3.2-vision / mllama support notes (v0.30+).
