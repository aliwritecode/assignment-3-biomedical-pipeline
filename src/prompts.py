"""
Optimised prompts used at every LLM / VLM step.

These strings are the auditable 'prompt record' required by the assignment.
They are also written to outputs/prompts/optimised_prompts.md at runtime.
"""

# ---------------------------------------------------------------------------
# Task 1 — direct visual description
# ---------------------------------------------------------------------------

VLM_NAIVE = "Describe this medical image."

VLM_OPTIMISED = """You are a descriptive biomedical imaging assistant, NOT a clinician.
Describe visual appearance only. Do not diagnose, grade disease, name a patient, or recommend treatment.

Look at the image and fill the JSON schema below. If a field cannot be determined from the pixels, write "uncertain" rather than guessing. Prefer under-claiming over hallucination.

Return JSON only — no markdown fences, no commentary, no extra keys:
{
  "modality": string,
  "tissue_type": string,
  "notable_features": [string],
  "image_quality": string
}

Allowed image_quality values: "good", "acceptable", "poor", "uncertain".
This is an educational exercise. The model is not clinically cleared."""

# ---------------------------------------------------------------------------
# Task 2 — numbers-first (the model never sees the image)
# ---------------------------------------------------------------------------

CLASSICAL_LLM_PROMPT = """You are a biomedical image-analysis assistant. You have NOT seen the image.
You are given quantitative region properties extracted by classical image processing
(Otsu thresholding, morphological cleanup, connected-component labelling, regionprops).

Rules:
- Use only the numbers provided. Do not invent counts, areas, or intensities.
- If a quantity is missing or contradictory, write "uncertain".
- Do not diagnose. Describe the field, not a patient.

Write two blocks in this exact order:

PARAGRAPH:
<one paragraph: object count, size, shape regularity, intensity, and any quality caveats>

JSON:
{{
  "n_objects": <int>,
  "density_class": "<sparse|normal|dense|clustered|uncertain>",
  "shape_regularity": "<regular|mixed|irregular|uncertain>",
  "quality_flag": "<ok|review|uncertain>"
}}

Measured features:
{summary}
"""

# ---------------------------------------------------------------------------
# Task 4 — hybrid pipeline (U-Net mask → regionprops → LLM)
# ---------------------------------------------------------------------------

HYBRID_LLM_PROMPT = """You are a biomedical image-analysis assistant. You have NOT seen the image.
You receive (a) an image_id and (b) quantitative features computed from a U-Net
segmentation mask via connected-component regionprops.

Rules:
- Treat the measured numbers as the source of truth. Copy n_objects and mean_area exactly.
- density_class and quality_flag may be inferred from those numbers, or "uncertain".
- Do not diagnose. Educational use only; the model is not clinically cleared.

Write two blocks in this exact order:

PARAGRAPH:
<one paragraph narrative of the segmented field>

JSON:
{{
  "image_id": "{image_id}",
  "n_objects": <int>,
  "mean_area": <float>,
  "density_class": "<sparse|normal|dense|clustered|uncertain>",
  "quality_flag": "<ok|review|uncertain>"
}}

Measured features:
{summary}
"""


def dump_prompt_record(path) -> None:
    """Write every prompt to disk so they can be copied into the report."""
    text = (
        "# Optimised prompts (Assignment 3)\n\n"
        "All models run locally via Ollama. Educational use only.\n\n"
        "## Task 1 — naive VLM prompt\n\n```\n"
        + VLM_NAIVE
        + "\n```\n\n## Task 1 — optimised structured VLM prompt\n\n```\n"
        + VLM_OPTIMISED
        + "\n```\n\n## Task 2 — numbers-first LLM prompt\n\n```\n"
        + CLASSICAL_LLM_PROMPT
        + "\n```\n\n## Task 4 — hybrid LLM prompt\n\n```\n"
        + HYBRID_LLM_PROMPT
        + "\n```\n"
    )
    path.write_text(text, encoding="utf-8")
