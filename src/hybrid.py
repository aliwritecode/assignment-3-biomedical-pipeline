"""
Task 4 hybrid pipeline: U-Net mask → regionprops → structured JSON + narrative.

Numeric fields (n_objects, mean_area) are computed in Python and written back
over the LLM JSON so the CSV remains auditable even if the model hallucinates.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from . import ollama_client as oc
from .classical import density_class, region_table, summarise_table
from .prompts import HYBRID_LLM_PROMPT


def quality_from_features(computed: dict) -> str:
    """
    Deterministic quality flag from the numbers (source of truth).

    Catches collapsed masks (area_fraction ~ 1), empty fields, and speckle storms
    even if the LLM labels them 'ok'.
    """
    n = int(computed.get("n_objects") or 0)
    af = float(computed.get("area_fraction") or 0.0)
    if n == 0:
        return "review"
    if af >= 0.80:
        return "review"
    if n >= 200:
        return "review"
    return "ok"


def features_from_mask(gray: np.ndarray, binary: np.ndarray) -> tuple[pd.DataFrame, dict, str]:
    df = region_table(binary, intensity=gray)
    n = int(len(df))
    mean_area = float(df["area"].mean()) if n else 0.0
    mean_sol = float(df["solidity"].mean()) if n else None
    rec = {
        "n_objects": n,
        "mean_area": round(mean_area, 2),
        "median_area": round(float(df["area"].median()), 2) if n else 0.0,
        "mean_eccentricity": round(float(df["eccentricity"].mean()), 4) if n else None,
        "mean_solidity": round(float(df["solidity"].mean()), 4) if n else None,
        "mean_intensity": round(float(df["mean_intensity"].mean()), 4) if n else None,
        "area_fraction": round(float(binary.mean()), 4),
        "density_class_computed": density_class(n, mean_sol),
    }
    rec["quality_flag_computed"] = quality_from_features(rec)
    summary = summarise_table(df, extra={"area_fraction_mask": rec["area_fraction"]})
    return df, rec, summary


def llm_record_and_narrative(
    image_id: str,
    summary: str,
    computed: dict,
    model: str = config.LLM_MODEL,
    temperature: float = 0.2,
) -> tuple[dict, str, str]:
    """
    Ask the text LLM for a paragraph + JSON, then overwrite numeric keys
    with the computed values (JSON is the source of truth for counts/areas).
    """
    prompt = HYBRID_LLM_PROMPT.format(image_id=image_id, summary=summary)
    raw = oc.generate(prompt, model=model, temperature=temperature, json_mode=False)
    try:
        parsed = oc.extract_json(raw)
    except (ValueError, json.JSONDecodeError):
        parsed = {}
    narrative = oc.extract_paragraph(raw)
    record = {
        "image_id": image_id,
        "n_objects": computed["n_objects"],
        "mean_area": computed["mean_area"],
        "density_class": parsed.get("density_class", computed["density_class_computed"]),
        "quality_flag": computed.get("quality_flag_computed", "ok"),
        "density_class_computed": computed["density_class_computed"],
        "quality_flag_computed": computed.get("quality_flag_computed"),
        "area_fraction": computed["area_fraction"],
        "mean_eccentricity": computed["mean_eccentricity"],
        "mean_solidity": computed["mean_solidity"],
        "mean_intensity": computed["mean_intensity"],
        "llm_n_objects_raw": parsed.get("n_objects"),
        "llm_mean_area_raw": parsed.get("mean_area"),
        "llm_quality_flag_raw": parsed.get("quality_flag"),
        "narrative": narrative,
    }
    return record, narrative, raw


def aggregate_records(records: list[dict], csv_path: Path) -> pd.DataFrame:
    df = pd.DataFrame(records)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return df
