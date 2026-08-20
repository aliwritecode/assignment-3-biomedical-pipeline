"""
Thin client for a local Ollama instance (VLM + text LLM).

The generate API is used because it accepts base64 images for llama3.2-vision
and a `format: json` flag for structured records.
"""
from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Optional

import requests

from . import config


def encode_image(path: Path) -> str:
    """Return raw base64 (no data-URI prefix) as required by Ollama."""
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def generate(
    prompt: str,
    model: str,
    *,
    images: Optional[list[str]] = None,
    temperature: float = 0.2,
    json_mode: bool = False,
    keep_alive: str = "10m",
    timeout: int = 300,
) -> str:
    """
    Call POST /api/generate and return the response text.

    json_mode=True asks Ollama to constrain the decoder to a JSON object.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": keep_alive,
        "options": {"temperature": temperature},
    }
    if images:
        payload["images"] = images
    if json_mode:
        payload["format"] = "json"

    url = f"{config.OLLAMA_URL}/api/generate"
    r = requests.post(url, json=payload, timeout=timeout)
    if r.status_code >= 400:
        raise RuntimeError(f"Ollama {r.status_code} for model={model}: {r.text[:800]}")
    data = r.json()
    if data.get("error"):
        raise RuntimeError(f"Ollama error for model={model}: {data['error']}")
    return data.get("response", "")


def listed_models() -> set[str]:
    """Names as returned by /api/tags (both 'llava' and 'llava:latest')."""
    r = requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=30)
    r.raise_for_status()
    names: set[str] = set()
    for m in r.json().get("models", []):
        n = m.get("name", "")
        names.add(n)
        names.add(n.split(":")[0])
    return names


def resolve_vlm(preferred: str | None = None) -> str:
    """
    Return a locally available VLM that can actually load.

    Tries llama3.2-vision first (assignment spec). On this Ollama version the
    mllama architecture is not supported, so we fall back to gemma3:4b / llava.
    """
    preferred = preferred or config.VLM_MODEL
    candidates = [preferred, *config.VLM_FALLBACKS]
    installed = listed_models()
    last_err = None
    for name in candidates:
        if name not in installed and name.split(":")[0] not in installed:
            continue
        try:
            generate("Reply with the single word OK.", name, temperature=0.0, keep_alive="10m")
            if name != preferred:
                print(f"  note: '{preferred}' is not usable on this Ollama; using '{name}'")
            return name
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"  could not load {name}: {e}")
    raise RuntimeError(f"No working local VLM. Last error: {last_err}")


def unload(model: str) -> None:
    """Free VRAM/RAM occupied by a loaded Ollama model."""
    try:
        requests.post(
            f"{config.OLLAMA_URL}/api/generate",
            json={"model": model, "prompt": " ", "keep_alive": 0, "stream": False},
            timeout=30,
        )
    except requests.RequestException:
        pass


def extract_json(text: str) -> dict:
    """
    Parse a JSON object out of a model response.

    Handles raw JSON, markdown fences, and 'PARAGRAPH / JSON' two-block replies.
    """
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in model output:\n{text[:500]}")
    return json.loads(match.group(0))


def extract_paragraph(text: str) -> str:
    """Take the prose before the first '{' if present, else the whole string."""
    if "PARAGRAPH:" in text:
        body = text.split("PARAGRAPH:", 1)[1]
        if "JSON:" in body:
            body = body.split("JSON:", 1)[0]
        return body.strip().strip("*").strip()
    brace = text.find("{")
    if brace > 0:
        return text[:brace].strip().strip("*").strip()
    return text.strip().strip("*").strip()
