"""
Paths, model names, and training defaults.

All large language models are queried locally through Ollama.
None of the models are clinically cleared; outputs are educational only.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT / "data" / "nuclei_dataset"
DATA_PROCESSED = ROOT / "data" / "processed"

OUTPUTS = ROOT / "outputs"
FIG_DIR = OUTPUTS / "figures"
JSON_DIR = OUTPUTS / "json"
CSV_DIR = OUTPUTS / "csv"
MODEL_DIR = OUTPUTS / "models"
PROMPT_DIR = OUTPUTS / "prompts"

IMG_SIZE = 256
SEED = 42

# Local Ollama endpoints / models (must already be pulled).
# Assignment spec: llama3.2-vision. Ollama ≥0.30 dropped the mllama backend
# (known issue: "unknown model architecture: 'mllama'"), so we try that tag
# first and fall back to a working local VLM on this machine.
OLLAMA_URL = "http://localhost:11434"
VLM_MODEL = "llama3.2-vision"
VLM_FALLBACKS = ["gemma3:4b", "llava"]  # extra-credit second model: llava
LLM_MODEL = "llama3.1:8b"  # text-only numbers-first / hybrid narration

# Small U-Net (Ronneberger-style, scaled down for the mini-dataset)
UNET_BASE = 32
EPOCHS = 30
BATCH_SIZE = 8
LR = 1e-3
MIN_OBJECT_AREA = 25  # px; drops Otsu speckles smaller than a nucleus

# Density bins aligned with make_dataset.py
SPARSE_MAX = 14
NORMAL_MAX = 40

REPRESENTATIVE_ID = "train_001"  # normal density, used for Task 1/2 demos


def ensure_dirs() -> None:
    """Create output folders if they do not exist."""
    for d in (DATA_PROCESSED, OUTPUTS, FIG_DIR, JSON_DIR, CSV_DIR, MODEL_DIR, PROMPT_DIR):
        d.mkdir(parents=True, exist_ok=True)


def get_device():
    """Prefer Apple MPS, then CUDA, else CPU."""
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
