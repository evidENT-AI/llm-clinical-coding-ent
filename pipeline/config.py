#!/usr/bin/env python3
"""Model settings, decoding parameters and paths for the coding pipeline.

Credentials and paths are read from the environment (see `.env.example`).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[1]

TEMPERATURE = 0
SEED = 42

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
LOCAL_BASE_URL = os.environ.get("LOCAL_BASE_URL", "http://localhost:8000/v1")
LOCAL_API_KEY = os.environ.get("LOCAL_API_KEY", "EMPTY")
AZURE_API_VERSION = os.environ.get("AZURE_API_VERSION", "2024-12-01-preview")
SNOMED_SERVER_URL = os.environ.get("SNOMED_SERVER_URL", "http://localhost:8081")

NOTES_DIR = Path(os.environ.get(
    "NOTES_DIR", REPO_ROOT / "data" / "synthetic" / "episodes"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", REPO_ROOT / "results" / "runs"))
EXAMPLE_NOTE_PATH = Path(os.environ.get(
    "EXAMPLE_NOTE_PATH", REPO_ROOT / "data" / "synthetic" / "example_note.json"))
EXAMPLE_ANNOTATION_PATH = Path(os.environ.get(
    "EXAMPLE_ANNOTATION_PATH",
    REPO_ROOT / "data" / "synthetic" / "example_annotation.json"))
CLASSIFICATION_LOOKUP = Path(os.environ.get(
    "CLASSIFICATION_LOOKUP", REPO_ROOT / "data" / "snomed_to_icd_opcs.json"))


def _azure(prefix):
    """Per-model Azure OpenAI settings; each model is its own Azure resource."""
    endpoint = os.environ.get(f"{prefix}_ENDPOINT")
    return {
        "provider": "azure" if endpoint else "openai",
        "azure_endpoint": endpoint,
        "azure_deployment": os.environ.get(f"{prefix}_DEPLOYMENT"),
        "azure_api_key": os.environ.get(f"{prefix}_API_KEY"),
        "api_version": AZURE_API_VERSION,
    }


# Keys are used on the command line; `model` is the served or API model name.
MODELS = {
    "LLAMA31_8B": {
        "model": "llama-3.1-8b-instruct",
        "provider": "local",
        "max_tokens": 8192,
        "price_per_mtok": (0.0, 0.0),
    },
    "LLAMA4_SCOUT": {
        "model": "llama-4-scout",
        "provider": "local",
        "max_tokens": 8192,
        "price_per_mtok": (0.0, 0.0),
    },
    "GPT4O": {
        "model": "gpt-4o-2024-11-20",
        "max_tokens": 8192,
        "price_per_mtok": (2.50, 10.00),
        **_azure("GPT4O"),
    },
    "GPT5": {
        "model": "gpt-5-2025-08-07",
        # Reasoning and visible output share this budget.
        "max_tokens": 32768,
        "token_param": "max_completion_tokens",
        "supports_temperature": False,
        "supports_seed": False,
        "reasoning_effort": "high",
        "price_per_mtok": (1.25, 10.00),
        **_azure("GPT5"),
    },
}
