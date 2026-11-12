"""Hugging Face cache helpers for suite scripts."""

from __future__ import annotations

import os
from pathlib import Path


def hf_cache_dir() -> str:
    """Return the directory expected by Transformers' ``cache_dir=`` argument."""
    if os.environ.get("HF_HUB_CACHE"):
        return os.environ["HF_HUB_CACHE"]
    hf_home = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
    return str(hf_home / "hub")
