"""Shared Hugging Face helpers for the standalone data scripts."""

from __future__ import annotations

import os
from pathlib import Path


def resolve_hf_token(explicit: str | None = None) -> str | None:
    """Resolve a Hugging Face token from an explicit value, $HF_TOKEN, or disk.

    The disk fallbacks mirror huggingface_hub's own token locations
    (``~/.hf_token`` and the hub cache token file), so scripts can run with a
    cached login without exporting anything.
    """
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip()
    value = os.environ.get("HF_TOKEN")
    if value:
        return value.strip()
    for path in (Path.home() / ".hf_token", Path.home() / ".cache/huggingface/token"):
        if path.is_file():
            return path.read_text(encoding="utf-8").strip()
    return None


def hf_file_url(repo: str, path: str) -> str:
    """Resolve a file URL inside a Hugging Face dataset repo."""
    return f"https://huggingface.co/datasets/{repo}/resolve/main/{path}"
