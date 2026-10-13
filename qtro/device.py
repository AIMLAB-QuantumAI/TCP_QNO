"""Device selection shared by every script."""
from __future__ import annotations

import torch


def resolve_device(name: str) -> str:
    """Return ``name`` unless it is ``auto``, which picks CUDA when available."""
    if name != "auto":
        return name
    return "cuda" if torch.cuda.is_available() else "cpu"
