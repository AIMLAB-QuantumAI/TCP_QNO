"""Model variant syntax shared by the training script and its drivers.

QTRO and the qubit-native models take ``LxD`` (circuit depth x environment
dimension); the baselines take ``rankN``. The returned label names the
checkpoint and the kwargs reach the registry.
"""
from __future__ import annotations

DEPTH_MODELS = ("qtro", "qubit_hea", "qubit_dense", "qubit_ucr")
RANK_MODELS = ("cnto", "seqpred", "deeponet")
MODEL_NAMES = DEPTH_MODELS + RANK_MODELS


def parse_variant(model_name: str, text: str, cfg) -> tuple[str, dict]:
    """Apply a variant to ``cfg`` and return ``(label, model_kwargs)``."""
    if model_name in DEPTH_MODELS:
        try:
            layers, denv = (int(x) for x in text.lower().split("x"))
        except Exception as exc:
            raise ValueError(
                "a depth variant must look like 4x8 (layers x d_env)") from exc
        if layers < 1 or denv < 1:
            raise ValueError("layers and d_env must be positive")
        cfg.model.n_layers, cfg.model.d_env = layers, denv
        return f"L{layers}_dE{denv}", {}
    if model_name not in RANK_MODELS:
        raise ValueError(f"unknown model {model_name!r}; choose from {MODEL_NAMES}")
    try:
        rank = int(text.lower().removeprefix("rank"))
    except ValueError as exc:
        raise ValueError("a baseline variant must look like rank27") from exc
    if rank < 1:
        raise ValueError("rank must be positive")
    return f"rank{rank}", {"rank": rank}
