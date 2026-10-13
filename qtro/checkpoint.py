"""Portable checkpoints: resolved config, model name and kwargs, weights, metadata.

Checkpoints written by the research code before the paper's names were adopted
load unchanged: their model names are mapped, and the configuration keys of
options this release removed are dropped when they carried the inert value.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from qtro.config import config_from_dict
from qtro.grid import Grid
from qtro.registry import build_model, canonical_name

FORMAT_VERSION = 1


class _AnyValue:
    """Marker: the key was a hyperparameter of a removed feature; any value is inert."""


class _Negative:
    """Marker: the feature was off whenever the value was negative."""


ANY, NEGATIVE = _AnyValue(), _Negative()

# Removed configuration keys and the value an older checkpoint may carry
# while still describing a model this code can build.
REMOVED_KEYS = {
    "net": {"n_flows": ANY, "p_err": ANY},
    "aqm": {"p_max": ANY, "min_th": ANY},
    "sim": {"n_snapshots": ANY, "warmup_s": ANY, "n_replications": ANY,
            "rto_s": ANY, "seed": ANY},
    "model": {
        "n_transport_experts": 1, "transport_gate_init": ANY,
        "transport_residual_mode": "none", "transport_move_init": ANY,
        "reset_init_probability": NEGATIVE, "reset_neighborhood_radius": 0,
        "base_channel_mode": "none", "base_channel_radius": ANY,
        "base_channel_allow_phase": ANY, "base_channel_identity_init": ANY,
        "base_channel_init": ANY, "base_gate_init": ANY,
        "local_channel_mode": "none", "local_init_probability": ANY,
        "local_max_probability": ANY, "local_gate_init": ANY,
        "qubit_topology": "chain",
    },
    "data": {"horizons_eval": ANY},
    "train": {
        "lambda_tv": 0.0, "lambda_entropy": 0.0, "entropy_bias_correct": ANY,
        "macro_balanced_distribution": False, "balance_memory_modes": False,
        "lambda_action_contrast": 0.0, "action_contrast_risks": ANY,
        "action_rank_weight": ANY, "action_rank_margin": ANY,
        "action_min_signal": ANY, "grad_checkpoint": ANY,
    },
}


def _inert(allowed, value) -> bool:
    if allowed is ANY:
        return True
    if allowed is NEGATIVE:
        return value < 0
    return value == allowed


def upgrade_config_dict(mapping: dict) -> dict:
    """Drop removed keys that carried their inert value; refuse any other."""
    out = {}
    for section, values in mapping.items():
        removed = REMOVED_KEYS.get(section, {})
        kept = {}
        for key, value in values.items():
            if key not in removed:
                kept[key] = value
            elif not _inert(removed[key], value):
                raise ValueError(
                    f"checkpoint uses the removed option {section}.{key}={value!r}; "
                    "this release builds only the models the paper reports")
        out[section] = kept
    return out


def save_checkpoint(path, model, cfg, *, model_name: str,
                    model_kwargs: dict | None = None, metadata: dict | None = None,
                    training: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": FORMAT_VERSION,
        "model_name": canonical_name(model_name),
        "model_kwargs": dict(model_kwargs or {}),
        "config": asdict(cfg),
        "state_dict": model.state_dict(),
        "metadata": dict(metadata or {}),
        "training": dict(training or {}),
    }
    torch.save(payload, path)
    return path


def load_payload(path, device: str = "cpu") -> dict:
    """The stored payload with names and configuration brought up to date."""
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"unsupported checkpoint format in {path}")
    payload["model_name"] = canonical_name(payload["model_name"])
    payload["config"] = upgrade_config_dict(payload["config"])
    return payload


def load_checkpoint(path, device: str = "cpu"):
    payload = load_payload(path, device)
    cfg = config_from_dict(payload["config"])
    cfg.sim.device = device
    grid = Grid(cfg.grid, cfg.net, device=device)
    model = build_model(payload["model_name"], grid, cfg, payload.get("model_kwargs"))
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model, grid, cfg, payload
