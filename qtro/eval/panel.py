"""Score models condition by condition on an ns-3 dataset.

Every evaluation script pairs models by seed and condition, so all of them need
the same per-condition scores computed by the same code on the same columns.
The scripts differ only in which metrics they keep and how they pair them.
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from qtro.checkpoint import load_checkpoint
from qtro.data import SweepData
from qtro.eval.metrics import column_values, configuration_column_aggregates
from qtro.losses import propagator_powers

Kernels = Callable[[dict], dict[int, torch.Tensor]]
Scores = dict[int, dict[str, dict[str, np.ndarray]]]


def score_kernels(kernels_for: Kernels, panel: SweepData, grid, c_min: int,
                  device: str, horizons: tuple[int, ...],
                  metrics: tuple[str, ...], batch_size: int = 4) -> Scores:
    """``{h: {metric: {aggregation: (n_conditions,) array}}}``.

    ``kernels_for(batch)`` returns ``{h: (B, M, M)}``. Analytic and learned
    models route through here, so both are scored by identical code on
    identical columns.
    """
    parts: dict[int, list] = {h: [] for h in horizons}
    with torch.no_grad():
        for start in range(0, len(panel), batch_size):
            idx = torch.arange(start, min(start + batch_size, len(panel)))
            batch = panel.batch(idx, device)
            kernels = kernels_for(batch)
            for h in horizons:
                parts[h].append(column_values(
                    kernels[h], batch["refs"][h], batch["ref_counts"][h],
                    batch["noise_fidelity"][h], batch["noise_tv"][h],
                    grid, c_min))
    out: Scores = {}
    for h in horizons:
        joined = {k: torch.cat([p[k].detach().cpu() for p in parts[h]], dim=0)
                  for k in parts[h][0]}
        agg = configuration_column_aggregates(joined, metrics)
        out[h] = {m: {k: agg[m][k].numpy() for k in agg[m]} for m in metrics}
    return out


def score_checkpoint(path, panel: SweepData, device: str, horizons, metrics,
                     batch_size: int = 4) -> Scores:
    model, grid, cfg, _ = load_checkpoint(path, device)
    model.eval()
    horizons = tuple(horizons)

    def kernels_for(batch: dict) -> dict[int, torch.Tensor]:
        return propagator_powers(model.propagator(batch["u"], batch["xi"]), sorted(horizons))

    scores = score_kernels(kernels_for, panel, grid, cfg.data.c_min, device, horizons,
                           tuple(metrics), batch_size)
    # The model goes out of scope on return; releasing the cache keeps a run
    # over many checkpoints from fragmenting GPU memory.
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return scores


def resolve_checkpoint(directory, record: dict) -> Path:
    """The checkpoint a training record names.

    The training script always writes a run's checkpoints under
    ``<directory>/checkpoints/``, so that file is taken first; a run read
    from another directory, or moved, still scores its own weights even when
    something else now lives at the path the manifest recorded. The recorded
    path (as typed on the training command line, relative to that working
    directory) is the fallback.
    """
    name = str(record["checkpoint"]).replace("\\", "/").rsplit("/", 1)[-1]
    own = Path(directory) / "checkpoints" / name
    return own if own.exists() else Path(record["checkpoint"])


def score_run_directory(directory, panel, device, horizons, metrics,
                        seeds=None) -> dict[int, Scores]:
    """One entry per checkpoint listed in the directory's ``train_*.json``, keyed by seed.

    A directory that holds several variants (one manifest each) has the same
    seed under more than one record; scoring it would silently keep one
    variant, so it is refused.
    """
    scores: dict[int, Scores] = {}
    for path in sorted(Path(directory).glob("train_*.json")):
        for record in json.loads(path.read_text(encoding="utf-8")):
            checkpoint = resolve_checkpoint(directory, record)
            seed = int(record["seed"])
            if not checkpoint.exists() or (seeds is not None and seed not in seeds):
                continue
            if seed in scores:
                raise ValueError(
                    f"seed {seed} appears in more than one training record under "
                    f"{directory}; score one variant's run directory at a time")
            scores[seed] = score_checkpoint(checkpoint, panel, device, horizons, metrics)
            print(f"  scored {checkpoint.name}", flush=True)
    return scores


def score_checkpoint_directory(directory, panel, device, horizons, metrics,
                               seeds=None) -> dict[int, Scores]:
    """One entry per ``<directory>/checkpoints/*.pt``, the seed read off the file name."""
    scores: dict[int, Scores] = {}
    for path in sorted(Path(directory).glob("checkpoints/*.pt")):
        found = re.search(r"seed(\d+)", path.stem)
        seed = int(found.group(1)) if found else len(scores)
        if seeds is not None and seed not in seeds:
            continue
        scores[seed] = score_checkpoint(path, panel, device, horizons, metrics)
    return scores


def split_half_floor(panel: SweepData, horizons, c_min: int) -> dict[int, tuple[float, float]]:
    """Mean split-half fidelity and TV over the columns a model is scored on."""
    floor = {}
    for h in horizons:
        valid = panel.ref_counts[h] >= c_min
        floor[h] = (float(panel.noise_fidelity[h][valid].mean()),
                    float(panel.noise_tv[h][valid].mean()))
    return floor


def parse_model_specs(text: str) -> dict[str, str]:
    """``name=path,name=path`` in the order given."""
    specs: dict[str, str] = {}
    for entry in text.split(","):
        name, sep, path = entry.partition("=")
        if not name.strip() or not sep or not path.strip():
            raise ValueError(f"--models entry must be name=path, got {entry!r}")
        specs[name.strip()] = path.strip()
    return specs
