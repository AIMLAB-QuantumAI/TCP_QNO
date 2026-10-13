"""Validation metrics recorded during and after training, and the depth screen's ranking."""
from __future__ import annotations

import statistics

import torch

from qtro.eval.metrics import column_values, summarise_columns
from qtro.losses import propagator_powers

LONG_HORIZONS = ("h5", "h10", "h20")


@torch.no_grad()
def validation_metrics(model, data, grid, cfg, device: str,
                       batch_size: int = 4) -> dict:
    """One-step metrics against the fitting targets of the validation rows."""
    parts = []
    for start in range(0, len(data), batch_size):
        idx = torch.arange(start, min(start + batch_size, len(data)))
        batch = data.batch(idx, device)
        pred = model.propagator(batch["u"], batch["xi"])
        ref = batch["targets"][1]
        counts = batch["counts"][1]
        zero = torch.zeros_like(counts)
        parts.append(column_values(pred, ref, counts, zero, zero, grid,
                                   cfg.data.c_min))
    return summarise_columns(parts)


@torch.no_grad()
def validation_metrics_multihorizon(
        model, data, grid, cfg, device: str,
        horizons=(1,), batch_size: int = 4) -> dict[str, dict]:
    """Composed-horizon metrics against the disjoint replication half.

    `validation_metrics` compares the one-step propagator with the fitting
    targets and never composes. This raises the propagator to each requested
    power and scores it against `ref_h`, the half of the replications the
    targets were not fitted on, which is the quantity reported. The two are
    therefore not comparable by construction, which is why both are recorded.
    """
    requested = sorted({int(h) for h in horizons})
    parts: dict[int, list] = {h: [] for h in requested}
    for start in range(0, len(data), batch_size):
        idx = torch.arange(start, min(start + batch_size, len(data)))
        batch = data.batch(idx, device)
        powers = propagator_powers(
            model.propagator(batch["u"], batch["xi"]), requested)
        for h in requested:
            parts[h].append(column_values(
                powers[h], batch["refs"][h], batch["ref_counts"][h],
                batch["noise_fidelity"][h], batch["noise_tv"][h],
                grid, cfg.data.c_min))
    return {f"h{h}": summarise_columns(parts[h]) for h in requested}


def depth_screen_score(record: dict) -> float:
    """Mean of (fidelity - TV) over h in {5, 10, 20}, occupancy weighted; higher is better."""
    metrics = record["validation_multihorizon"]
    if not metrics:
        raise KeyError(
            f"record for {record.get('variant')} seed {record.get('seed')} has "
            f"no validation_multihorizon; it was trained without "
            f"--val-horizons and cannot be ranked")
    return statistics.fmean(
        metrics[h]["fidelity"]["occupancy_weighted"]
        - metrics[h]["tv"]["occupancy_weighted"]
        for h in LONG_HORIZONS)


def rank_depth_variants(records: list[dict]) -> list[dict]:
    """Variants ordered by their seed-averaged composed-horizon score."""
    by_variant: dict[str, list[float]] = {}
    for record in records:
        by_variant.setdefault(record["variant"], []).append(depth_screen_score(record))
    out = [{"variant": name, "score": statistics.fmean(scores),
            "n_seeds": len(scores), "scores": scores}
           for name, scores in by_variant.items()]
    return sorted(out, key=lambda r: r["score"], reverse=True)
