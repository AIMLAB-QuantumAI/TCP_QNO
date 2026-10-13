"""Distribution, observable, measurement-floor and validity metrics, per source column."""
from __future__ import annotations

import torch
from torch import Tensor

from qtro.data import fidelity, tv
from qtro.losses import propagator_powers


def normalised_axis_w1(pred: Tensor, ref: Tensor, bins: Tensor,
                       centres: Tensor) -> Tensor:
    """Per-source 1-Wasserstein distance after marginalising destinations.

    ``pred`` and ``ref`` use destination x source columns. ``bins`` maps
    each destination cell to one ordered physical-axis bin. The result is
    normalised by the represented axis range, so queue and congestion-window
    distances are both in [0, 1] for probability columns and remain directly
    comparable across grid resolutions.
    """
    n_bins = int(centres.numel())
    assignment = torch.nn.functional.one_hot(
        bins.to(device=pred.device), num_classes=n_bins
    ).to(dtype=pred.dtype).transpose(0, 1)
    pred_marginal = torch.einsum("kd,bds->bks", assignment, pred)
    ref_marginal = torch.einsum("kd,bds->bks", assignment, ref)
    if n_bins < 2:
        return torch.zeros(pred.shape[0], pred.shape[2], device=pred.device,
                           dtype=pred.dtype)
    x = centres.to(device=pred.device, dtype=pred.dtype)
    widths = x[1:] - x[:-1]
    cdf_gap = (pred_marginal - ref_marginal).cumsum(dim=1)[:, :-1].abs()
    raw = (cdf_gap * widths.view(1, -1, 1)).sum(dim=1)
    return raw / (x[-1] - x[0]).clamp_min(torch.finfo(pred.dtype).eps)


def column_values(pred: Tensor, ref: Tensor, ref_counts: Tensor,
                  noise_fidelity: Tensor, noise_tv: Tensor, grid,
                  c_min: int) -> dict[str, Tensor]:
    """Per-source-column values before the three aggregations."""
    pred_col = pred.transpose(1, 2)
    ref_col = ref.transpose(1, 2)
    f = fidelity(pred_col, ref_col, eps=0.0)
    distance = tv(pred_col, ref_col)
    q = grid.q_rep.to(pred)
    tail = grid.tail_mask.to(pred.device)
    mean_q_error = (q @ pred - q @ ref).abs()
    overflow_error = (tail.to(pred.dtype) @ pred
                      - tail.to(pred.dtype) @ ref).abs()
    w_bin, q_bin, _ = grid.axis_bins()
    w_centres = 0.5 * (grid.w_edges[:-1] + grid.w_edges[1:])
    q_centres = 0.5 * (grid.q_edges[:-1] + grid.q_edges[1:])
    queue_w1 = normalised_axis_w1(pred, ref, q_bin, q_centres)
    cwnd_w1 = normalised_axis_w1(pred, ref, w_bin, w_centres)
    ref_mass = ref.sum(dim=1)
    valid = ((ref_counts >= c_min) & torch.isfinite(noise_fidelity)
             & torch.isfinite(noise_tv) & (ref_mass > 0.99))
    source_tail = tail.unsqueeze(0).expand_as(valid)
    return {
        "fidelity": f,
        "tv": distance,
        "mean_queue_abs_error": mean_q_error,
        "overflow_abs_error": overflow_error,
        "queue_w1": queue_w1,
        "cwnd_w1": cwnd_w1,
        "noise_fidelity": noise_fidelity,
        "noise_tv": noise_tv,
        "valid": valid,
        "occupancy": ref_counts,
        "source_tail": source_tail,
    }


def _aggregate(values: Tensor, valid: Tensor, occupancy: Tensor,
               source_tail: Tensor) -> dict:
    values = values.reshape(-1).float()
    valid = valid.reshape(-1) & torch.isfinite(values.reshape(-1))
    occupancy = occupancy.reshape(-1).float()
    source_tail = source_tail.reshape(-1)

    def mean(mask: Tensor, weights: Tensor | None = None) -> float:
        mask = valid & mask
        if not bool(mask.any()):
            return float("nan")
        if weights is None:
            return float(values[mask].mean())
        w = weights[mask]
        return float((values[mask] * w).sum() / w.sum().clamp_min(1.0e-30))

    all_columns = torch.ones_like(valid)
    return {
        "uniform": mean(all_columns),
        "occupancy_weighted": mean(all_columns, occupancy),
        "tail_only": mean(source_tail),
        "n_columns": int(valid.sum()),
        "n_tail_columns": int((valid & source_tail).sum()),
    }


def summarise_columns(parts: list[dict[str, Tensor]]) -> dict:
    """Combine evaluation batches without averaging their averages."""
    if not parts:
        raise ValueError("no metric batches to summarise")
    keys = parts[0]
    cat = {k: torch.cat([p[k].detach().cpu() for p in parts], dim=0)
           for k in keys}
    common = (cat["valid"], cat["occupancy"], cat["source_tail"])
    return {
        name: _aggregate(cat[name], *common)
        for name in ("fidelity", "tv", "mean_queue_abs_error",
                     "overflow_abs_error", "queue_w1", "cwnd_w1",
                     "noise_fidelity", "noise_tv")
    }


def configuration_column_aggregates(
        values: dict[str, Tensor],
        metrics: tuple[str, ...] | None = None,
) -> dict[str, dict[str, Tensor]]:
    """Aggregate source-column scores separately for each condition.

    :func:`summarise_columns` intentionally pools all valid columns in a
    split, which is appropriate for the headline metric but discards the
    sampling unit a condition-paired uncertainty analysis needs. This keeps
    one uniform / occupancy-weighted / tail-only value per condition. A
    missing class (for example, no reliable tail-source column) is NaN rather
    than a made-up zero and is ignored by the paired analysis.

    The occupancy aggregation is normalised *within* each condition, so a
    later mean over conditions estimates performance for a uniformly drawn
    condition rather than letting conditions with more observed transitions
    dominate the uncertainty calculation.
    """
    if metrics is None:
        metrics = tuple(name for name in (
            "fidelity", "tv", "overflow_abs_error", "queue_w1", "cwnd_w1"
        ) if name in values)
    valid = values["valid"].bool()
    occupancy = values["occupancy"].float()
    source_tail = values["source_tail"].bool()

    def row_mean(score: Tensor, mask: Tensor,
                 weight: Tensor | None = None) -> Tensor:
        mask = valid & mask & torch.isfinite(score)
        if weight is None:
            w = mask.to(score.dtype)
        else:
            w = torch.where(mask, weight.to(score.dtype),
                            torch.zeros_like(score))
        denominator = w.sum(dim=1)
        numerator = torch.where(mask, score, torch.zeros_like(score)).mul(w).sum(1)
        nan = torch.full_like(numerator, float("nan"))
        return torch.where(denominator > 0, numerator / denominator.clamp_min(1e-30),
                           nan)

    everywhere = torch.ones_like(valid)
    out = {}
    for name in metrics:
        if name not in values:
            raise KeyError(f"column metric {name!r} is unavailable")
        score = values[name]
        out[name] = {
            "uniform": row_mean(score, everywhere),
            "occupancy_weighted": row_mean(score, everywhere, occupancy),
            "tail_only": row_mean(score, source_tail),
        }
    return out


def structural_validity(P: Tensor, horizons: list[int]) -> dict[str, dict]:
    """Raw finite-precision validity of P^h, with no projection or repair."""
    powers = propagator_powers(P, horizons)
    out = {}
    for h in sorted(horizons):
        Ph = powers[h]
        has_nonfinite = not bool(torch.isfinite(Ph).all())
        out[str(h)] = {
            "max_negative_mass": (float("inf") if has_nonfinite else
                                  float((-Ph).clamp_min(0).max())),
            "max_column_mass_deviation": (float("inf") if has_nonfinite else
                                          float((Ph.sum(dim=1) - 1).abs().max())),
            "has_nonfinite": has_nonfinite,
        }
    return out


def configuration_observables(pred: Tensor, ref: Tensor, ref_counts: Tensor,
                              grid) -> dict[str, Tensor]:
    """One future mean queue and overflow probability per condition."""
    source = ref_counts / ref_counts.sum(dim=1, keepdim=True).clamp_min(1.0)
    q = grid.q_rep.to(pred)
    tail = grid.tail_mask.to(pred.device, dtype=pred.dtype)
    return {
        "pred_mean_queue": torch.einsum("bi,bij,bj->b", q.expand(pred.shape[0], -1),
                                        pred, source),
        "ref_mean_queue": torch.einsum("bi,bij,bj->b", q.expand(ref.shape[0], -1),
                                       ref, source),
        "pred_overflow": torch.einsum("bi,bij,bj->b", tail.expand(pred.shape[0], -1),
                                      pred, source),
        "ref_overflow": torch.einsum("bi,bij,bj->b", tail.expand(ref.shape[0], -1),
                                     ref, source),
    }
