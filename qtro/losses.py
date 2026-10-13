"""Training objective, applied column by column.

The dataset conditions on the source cell, so every term scores one
destination distribution per column: a fidelity term, an observable term on
the mean queue and the overflow probability, a topology-aware Wasserstein term
on the queue and window marginals, a multi-threshold tail term, and an L2
penalty on the parameters.

There is deliberately no penalty on non-negativity, normalisation or
diagonality. Validity is structural, and a penalty term would have the
implementation quietly concede what the construction asserts.
"""
from __future__ import annotations

import torch
from torch import Tensor

from qtro.data import fidelity


def column_weights(counts: Tensor, c_min: int, c_cap_quantile: float) -> Tensor:
    """Per-column loss weights from observed sample counts.

    Both extremes fail, so neither is used. Weighting purely by occupancy
    buries the tail columns, which carry under one percent of the mass at the
    calibrated operating point and are what the risk readout depends on.
    Weighting uniformly gives a column estimated from one sample the same say
    as one estimated from a quarter of a million, so estimation noise reads
    as a property of the dynamics.

    So: drop columns below `c_min` outright, and cap the rest at the
    `c_cap_quantile` quantile of the observed counts. The cap is a quantile
    rather than an absolute count because the count distribution shifts with
    the condition, and a constant would mean something different in each.

    `counts` is (B, M); the result is (B, M), each row summing to 1. A row with
    no column above `c_min` returns all zeros rather than NaN; that condition
    then contributes nothing to the loss, and `dropped_columns` counts it.
    """
    counts = counts.float()
    keep = counts >= c_min
    # Quantile over the columns that survive, per condition. quantile()
    # cannot take a ragged input, so it runs per row.
    caps = []
    for b in range(counts.shape[0]):
        row = counts[b][keep[b]]
        caps.append(torch.quantile(row, c_cap_quantile) if row.numel()
                    else counts.new_tensor(1.0))
    cap = torch.stack(caps).unsqueeze(1)
    w = torch.minimum(counts, cap) * keep
    return w / w.sum(1, keepdim=True).clamp(min=1e-30)


def balanced_column_weights(counts: Tensor, tail_mask: Tensor, c_min: int,
                            c_cap_quantile: float, *, occupancy: float,
                            uniform: float, tail: float) -> Tensor:
    """Mix reliable occupancy, uniform, and tail-source column weights.

    Each requested component is normalised per condition before mixing.
    The result is normalised once more because a condition can contain no
    reliable tail-source column; in that case its available occupancy and
    uniform components retain unit total weight rather than silently
    down-weighting the whole condition. A row with no reliable column remains
    all zero.
    """
    coefficients = counts.new_tensor([occupancy, uniform, tail]).float()
    if not bool(torch.isfinite(coefficients).all()) or \
            bool((coefficients < 0).any()) or float(coefficients.sum()) <= 0:
        raise ValueError(
            "distribution source weights must be finite, non-negative, and "
            "have a positive sum")

    counts = counts.float()
    valid = counts >= c_min
    occ = column_weights(counts, c_min, c_cap_quantile)
    uni = valid.float()
    uni = uni / uni.sum(1, keepdim=True).clamp_min(1.0)
    source_tail = tail_mask.to(device=counts.device, dtype=torch.bool)
    tail_valid = valid & source_tail.unsqueeze(0)
    tail_w = tail_valid.float()
    tail_w = tail_w / tail_w.sum(1, keepdim=True).clamp_min(1.0)

    mixed = occupancy * occ + uniform * uni + tail * tail_w
    return mixed / mixed.sum(1, keepdim=True).clamp_min(1.0e-30)


def dropped_columns(counts: Tensor, c_min: int) -> dict:
    """What `column_weights` discards, so the run can report it.

    Silent truncation reads as full coverage. This returns the number of
    columns below threshold and the share of observed transitions they carry.
    """
    counts = counts.float()
    dropped = counts < c_min
    total = counts.sum().clamp(min=1.0)
    return {
        "n_dropped": int(dropped.sum()),
        "n_columns": int(counts.numel()),
        "occupancy_share_dropped": float(counts[dropped].sum() / total),
    }


def propagator_powers(P: Tensor, horizons: list[int]) -> dict[int, Tensor]:
    """Return the requested powers of ``P`` using a cached addition chain.

    Recursive squaring makes the common ``[1, 5, 10]`` set of horizons cost
    four matrix multiplications where a sequential prefix needs nine. Every
    returned matrix is the exact algebraic power of ``P``.
    """
    requested = sorted(set(int(h) for h in horizons))
    if not requested or requested[0] < 1:
        raise ValueError("horizons must contain positive integers")
    cache: dict[int, Tensor] = {1: P}

    def power(horizon: int) -> Tensor:
        if horizon in cache:
            return cache[horizon]
        if horizon % 2 == 0:
            half = power(horizon // 2)
            value = torch.bmm(half, half)
        else:
            value = torch.bmm(P, power(horizon - 1))
        cache[horizon] = value
        return value

    return {h: power(h) for h in requested}


def _horizon_value(value: Tensor | dict[int, Tensor], horizon: int) -> Tensor:
    """Select a horizon-specific tensor, or pass a shared one through."""
    return value[horizon] if isinstance(value, dict) else value


def distribution_loss(powers: dict[int, Tensor], targets: dict[int, Tensor],
                      weights: Tensor | dict[int, Tensor],
                      omega: dict[int, float],
                      eps: float) -> Tensor:
    """Weighted 1 - fidelity, per column, per horizon.

    `eps` must be the training value. Measurement uses eps = 0: the
    Bhattacharyya coefficient has an unbounded derivative where one argument
    vanishes and the other does not, so a backpropagated call needs the
    guard, while any eps large enough to bound that gradient also biases the
    forward value upward.
    """
    total = None
    for h, Ph in powers.items():
        # (B, M) per-column fidelity: transpose so the last axis, which
        # fidelity() reduces over, is the destination cell.
        f = fidelity(Ph.transpose(1, 2), targets[h].transpose(1, 2), eps=eps)
        term = omega[h] * (_horizon_value(weights, h) * (1.0 - f)).sum()
        total = term if total is None else total + term
    return total


def observable_loss(powers: dict[int, Tensor], targets: dict[int, Tensor],
                    weights: Tensor | dict[int, Tensor],
                    omega: dict[int, float], q_rep: Tensor,
                    tail_mask: Tensor, alpha_tail: float) -> Tensor:
    """Mean-queue error plus alpha_tail times the overflow error.

    Both observables are linear functionals of the predicted column, so this
    is an absolute error between two scalars per column. The mean queue is
    normalised by its own scale before being added to a probability error;
    otherwise the queue term, in packets, would dwarf the tail term, in
    probability, and alpha_tail would be tuning against an accident of units.
    """
    q_scale = q_rep.max().clamp(min=1e-12)
    total = None
    for h, Ph in powers.items():
        pred, ref = Ph, targets[h]
        # Column j of P is a distribution over destinations, so the observable
        # of column j contracts over the destination (row) axis.
        q_err = ((q_rep @ pred - q_rep @ ref).abs()) / q_scale
        t_err = (tail_mask.float() @ pred - tail_mask.float() @ ref).abs()
        term = omega[h] * (
            _horizon_value(weights, h) * (q_err + alpha_tail * t_err)
        ).sum()
        total = term if total is None else total + term
    return total


def _axis_marginal(matrix: Tensor, bins: Tensor, n_bins: int) -> Tensor:
    """Marginalise destination states onto one physical grid axis.

    ``matrix`` is ``(B, destination, source)`` and the result is
    ``(B, axis_bin, source)``. A fixed projection matrix keeps the operation
    differentiable while avoiding a Python loop over destination cells.
    """
    projector = torch.nn.functional.one_hot(
        bins.long(), num_classes=n_bins
    ).to(device=matrix.device, dtype=matrix.dtype).T
    return torch.einsum("km,bms->bks", projector, matrix)


def _axis_positions(values: Tensor, bins: Tensor, n_bins: int, *,
                    logarithmic: bool) -> Tensor:
    values = values.to(dtype=torch.float32)
    if logarithmic:
        values = values.clamp_min(1.0e-12).log()
    total = values.new_zeros(n_bins)
    count = values.new_zeros(n_bins)
    total.scatter_add_(0, bins.long(), values)
    count.scatter_add_(0, bins.long(), torch.ones_like(values))
    return total / count.clamp_min(1.0)


def marginal_wasserstein_loss(
        powers: dict[int, Tensor], targets: dict[int, Tensor],
        weights: Tensor | dict[int, Tensor], omega: dict[int, float], grid,
        queue_weight: float = 1.0, window_weight: float = 1.0) -> Tensor:
    """Topology-aware W1 loss on queue and congestion-window marginals.

    Fidelity treats every wrong destination as unrelated. This loss uses the
    ordered physical supports, so moving probability to an adjacent queue bin
    costs less than moving it across the buffer. Queue distance is linear in
    packets and window distance is logarithmic because the window grid itself
    is logarithmic. Each axis is normalised to the unit interval.
    """
    prototype = next(iter(weights.values())) if isinstance(weights, dict) else weights
    coefficients = prototype.new_tensor([queue_weight, window_weight])
    if not bool(torch.isfinite(coefficients).all()) or \
            bool((coefficients < 0).any()):
        raise ValueError("Wasserstein axis weights must be finite and non-negative")
    w_bins, q_bins, _ = grid.axis_bins()

    def axis_term(pred: Tensor, ref: Tensor, bins: Tensor, n_bins: int,
                  positions: Tensor) -> Tensor:
        if n_bins < 2:
            return pred.new_zeros(pred.shape[0], pred.shape[2])
        pred_m = _axis_marginal(pred, bins, n_bins)
        ref_m = _axis_marginal(ref, bins, n_bins)
        delta = positions.diff().to(device=pred.device, dtype=pred.dtype)
        delta = delta / (positions[-1] - positions[0]).clamp_min(1.0e-12)
        cdf_gap = (pred_m.cumsum(1) - ref_m.cumsum(1)).abs()[:, :-1]
        return (cdf_gap * delta.reshape(1, -1, 1)).sum(1)

    q_pos = _axis_positions(grid.q_rep, q_bins, grid.n_q, logarithmic=False)
    w_pos = _axis_positions(grid.w_rep, w_bins, grid.n_w, logarithmic=True)
    total = None
    for h, pred in powers.items():
        ref = targets[h]
        per_source = (
            queue_weight * axis_term(pred, ref, q_bins, grid.n_q, q_pos)
            + window_weight * axis_term(pred, ref, w_bins, grid.n_w, w_pos))
        term = omega[h] * (_horizon_value(weights, h) * per_source).sum()
        total = term if total is None else total + term
    return total


def multitail_loss(powers: dict[int, Tensor], targets: dict[int, Tensor],
                   weights: Tensor | dict[int, Tensor],
                   omega: dict[int, float], q_rep: Tensor,
                   buffer_pkts: float, threshold_fracs: list[float]) -> Tensor:
    """Average absolute tail-probability error over several queue thresholds."""
    if not threshold_fracs:
        raise ValueError("tail_threshold_fracs must not be empty")
    fractions = q_rep.new_tensor(threshold_fracs)
    if not bool(torch.isfinite(fractions).all()) or \
            bool(((fractions <= 0) | (fractions >= 1)).any()):
        raise ValueError("tail threshold fractions must lie strictly in (0, 1)")
    masks = (q_rep.unsqueeze(0) >= fractions.unsqueeze(1) * buffer_pkts).to(
        q_rep.dtype)
    total = None
    for h, pred in powers.items():
        ref = targets[h]
        pred_tail = torch.einsum("km,bms->bks", masks, pred)
        ref_tail = torch.einsum("km,bms->bks", masks, ref)
        per_source = (pred_tail - ref_tail).abs().mean(1)
        term = omega[h] * (_horizon_value(weights, h) * per_source).sum()
        total = term if total is None else total + term
    return total


def total_loss(P: Tensor, targets: dict[int, Tensor],
               counts: Tensor | dict[int, Tensor],
               grid, cfg, parameters=None) -> tuple[Tensor, dict]:
    """Combine the distribution, observable, topology, tail and L2 terms.

    `targets[h]` is (B, M, M) column-normalised; `counts` is (B, M), the
    per-column sample count of the fitting half. Returns the scalar loss and
    a dict of components for logging.
    """
    horizons = sorted(targets)
    raw_omega = {h: h ** (-cfg.train.horizon_weight_power) for h in horizons}
    # Preserve the overall loss scale (and therefore the meaning of the
    # learning rate) while changing only relative horizon emphasis.
    normaliser = len(horizons) / sum(raw_omega.values())
    omega = {h: value * normaliser for h, value in raw_omega.items()}
    count_by_horizon = (
        {h: counts[h] for h in horizons} if isinstance(counts, dict)
        else {h: counts for h in horizons})
    obs_weights = {
        h: column_weights(value, cfg.data.c_min, cfg.data.c_cap_quantile)
        for h, value in count_by_horizon.items()
    }
    dist_weights = {
        h: balanced_column_weights(
            value, grid.tail_mask, cfg.data.c_min, cfg.data.c_cap_quantile,
            occupancy=cfg.train.dist_weight_occupancy,
            uniform=cfg.train.dist_weight_uniform,
            tail=cfg.train.dist_weight_tail)
        for h, value in count_by_horizon.items()
    }
    powers = propagator_powers(P, horizons)

    l_dist = distribution_loss(powers, targets, dist_weights, omega,
                               cfg.train.fidelity_eps)
    l_obs = observable_loss(powers, targets, obs_weights, omega, grid.q_rep,
                            grid.tail_mask, cfg.train.alpha_tail)
    l_wasserstein = P.new_zeros(())
    if cfg.train.lambda_wasserstein > 0:
        l_wasserstein = marginal_wasserstein_loss(
            powers, targets, dist_weights, omega, grid,
            queue_weight=cfg.train.wasserstein_queue_weight,
            window_weight=cfg.train.wasserstein_window_weight)
    l_multitail = P.new_zeros(())
    if cfg.train.lambda_multitail > 0:
        l_multitail = multitail_loss(
            powers, targets, obs_weights, omega, grid.q_rep,
            grid.cfg.buffer_pkts, cfg.train.tail_threshold_fracs)
    loss = (l_dist + cfg.train.lambda_obs * l_obs
            + cfg.train.lambda_wasserstein * l_wasserstein
            + cfg.train.lambda_multitail * l_multitail)
    l_reg = P.new_zeros(())
    if parameters is not None and cfg.train.lambda_reg > 0:
        l_reg = sum((p ** 2).sum() for p in parameters)
        loss = loss + cfg.train.lambda_reg * l_reg
    return loss, {
        "loss": float(loss.detach()),
        "dist": float(l_dist.detach()),
        "obs": float(l_obs.detach()),
        "wasserstein": float(l_wasserstein.detach()),
        "multitail": float(l_multitail.detach()),
        "reg": float(l_reg.detach()) if torch.is_tensor(l_reg) else float(l_reg),
    }
