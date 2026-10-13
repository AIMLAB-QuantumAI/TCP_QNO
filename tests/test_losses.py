"""Training objective: column weights, powers, and the loss terms."""
from __future__ import annotations

import math

import pytest
import torch

from qtro.config import load_config
from qtro.grid import Grid
from qtro.losses import (
    balanced_column_weights,
    column_weights,
    distribution_loss,
    dropped_columns,
    marginal_wasserstein_loss,
    multitail_loss,
    observable_loss,
    propagator_powers,
    total_loss,
)


def _grid_cfg():
    cfg = load_config(None, ["sim.device=cpu"])
    return Grid(cfg.grid, cfg.net, device="cpu"), cfg


def test_column_weights_sum_to_one_and_drop_thin_columns():
    counts = torch.tensor([[100.0, 5.0, 50.0, 0.0]])
    w = column_weights(counts, c_min=32, c_cap_quantile=1.0)
    assert w[0, 1] == 0.0 and w[0, 3] == 0.0
    assert w[0, 0] > 0 and w[0, 2] > 0
    assert math.isclose(float(w.sum()), 1.0, rel_tol=1e-6)


def test_column_weights_cap_limits_the_hottest_column():
    # One column 1000x the others. With the cap at the median of the surviving
    # counts, it must not own the loss.
    counts = torch.tensor([[100.0, 100.0, 100.0, 100_000.0]])
    w = column_weights(counts, c_min=32, c_cap_quantile=0.5)
    assert math.isclose(float(w[0, 3]), float(w[0, 0]), rel_tol=1e-6)
    uncapped = column_weights(counts, c_min=32, c_cap_quantile=1.0)
    assert float(uncapped[0, 3]) > 0.99


def test_column_weights_row_with_nothing_above_threshold_is_all_zero():
    counts = torch.tensor([[1.0, 2.0], [100.0, 100.0]])
    w = column_weights(counts, c_min=32, c_cap_quantile=0.9)
    assert torch.all(w[0] == 0.0)
    assert not torch.isnan(w).any()
    assert math.isclose(float(w[1].sum()), 1.0, rel_tol=1e-6)


def test_balanced_weights_default_matches_original_occupancy_weights():
    counts = torch.tensor([[100.0, 5.0, 50.0, 1000.0]])
    tail = torch.tensor([False, False, True, True])
    original = column_weights(counts, c_min=32, c_cap_quantile=0.9)
    mixed = balanced_column_weights(
        counts, tail, 32, 0.9, occupancy=1.0, uniform=0.0, tail=0.0)
    assert torch.equal(mixed, original)


def test_balanced_weights_mix_three_separately_normalised_components():
    counts = torch.full((1, 4), 100.0)
    tail = torch.tensor([False, False, True, True])
    mixed = balanced_column_weights(
        counts, tail, 32, 0.9, occupancy=0.4, uniform=0.3, tail=0.3)
    assert torch.allclose(mixed, torch.tensor([[0.175, 0.175, 0.325, 0.325]]))
    assert math.isclose(float(mixed.sum()), 1.0, rel_tol=1e-6)


def test_balanced_weights_renormalise_when_no_reliable_tail_exists():
    counts = torch.tensor([[100.0, 100.0, 1.0, 1.0]])
    tail = torch.tensor([False, False, True, True])
    mixed = balanced_column_weights(
        counts, tail, 32, 0.9, occupancy=0.4, uniform=0.3, tail=0.3)
    assert torch.allclose(mixed, torch.tensor([[0.5, 0.5, 0.0, 0.0]]))


@pytest.mark.parametrize("weights", [(-0.1, 1.0, 0.1), (0.0, 0.0, 0.0)])
def test_balanced_weights_reject_invalid_mixtures(weights):
    with pytest.raises(ValueError, match="source weights"):
        balanced_column_weights(torch.ones(1, 4) * 100,
                                torch.tensor([False, False, True, True]),
                                32, 0.9, occupancy=weights[0],
                                uniform=weights[1], tail=weights[2])


def test_dropped_columns_reports_what_the_weighting_discards():
    counts = torch.tensor([[100.0, 5.0, 5.0]])
    info = dropped_columns(counts, c_min=32)
    assert info["n_dropped"] == 2 and info["n_columns"] == 3
    assert math.isclose(info["occupancy_share_dropped"], 10.0 / 110.0,
                        rel_tol=1e-6)


def test_propagator_powers_match_repeated_multiplication():
    torch.manual_seed(0)
    P = torch.rand(2, 5, 5)
    P = P / P.sum(1, keepdim=True)
    powers = propagator_powers(P, [1, 2, 5])
    assert torch.allclose(powers[1], P)
    assert torch.allclose(powers[2], torch.bmm(P, P), atol=1e-6)
    manual = P
    for _ in range(4):
        manual = torch.bmm(P, manual)
    assert torch.allclose(powers[5], manual, atol=1e-5)


def test_propagator_powers_preserve_column_stochasticity():
    torch.manual_seed(0)
    P = torch.rand(1, 6, 6)
    P = P / P.sum(1, keepdim=True)
    for h, Ph in propagator_powers(P, [1, 3, 7]).items():
        assert torch.allclose(Ph.sum(1), torch.ones(1, 6), atol=1e-5), h


def test_distribution_loss_is_zero_on_a_perfect_prediction():
    torch.manual_seed(0)
    T = torch.rand(1, 4, 4)
    T = T / T.sum(1, keepdim=True)
    w = torch.full((1, 4), 0.25)
    loss = distribution_loss({1: T}, {1: T}, w, {1: 1.0}, eps=0.0)
    assert float(loss) < 1e-6


def test_distribution_loss_matches_a_hand_computation():
    # Two cells, one column each way; fidelity of (1,0) against (0.5,0.5) is
    # (sqrt(0.5))^2 = 0.5, so the loss on that column is 0.5.
    P = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    target = torch.tensor([[[0.5, 0.0], [0.5, 1.0]]])
    w = torch.tensor([[1.0, 0.0]])          # score only the first column
    loss = distribution_loss({1: P}, {1: target}, w, {1: 1.0}, eps=0.0)
    assert math.isclose(float(loss), 0.5, rel_tol=1e-5)


def test_distribution_loss_gradient_is_finite_at_a_one_sided_zero():
    """The training eps exists for exactly this case."""
    target = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])
    logits = torch.zeros(1, 2, 2, requires_grad=True)
    P = torch.softmax(logits, dim=1)
    w = torch.full((1, 2), 0.5)
    distribution_loss({1: P}, {1: target}, w, {1: 1.0}, eps=1e-12).backward()
    assert torch.isfinite(logits.grad).all()


def test_observable_loss_is_zero_on_a_perfect_prediction():
    grid, _ = _grid_cfg()
    torch.manual_seed(0)
    T = torch.rand(1, grid.M, grid.M)
    T = T / T.sum(1, keepdim=True)
    w = torch.full((1, grid.M), 1.0 / grid.M)
    loss = observable_loss({1: T}, {1: T}, w, {1: 1.0}, grid.q_rep,
                           grid.tail_mask, alpha_tail=10.0)
    assert float(loss) < 1e-5


def test_observable_loss_sees_a_tail_error_the_mean_queue_hides():
    """Same mean queue, different overflow mass: the tail term must separate them."""
    grid, _ = _grid_cfg()
    q = grid.q_rep
    tail = grid.tail_mask
    lo = int(torch.argmin(q))
    hi = int(torch.argmax(q))
    mid = int(torch.argmin((q - 0.5 * (q[lo] + q[hi])).abs()))
    assert bool(tail[hi]) and not bool(tail[mid])

    def column(idx_weights):
        col = torch.zeros(1, grid.M, grid.M)
        for i, wt in idx_weights:
            col[0, i, 0] = wt
        col[0, 0, 1:] = 1.0          # remaining columns are arbitrary but valid
        return col

    # Both put the same expected queue in column 0; one splits it across the
    # extremes (mass in the tail) and one concentrates it in the middle.
    a = column([(lo, 0.5), (hi, 0.5)])
    b = column([(mid, 1.0)])
    w = torch.zeros(1, grid.M)
    w[0, 0] = 1.0
    mean_only = observable_loss({1: a}, {1: b}, w, {1: 1.0}, q, tail,
                                alpha_tail=0.0)
    with_tail = observable_loss({1: a}, {1: b}, w, {1: 1.0}, q, tail,
                                alpha_tail=10.0)
    assert float(with_tail) > float(mean_only) + 1.0


def test_marginal_wasserstein_is_zero_for_identical_transitions():
    grid, _ = _grid_cfg()
    torch.manual_seed(11)
    target = torch.rand(1, grid.M, grid.M)
    target = target / target.sum(1, keepdim=True)
    weights = torch.full((1, grid.M), 1.0 / grid.M)
    loss = marginal_wasserstein_loss(
        {1: target}, {1: target}, weights, {1: 1.0}, grid)
    assert float(loss) < 1.0e-6


def test_marginal_wasserstein_penalises_far_queue_more_than_near_queue():
    grid, _ = _grid_cfg()
    _, q_bin, _ = grid.axis_bins()
    source = 0
    low = int(torch.nonzero(q_bin == 0)[0])
    near = int(torch.nonzero(q_bin == 1)[0])
    far = int(torch.nonzero(q_bin == grid.n_q - 1)[0])

    def deterministic(destination: int) -> torch.Tensor:
        matrix = torch.eye(grid.M).unsqueeze(0)
        matrix[0, :, source] = 0.0
        matrix[0, destination, source] = 1.0
        return matrix

    target = deterministic(low)
    weights = torch.zeros(1, grid.M)
    weights[0, source] = 1.0
    near_loss = marginal_wasserstein_loss(
        {1: deterministic(near)}, {1: target}, weights, {1: 1.0}, grid,
        queue_weight=1.0, window_weight=0.0)
    far_loss = marginal_wasserstein_loss(
        {1: deterministic(far)}, {1: target}, weights, {1: 1.0}, grid,
        queue_weight=1.0, window_weight=0.0)
    assert 0.0 < float(near_loss) < float(far_loss)


def test_multitail_loss_uses_multiple_buffer_thresholds():
    grid, _ = _grid_cfg()
    low = int(torch.argmin(grid.q_rep))
    high = int(torch.argmax(grid.q_rep))
    target = torch.eye(grid.M).unsqueeze(0)
    pred = target.clone()
    pred[0, :, 0] = 0.0
    pred[0, high, 0] = 1.0
    target[0, :, 0] = 0.0
    target[0, low, 0] = 1.0
    weights = torch.zeros(1, grid.M)
    weights[0, 0] = 1.0
    loss = multitail_loss(
        {1: pred}, {1: target}, weights, {1: 1.0}, grid.q_rep,
        grid.cfg.buffer_pkts, [0.6, 0.7, 0.8, 0.9])
    assert math.isclose(float(loss), 1.0, rel_tol=1.0e-6)


def test_total_loss_components_add_up():
    grid, cfg = _grid_cfg()
    torch.manual_seed(0)
    P = torch.rand(2, grid.M, grid.M)
    P = P / P.sum(1, keepdim=True)
    target = torch.rand(2, grid.M, grid.M)
    target = target / target.sum(1, keepdim=True)
    counts = torch.full((2, grid.M), 100.0)
    loss, parts = total_loss(P, {1: target}, counts, grid, cfg)
    assert math.isclose(parts["loss"],
                        parts["dist"] + cfg.train.lambda_obs * parts["obs"]
                        + cfg.train.lambda_wasserstein * parts["wasserstein"]
                        + cfg.train.lambda_multitail * parts["multitail"],
                        rel_tol=1e-5)
    assert parts["reg"] == 0.0
    assert set(parts) == {"loss", "dist", "obs", "wasserstein", "multitail", "reg"}


def test_total_loss_adds_topology_and_multitail_terms_when_enabled():
    grid, cfg = _grid_cfg()
    cfg.train.lambda_wasserstein = 0.2
    cfg.train.lambda_multitail = 0.3
    torch.manual_seed(17)
    P = torch.rand(1, grid.M, grid.M)
    P = P / P.sum(1, keepdim=True)
    target = torch.rand(1, grid.M, grid.M)
    target = target / target.sum(1, keepdim=True)
    counts = torch.full((1, grid.M), 100.0)
    loss, parts = total_loss(P, {1: target}, counts, grid, cfg)
    expected = (parts["dist"] + cfg.train.lambda_obs * parts["obs"]
                + 0.2 * parts["wasserstein"]
                + 0.3 * parts["multitail"])
    assert math.isclose(float(loss), expected, rel_tol=1.0e-5)
    assert parts["wasserstein"] > 0.0
    assert parts["multitail"] > 0.0


def test_total_loss_regularisation_enters_only_with_parameters():
    grid, cfg = _grid_cfg()
    cfg.train.lambda_reg = 1.0
    torch.manual_seed(0)
    P = torch.rand(1, grid.M, grid.M)
    P = P / P.sum(1, keepdim=True)
    counts = torch.full((1, grid.M), 100.0)
    plain, _ = total_loss(P, {1: P}, counts, grid, cfg)
    theta = [torch.full((3,), 2.0)]
    with_reg, parts = total_loss(P, {1: P}, counts, grid, cfg, parameters=theta)
    assert math.isclose(parts["reg"], 12.0, rel_tol=1e-6)
    assert float(with_reg) > float(plain)


@pytest.mark.parametrize("h0", [1, 3, 5])
def test_more_horizons_never_lowers_the_loss(h0):
    """Each horizon adds a non-negative term, so the sum is monotone in H_0."""
    torch.manual_seed(0)
    P = torch.rand(1, 8, 8)
    P = P / P.sum(1, keepdim=True)
    targets = {}
    for h in range(1, h0 + 1):
        t = torch.rand(1, 8, 8)
        targets[h] = t / t.sum(1, keepdim=True)
    w = torch.full((1, 8), 1.0 / 8)
    powers = propagator_powers(P, sorted(targets))
    omega = {h: 1.0 for h in targets}
    full = distribution_loss(powers, targets, w, omega, 0.0)
    first = distribution_loss({1: powers[1]}, {1: targets[1]}, w, {1: 1.0}, 0.0)
    assert float(full) >= float(first) - 1e-6


def test_horizon_weight_power_emphasises_one_step_without_changing_scale():
    grid, cfg = _grid_cfg()
    cfg.train.horizon_weight_power = 1.0
    torch.manual_seed(4)
    P = torch.rand(1, grid.M, grid.M)
    P = P / P.sum(1, keepdim=True)
    targets = {h: torch.eye(grid.M).unsqueeze(0) for h in (1, 2, 4)}
    counts = torch.full((1, grid.M), 100.0)
    loss, parts = total_loss(P, targets, counts, grid, cfg)
    assert torch.isfinite(loss)
    assert parts["dist"] >= 0.0
