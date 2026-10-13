import torch

from qtro.config import load_config
from qtro.grid import Grid


def _grid():
    cfg = load_config(None, None)
    return Grid(cfg.grid, cfg.net, device="cpu")


def test_shape_and_validity():
    g = _grid()
    pi = g.reset_targets()
    assert pi.shape == (2, g.M)
    assert torch.all(pi >= 0) and torch.all(pi < g.M)


def test_loss_halves_the_window_and_selects_congestion_avoidance():
    g = _grid()
    pi = g.reset_targets()
    tgt = pi[g.EVENT_LOSS]
    # The window never increases, and strictly decreases wherever beta * w
    # still lands above the lowest bin.
    assert torch.all(g.w_rep[tgt] <= g.w_rep + 1e-6)
    assert torch.all(g.phase_rep[tgt] == 1)
    # Excludes the lowest window bins, where beta * w falls under w_min and
    # the window floor applies instead (covered below).
    high = g.w_rep > 4.0
    assert torch.all(g.w_rep[tgt][high] < g.w_rep[high])


def test_loss_preserves_the_queue_bin():
    g = _grid()
    tgt = g.reset_targets()[g.EVENT_LOSS]
    assert torch.allclose(g.q_rep[tgt], g.q_rep)


def test_timeout_goes_to_lowest_window_and_slow_start():
    g = _grid()
    tgt = g.reset_targets()[g.EVENT_TIMEOUT]
    assert torch.all(g.w_rep[tgt] == g.w_rep.min())
    assert torch.all(g.phase_rep[tgt] == 0)
    assert torch.allclose(g.q_rep[tgt], g.q_rep)


def test_timeout_is_many_to_one_so_it_can_reduce_entropy():
    g = _grid()
    tgt = g.reset_targets()[g.EVENT_TIMEOUT]
    assert torch.unique(tgt).numel() < g.M


def test_loss_is_pinned_at_the_window_floor():
    """TCP cannot halve below one packet, so the lowest bins stay put.

    For a cell already in congestion avoidance at the lowest window bin, the
    loss reset is the identity. Loss then moves no mass out of those cells;
    pinning it keeps a later change to w_min or the bin layout from altering
    that silently.
    """
    g = _grid()
    a, _, c = g.axis_bins()
    tgt = g.reset_targets()[g.EVENT_LOSS]
    floor = g.cfg.beta * g.w_rep < g.cfg.w_min
    assert floor.any()
    assert torch.all(a[tgt][floor] == 0)
    # beta * w < w_min holds for window bins 0-2 on the default grid and all
    # of them clamp to bin 0, but only cells already in bin 0 are unchanged.
    stuck = floor & (c == 1) & (a == 0)
    assert torch.equal(tgt[stuck], torch.arange(g.M)[stuck])
