import torch

from qtro.config import load_config
from qtro.grid import Grid
from qtro.quantum.pairs import build_sweeps


def _grid():
    cfg = load_config(None, ["sim.device=cpu"])
    return Grid(cfg.grid, cfg.net, device="cpu")


def test_sweep_names_and_order():
    sweeps = build_sweeps(_grid(), d_env=4)
    assert [s[0] for s in sweeps] == ["w_even", "w_odd", "q_even", "q_odd",
                                      "phase", "env_even", "env_odd"]


def test_indices_within_a_sweep_are_disjoint():
    for name, lo, hi in build_sweeps(_grid(), d_env=4):
        both = torch.cat([lo, hi])
        assert torch.unique(both).numel() == both.numel(), name


def test_indices_are_in_range():
    g = _grid()
    n = g.M * 4
    for name, lo, hi in build_sweeps(g, d_env=4):
        assert torch.all(lo >= 0) and torch.all(lo < n), name
        assert torch.all(hi >= 0) and torch.all(hi < n), name


def test_pair_counts_match_the_grid():
    counts = {n: lo.numel() for n, lo, _ in build_sweeps(_grid(), d_env=4)}
    assert counts["w_even"] + counts["w_odd"] == 1920
    assert counts["q_even"] + counts["q_odd"] == 1920
    assert counts["phase"] == 1024
    assert counts["env_even"] + counts["env_odd"] == 1536
    assert sum(counts.values()) == 6400


def test_cell_axis_sweeps_move_one_bin_and_keep_the_environment():
    g = _grid()
    for name, lo, hi in build_sweeps(g, d_env=4):
        if not name.startswith(("w_", "q_")):
            continue
        assert torch.all(lo % 4 == hi % 4), name
        a_lo = torch.div(lo // 4, g.n_q * g.n_phase, rounding_mode="floor")
        a_hi = torch.div(hi // 4, g.n_q * g.n_phase, rounding_mode="floor")
        b_lo = torch.div(lo // 4, g.n_phase, rounding_mode="floor") % g.n_q
        b_hi = torch.div(hi // 4, g.n_phase, rounding_mode="floor") % g.n_q
        moved = (a_lo != a_hi).long() + (b_lo != b_hi).long()
        assert torch.all(moved == 1), name
        assert torch.all((a_hi - a_lo).abs() <= 1), name
        assert torch.all((b_hi - b_lo).abs() <= 1), name


def test_disjoint_at_alternative_d_env_and_grid_sizes():
    """Empty sweeps are no-ops, not crashes: d_env=2 makes env_odd empty and
    n_w=2 makes w_odd empty, and one step further (d_env=1, n_w=1, n_q=1)
    torch.arange(1, 0, 2) would raise were the empty range not guarded."""
    g = _grid()
    for d_env in (1, 2, 3, 4, 5):
        sweeps = build_sweeps(g, d_env=d_env)
        names = [s[0] for s in sweeps]
        assert names == ["w_even", "w_odd", "q_even", "q_odd", "phase",
                         "env_even", "env_odd"]
        for name, lo, hi in sweeps:
            both = torch.cat([lo, hi])
            assert torch.unique(both).numel() == both.numel(), (d_env, name)
        if d_env == 1:
            counts = {n: lo.numel() for n, lo, _ in sweeps}
            assert counts["env_even"] == 0 and counts["env_odd"] == 0
        if d_env == 2:
            counts = {n: lo.numel() for n, lo, _ in sweeps}
            assert counts["env_odd"] == 0 and counts["env_even"] > 0

    for override, empty_name in (("grid.n_w=1", "w_odd"), ("grid.n_w=2", "w_odd"),
                                 ("grid.n_q=1", "q_odd"), ("grid.n_q=2", "q_odd")):
        cfg = load_config(None, ["sim.device=cpu", override])
        small = Grid(cfg.grid, cfg.net, device="cpu")
        sweeps = build_sweeps(small, d_env=4)
        counts = {n: lo.numel() for n, lo, _ in sweeps}
        assert counts[empty_name] == 0, override
        for name, lo, hi in sweeps:
            both = torch.cat([lo, hi])
            assert torch.unique(both).numel() == both.numel(), (override, name)
