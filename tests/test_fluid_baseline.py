"""The fluid and diffusion domain baselines.

Each test pins one property that, if broken, would make the baseline lose for
the wrong reason.
"""
from __future__ import annotations

import torch

from qtro.aqm import red_prob
from qtro.baselines.fluid import ANCHORS, MODES, FluidPropagator, fluid_equilibrium
from qtro.config import load_config
from qtro.grid import Grid

HORIZONS = (1, 5, 20)


def _setup():
    cfg = load_config(None, ["sim.device=cpu"])
    grid = Grid(cfg.grid, cfg.net, device="cpu")
    u = torch.tensor([[0.05, 16.0], [0.1, 20.0], [0.2, 30.0]])
    xi = torch.tensor([[1500.0, 0.020, 2.0, 0.0],
                       [2000.0, 0.030, 4.0, 0.0],
                       [3000.0, 0.045, 8.0, 0.001]])
    return cfg, grid, u, xi


def test_every_mode_returns_a_valid_transition_kernel():
    cfg, grid, u, xi = _setup()
    assert MODES == ("identity", "fluid", "fluid_jump", "diffusion")
    for mode in MODES:
        for anchor in ANCHORS:
            kernels = FluidPropagator(grid, cfg, mode, anchor).kernels(
                u, xi, HORIZONS)
            assert sorted(kernels) == sorted(HORIZONS)
            for h, kernel in kernels.items():
                assert kernel.shape == (u.shape[0], grid.M, grid.M)
                assert bool((kernel >= 0).all()), f"{mode}/{anchor} h={h}"
                assert torch.allclose(kernel.sum(dim=1),
                                      torch.ones(u.shape[0], grid.M),
                                      atol=1e-5), f"{mode}/{anchor} h={h}"


def test_identity_mode_is_the_identity_at_every_horizon():
    cfg, grid, u, xi = _setup()
    kernels = FluidPropagator(grid, cfg, "identity").kernels(u, xi, HORIZONS)
    identity = torch.eye(grid.M).expand(u.shape[0], grid.M, grid.M)
    for kernel in kernels.values():
        assert torch.allclose(kernel, identity)


def test_equilibrium_satisfies_both_defining_conditions():
    """The operating point must fill the link and balance the window."""
    cfg, grid, u, xi = _setup()
    p_max, min_th = u[:, 0:1], u[:, 1:2]
    capacity, rtt_prop = xi[:, 0:1], xi[:, 1:2]
    n_flows, p_err = xi[:, 2:3], xi[:, 3:4]
    max_th = torch.full_like(p_max, cfg.aqm.max_th_frac * cfg.grid.buffer_pkts)
    w_star, q_star = fluid_equilibrium(
        p_max, min_th, max_th, p_err, capacity, rtt_prop, n_flows,
        cfg.grid.buffer_pkts, cfg.grid.beta, cfg.grid.w_min, cfg.grid.w_max)

    rtt = rtt_prop + q_star / capacity
    # The link is exactly filled.
    assert torch.allclose(n_flows * w_star / rtt, capacity, rtol=1e-3)
    # Growth balances the mean decrease, i.e. w = sqrt(1 / ((1 - beta) p)).
    p = red_prob(q_star, p_max, min_th, max_th)
    p = 1.0 - (1.0 - p_err) * (1.0 - p)
    balanced = (1.0 / ((1.0 - cfg.grid.beta) * p.clamp_min(1e-9))).sqrt()
    assert torch.allclose(w_star, balanced, rtol=1e-2)


def test_balanced_anchoring_starts_with_no_queue_drift():
    """Balanced anchoring exists so the fluid model is not charged for a drift
    that only comes from assuming the other sessions sit somewhere else."""
    cfg, grid, u, xi = _setup()
    q = grid.q_rep
    kernels = {a: FluidPropagator(grid, cfg, "fluid", a).kernels(u, xi, (1,))[1]
               for a in ANCHORS}
    drift = {a: (torch.einsum("i,bij->bj", q, k)
                 - q.unsqueeze(0)).abs().mean() for a, k in kernels.items()}
    assert drift["balanced"] < drift["equilibrium"]


def test_diffusion_spreads_mass_and_the_fluid_variants_do_not():
    cfg, grid, u, xi = _setup()
    counts = {}
    for mode in ("fluid", "fluid_jump", "diffusion"):
        kernel = FluidPropagator(grid, cfg, mode).kernels(u, xi, (20,))[20]
        counts[mode] = float((kernel > 1e-9).float().sum(dim=1).mean())
    assert counts["fluid"] == 1.0
    assert counts["fluid_jump"] == 1.0
    assert counts["diffusion"] > 10.0


def test_random_loss_lowers_the_equilibrium_window():
    """Loss that is not congestion still shrinks the window: p_err enters the
    window balance but not the link-filling condition."""
    cfg, grid, u, xi = _setup()
    p_max, min_th = u[:1, 0:1], u[:1, 1:2]
    capacity, rtt_prop, n_flows = xi[:1, 0:1], xi[:1, 1:2], xi[:1, 2:3]
    max_th = torch.full_like(p_max, cfg.aqm.max_th_frac * cfg.grid.buffer_pkts)

    def solve(p_err_value: float):
        return fluid_equilibrium(
            p_max, min_th, max_th, torch.full_like(p_max, p_err_value),
            capacity, rtt_prop, n_flows, cfg.grid.buffer_pkts, cfg.grid.beta,
            cfg.grid.w_min, cfg.grid.w_max)

    clean_w, clean_q = solve(0.0)
    lossy_w, lossy_q = solve(0.01)
    assert float(lossy_w) < float(clean_w)
    assert float(lossy_q) < float(clean_q)
