"""The state grid: window, queue and phase bins, and the protocol's reset targets."""
from __future__ import annotations

import torch
from torch import Tensor

from qtro.config import GridConfig, NetConfig


def _clustered_edges(lo: float, hi: float, n: int, centre_frac: float,
                     strength: float, device: str) -> Tensor:
    """Monotone edges on [lo, hi] with bins concentrated around centre_frac.

    The target bin density is 1 + strength * exp(-((t - c) / 0.15)^2) on
    [0, 1]. Edges are placed at equally spaced quantiles of that density
    (inverse-CDF sampling), so bins are narrow where the density is high
    (near centre_frac) and wide where it is low. A finely sampled CDF is
    built by trapezoidal integration and inverted by interpolation; this
    guarantees edges are strictly increasing and hit both endpoints exactly.
    """
    n_fine = max(2001, 50 * n)
    t_fine = torch.linspace(0.0, 1.0, n_fine, dtype=torch.float64, device=device)
    density = 1.0 + strength * torch.exp(-(((t_fine - centre_frac) / 0.15) ** 2))
    dt = t_fine[1] - t_fine[0]
    seg = (density[:-1] + density[1:]) * 0.5 * dt
    cdf = torch.cumsum(seg, dim=0)
    cdf = torch.cat([torch.zeros(1, dtype=torch.float64, device=device), cdf])
    cdf = cdf / cdf[-1]

    u = torch.linspace(0.0, 1.0, n + 1, dtype=torch.float64, device=device)
    idx = torch.searchsorted(cdf, u).clamp(1, n_fine - 1)
    c0, c1 = cdf[idx - 1], cdf[idx]
    t0, t1 = t_fine[idx - 1], t_fine[idx]
    frac = (u - c0) / (c1 - c0).clamp_min(1e-12)
    t_edges = t0 + frac * (t1 - t0)
    t_edges[0], t_edges[-1] = 0.0, 1.0

    return (lo + (hi - lo) * t_edges).to(torch.float32)


class Grid:
    """Discretisation of the TCP state space into M = n_w * n_q * n_phase cells.

    RTT is not a grid axis: r = rtt_prop + q / capacity is determined by the
    queue bin, so it is derived rather than discretised.
    """

    def __init__(self, cfg: GridConfig, net: NetConfig, device: str = "cpu"):
        self.cfg = cfg
        self.device = device
        self.n_w, self.n_q, self.n_phase = cfg.n_w, cfg.n_q, cfg.n_phase
        self.M = self.n_w * self.n_q * self.n_phase
        self.q_crit = cfg.q_crit_frac * cfg.buffer_pkts

        self.w_edges = torch.logspace(
            torch.log10(torch.tensor(cfg.w_min)).item(),
            torch.log10(torch.tensor(cfg.w_max)).item(),
            self.n_w + 1, device=device, dtype=torch.float32,
        )
        self.q_edges = _clustered_edges(
            0.0, cfg.buffer_pkts, self.n_q, cfg.q_crit_frac,
            cfg.q_cluster_strength, device,
        )

        w_centres = 0.5 * (self.w_edges[:-1] + self.w_edges[1:])
        q_centres = 0.5 * (self.q_edges[:-1] + self.q_edges[1:])

        self.w_rep = w_centres.repeat_interleave(self.n_q * self.n_phase)
        self.q_rep = q_centres.repeat_interleave(self.n_phase).repeat(self.n_w)
        self.phase_rep = torch.arange(
            self.n_phase, device=device
        ).repeat(self.n_w * self.n_q)

        self.rtt_rep = net.rtt_prop_s + self.q_rep / net.capacity_pps
        self.tail_mask = self.q_rep >= self.q_crit

    def cell_index(self, a: Tensor, b: Tensor, c: Tensor) -> Tensor:
        return (a * self.n_q + b) * self.n_phase + c

    def axis_bins(self) -> tuple[Tensor, Tensor, Tensor]:
        """Inverse of cell_index over all M cells."""
        idx = torch.arange(self.M, device=self.device)
        a = torch.div(idx, self.n_q * self.n_phase, rounding_mode="floor")
        b = torch.div(idx, self.n_phase, rounding_mode="floor") % self.n_q
        return a, b, idx % self.n_phase

    def digitize(self, w: Tensor, q: Tensor, phase: Tensor) -> Tensor:
        a = torch.bucketize(w.contiguous(), self.w_edges, right=True) - 1
        b = torch.bucketize(q.contiguous(), self.q_edges, right=True) - 1
        a = a.clamp(0, self.n_w - 1)
        b = b.clamp(0, self.n_q - 1)
        c = phase.long().clamp(0, self.n_phase - 1)
        return self.cell_index(a, b, c)

    EVENT_LOSS = 0
    EVENT_TIMEOUT = 1
    n_events = 2

    def reset_targets(self) -> Tensor:
        """Cell-level reset map: where loss and timeout take each cell.

        Loss multiplies the window by beta and moves the sender to congestion
        avoidance; a timeout returns it to w_min and to slow start. Neither
        event moves the queue. Targets are fixed by the protocol and are not
        learned.
        """
        _, q_bin, _ = self.axis_bins()

        loss_w = self.cfg.beta * self.w_rep
        loss_a = (torch.bucketize(loss_w, self.w_edges, right=True) - 1)
        loss_a = loss_a.clamp(0, self.n_w - 1)
        loss = self.cell_index(loss_a, q_bin, torch.ones_like(q_bin))

        zero = torch.zeros_like(q_bin)
        timeout = self.cell_index(zero, q_bin, zero)

        return torch.stack([loss, timeout], dim=0)
