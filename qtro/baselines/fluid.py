"""Fluid and diffusion domain baselines for the TCP/RED transition kernel.

The fluid and control-theoretic line (Misra et al., Hollot et al.) and the
diffusion-approximation line (Domanski, Marek et al.) are the prior art the
learned operator is compared against. These propagators make that comparison
measurable on the same held-out data.

Neither model is trained. Both are determined by the condition ``xi`` and the
control ``u`` alone, so no data source can favour them and no seed axis
exists: the comparison is method against method, not fit against fit.

Three variants, in increasing order of what they are allowed to represent:

``fluid``
    The Misra/Hollot mean-field equation. The loss jump is replaced by its
    mean rate, so the window decays smoothly and the protocol phase never
    changes. Its prediction is a point mass.

``fluid_jump``
    The same dynamics with the jump retained deterministically: the window is
    multiplied by beta (or reset on timeout) each time the expected loss count
    crosses an integer. This is strictly more capable than the cited model and
    is included so the comparison cannot be accused of testing a strawman; a
    phase-blind predictor scores zero on every column whose source phase is
    left behind.

``diffusion``
    ``fluid_jump`` plus the second moment, in the spirit of the combined
    diffusion-approximation models: Poisson arrival variance spreads the
    queue, the Poisson loss count spreads the window, and the loss count sets
    the probability of having left the source phase.

The diffusion variant returns a *product* of the three marginals. That is the
approximation the diffusion literature makes, and it is the substantive
difference from an operator that learns the joint law. ``identity`` is not a
domain model: it holds every cell in place and says how much of a model's
error is worse than predicting no change at all.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

from qtro.aqm import red_prob

MODES = ("identity", "fluid", "fluid_jump", "diffusion")
# How the load of the other N-1 sessions is set. The fluid model describes an
# aggregate; a conditional kernel asks it to start from one session's window
# beside a queue, which the aggregate description does not determine. Both
# readings are run because the choice changes the answer and neither is
# obviously the intended one.
#
#   equilibrium -- the others sit at the fluid operating point.
#   balanced    -- the others carry whatever load makes the link exactly full
#                  at the observed queue, so the initial queue drift is zero.
#                  This is the more favourable reading and is the one to beat.
ANCHORS = ("equilibrium", "balanced")
PHASE_SS, PHASE_CA = 0, 1
# 0.25 ms. One packet service slot is 1/C, which is 0.33 ms at the slowest
# capacity on the ns-3 dataset, so the integrator resolves the fastest
# timescale the discrete simulator itself has.
SUBSTEP_S = 0.00025


def _normal_bin_mass(edges: Tensor, mean: Tensor, sd: Tensor) -> Tensor:
    """Mass a normal law puts in each bin, renormalised over the axis.

    ``edges`` is ``(n + 1,)`` and ``mean``/``sd`` are ``(...,)``. Renormalising
    truncates the law to the axis rather than piling the tails onto the end
    bins, which would invent occupancy at exactly the boundary being measured.
    """
    z = (edges.reshape((1,) * mean.ndim + (-1,)) - mean.unsqueeze(-1)) / (
        sd.unsqueeze(-1).clamp_min(1e-6) * math.sqrt(2.0))
    cdf = 0.5 * (1.0 + torch.erf(z))
    mass = (cdf[..., 1:] - cdf[..., :-1]).clamp_min(0.0)
    return mass / mass.sum(-1, keepdim=True).clamp_min(1e-30)


def fluid_equilibrium(p_max: Tensor, min_th: Tensor, max_th: Tensor,
                      p_err: Tensor, capacity: Tensor, rtt_prop: Tensor,
                      n_flows: Tensor, buffer_pkts: float, beta: float,
                      w_min: float, w_max: float,
                      iterations: int = 64) -> tuple[Tensor, Tensor]:
    """Hollot's operating point: the (window, queue) the fluid model settles at.

    Two conditions meet at the equilibrium. The link is exactly filled,
    ``N w / R = C``, and the window's growth balances its mean decrease,
    ``1 / R = (w / R) p (1 - beta) w``, which is the familiar
    ``w = sqrt(1 / ((1 - beta) p))``. Substituting the first into the second
    leaves one increasing function of the queue, so bisection converges.

    This matters for more than reporting. The transition kernel is asked for a
    prediction from every source cell, including cells where the tagged flow's
    window is far from equilibrium. Assuming the other N-1 flows share that
    window drives the queue to a boundary within one snapshot and makes the
    fluid model look far worse than it is; pinning them here is the standard
    operating-point treatment and is what the cited work actually does.
    """
    low = torch.zeros_like(p_max)
    high = torch.full_like(p_max, float(buffer_pkts))
    for _ in range(iterations):
        q = 0.5 * (low + high)
        p = red_prob(q, p_max, min_th, max_th)
        p = 1.0 - (1.0 - p_err) * (1.0 - p)
        w_congestion = (1.0 / ((1.0 - beta) * p.clamp_min(1e-9))).sqrt()
        w_load = capacity * (rtt_prop + q / capacity) / n_flows
        # w_load rises with the queue and w_congestion falls with it, so the
        # difference is increasing and the root is unique.
        too_low = w_load < w_congestion
        low = torch.where(too_low, q, low)
        high = torch.where(too_low, high, q)
    q_star = 0.5 * (low + high)
    w_star = (capacity * (rtt_prop + q_star / capacity) / n_flows).clamp(
        w_min, w_max)
    return w_star, q_star


def _delta_bin_mass(edges: Tensor, value: Tensor) -> Tensor:
    """One-hot over the bin containing ``value``, clamped to the axis."""
    n = edges.numel() - 1
    index = (torch.bucketize(value.contiguous(), edges, right=True) - 1)
    index = index.clamp(0, n - 1)
    return torch.zeros(value.shape + (n,), device=value.device,
                       dtype=torch.float32).scatter_(-1, index.unsqueeze(-1),
                                                     1.0)


class FluidPropagator:
    """Fluid or diffusion h-step kernels for one batch of conditions.

    Unlike the learned propagators this class does not expose a one-step
    matrix to be raised to a power. Composing a one-step kernel h times adds
    the discretisation's own diffusion at every hop, which would credit the
    baseline with spread it does not model; the fluid and diffusion
    literatures solve to time t directly, and so does this.
    """

    def __init__(self, grid, cfg, mode: str = "fluid",
                 anchor: str = "balanced"):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if anchor not in ANCHORS:
            raise ValueError(f"anchor must be one of {ANCHORS}, got {anchor!r}")
        self.grid = grid
        self.cfg = cfg
        self.mode = mode
        self.anchor = anchor
        self.last_equilibrium: tuple[Tensor, Tensor] | None = None

    def _initial_state(self, B: int, device: str) -> dict[str, Tensor]:
        grid = self.grid
        w = grid.w_rep.to(device).expand(B, -1).clone().float()
        q = grid.q_rep.to(device).expand(B, -1).clone().float()
        phase = grid.phase_rep.to(device).expand(B, -1).clone().long()
        # The RED average is not a grid axis, so its initial value has to be
        # assumed. Starting it at the queue is the stationary choice. With
        # w_q = 1e-4 per packet slot the average moves by well under one
        # packet over the longest horizon here, so the assumption carries
        # almost no weight, and that slowness is itself a property of the
        # configuration, not an artefact of this baseline.
        # loss_count is the residual that drives the deterministic jump and
        # is decremented when one fires; loss_total is the cumulative expected
        # count, which is what sets the diffusion spread.
        return {"w": w, "q": q, "phase": phase, "avg_q": q.clone(),
                "loss_count": torch.zeros_like(w),
                "loss_total": torch.zeros_like(w)}

    def kernels(self, u: Tensor, xi: Tensor, horizons) -> dict[int, Tensor]:
        """Return ``{h: (B, M, M)}`` with destination first, source second."""
        grid, cfg = self.grid, self.cfg
        device = u.device
        horizons = sorted({int(h) for h in horizons})
        B, M = u.shape[0], grid.M

        p_max = u[:, 0:1]
        min_th = u[:, 1:2]
        capacity = xi[:, 0:1]
        rtt_prop = xi[:, 1:2]
        n_flows = xi[:, 2:3]
        p_err = xi[:, 3:4]
        max_th = cfg.aqm.max_th_frac * cfg.grid.buffer_pkts
        max_th = torch.full_like(p_max, float(max_th))
        buffer_pkts = float(cfg.grid.buffer_pkts)
        beta = float(cfg.grid.beta)
        w_min = float(cfg.grid.w_min)
        w_max = float(cfg.grid.w_max)
        fast_w = float(cfg.grid.fast_retransmit_w)
        log_beta_squared = math.log(beta) ** 2

        w_star, q_star = fluid_equilibrium(
            p_max, min_th, max_th, p_err, capacity, rtt_prop, n_flows,
            buffer_pkts, beta, w_min, w_max)
        self.last_equilibrium = (w_star, q_star)

        state = self._initial_state(B, device)
        if self.mode == "identity":
            identity = torch.eye(M, device=device).expand(B, M, M)
            return {h: identity.contiguous() for h in horizons}
        if self.anchor == "balanced":
            rtt0 = rtt_prop + state["q"] / capacity
            other_load = (capacity * rtt0 - state["w"]).clamp_min(0.0)
        else:
            other_load = (n_flows - 1.0) * w_star
        # Integrated arrival rate, the diffusion coefficient of a queue fed by
        # Poisson arrivals and drained by a deterministic server: the service
        # process contributes no variance, so alpha reduces to the arrival
        # rate itself.
        queue_variance = torch.zeros_like(state["w"])

        out: dict[int, Tensor] = {}
        n_sub = max(1, int(round(cfg.sim.dt_s / SUBSTEP_S)))
        dt = cfg.sim.dt_s / n_sub
        for step in range(1, max(horizons) + 1):
            for _ in range(n_sub):
                self._advance(state, queue_variance, dt, capacity, rtt_prop,
                              n_flows, p_err, p_max, min_th, max_th,
                              buffer_pkts, beta, w_min, w_max, fast_w,
                              other_load)
            if step in horizons:
                out[step] = self._emit(state, queue_variance,
                                       log_beta_squared, fast_w, B, M)
        return out

    def _advance(self, state, queue_variance, dt, capacity, rtt_prop, n_flows,
                 p_err, p_max, min_th, max_th, buffer_pkts, beta, w_min,
                 w_max, fast_w, other_load) -> None:
        w, q = state["w"], state["q"]
        rtt = rtt_prop + q / capacity
        ack_rate = w / rtt
        p_red = red_prob(state["avg_q"], p_max, min_th, max_th)
        p_total = 1.0 - (1.0 - p_err) * (1.0 - p_red)

        # Slow start adds one segment per ACK, congestion avoidance one per
        # window of ACKs. Misra et al. write only the second; giving the
        # baseline the first as well can only help it.
        growth = torch.where(state["phase"] == PHASE_SS,
                             ack_rate, ack_rate / w.clamp_min(1e-6))
        loss_rate = ack_rate * p_total
        state["loss_count"] = state["loss_count"] + loss_rate * dt
        state["loss_total"] = state["loss_total"] + loss_rate * dt

        if self.mode == "fluid":
            # Jump replaced by its mean rate: the classical fluid equation,
            # dW/dt = 1/R - (W/2)(W/R)p for beta = 1/2.
            decrease = loss_rate * (1.0 - beta) * w
            w = w + (growth - decrease) * dt
        else:
            w = w + growth * dt
            # Applied unconditionally: an `if fired.any()` here would force a
            # device synchronisation on every one of the 800 substeps.
            fired = state["loss_count"] >= 1.0
            state["loss_count"] = torch.where(
                fired, state["loss_count"] - 1.0, state["loss_count"])
            halve = fired & (w >= fast_w)
            timeout = fired & (w < fast_w)
            w = torch.where(halve, (beta * w).clamp_min(w_min), w)
            w = torch.where(timeout, torch.full_like(w, w_min), w)
            state["phase"] = torch.where(
                halve, torch.full_like(state["phase"], PHASE_CA),
                state["phase"])
            state["phase"] = torch.where(
                timeout, torch.full_like(state["phase"], PHASE_SS),
                state["phase"])
        state["w"] = w.clamp(w_min, w_max)

        # The tagged session carries its own evolving window; the rest carry
        # the anchored load, which is what keeps the queue on the scale the
        # reference kernel actually visits.
        aggregate_w = state["w"] + other_load
        admitted_rate = aggregate_w / rtt * (1.0 - p_total)
        drain = torch.where(q > 0.0, capacity.expand_as(q),
                            torch.minimum(admitted_rate,
                                          capacity.expand_as(q)))
        state["q"] = (q + (admitted_rate - drain) * dt).clamp(0.0, buffer_pkts)
        queue_variance += admitted_rate * dt
        state["avg_q"] = state["avg_q"] + (
            self.cfg.aqm.w_q * capacity * dt) * (state["q"] - state["avg_q"])

    def _emit(self, state, queue_variance, log_beta_squared, fast_w,
              B: int, M: int) -> Tensor:
        """Assemble the destination distribution for every source cell."""
        grid = self.grid
        n_w, n_q, n_phase = grid.n_w, grid.n_q, grid.n_phase
        w_edges = grid.w_edges.to(state["w"].device)
        q_edges = grid.q_edges.to(state["q"].device)

        if self.mode == "diffusion":
            # Poisson loss count spreads log-window; the window axis is
            # log-spaced, so a lognormal law is a normal law on its edges.
            sd_log_w = (state["loss_total"]
                        * log_beta_squared).clamp_min(0.0).sqrt()
            w_mass = _normal_bin_mass(w_edges.log(), state["w"].log(),
                                      sd_log_w)
            q_mass = _normal_bin_mass(q_edges, state["q"],
                                      queue_variance.clamp_min(0.0).sqrt())
            stay = torch.exp(-state["loss_total"])
            jumped_phase = torch.where(state["w"] >= fast_w,
                                       torch.full_like(state["phase"],
                                                       PHASE_CA),
                                       torch.full_like(state["phase"],
                                                       PHASE_SS))
            phase_mass = torch.zeros(B, M, n_phase, device=state["w"].device)
            phase_mass.scatter_add_(
                -1, grid.phase_rep.to(state["w"].device).expand(
                    B, -1).unsqueeze(-1), stay.unsqueeze(-1))
            phase_mass.scatter_add_(-1, jumped_phase.unsqueeze(-1),
                                    (1.0 - stay).unsqueeze(-1))
        else:
            w_mass = _delta_bin_mass(w_edges, state["w"])
            q_mass = _delta_bin_mass(q_edges, state["q"])
            phase_mass = torch.zeros(B, M, n_phase, device=state["w"].device)
            phase_mass.scatter_(-1, state["phase"].unsqueeze(-1), 1.0)

        joint = (w_mass.unsqueeze(-1).unsqueeze(-1)
                 * q_mass.unsqueeze(-2).unsqueeze(-1)
                 * phase_mass.unsqueeze(-2).unsqueeze(-2))
        joint = joint.reshape(B, M, n_w * n_q * n_phase)
        joint = joint / joint.sum(-1, keepdim=True).clamp_min(1e-30)
        # (B, source, destination) -> (B, destination, source).
        return joint.transpose(1, 2).contiguous()
