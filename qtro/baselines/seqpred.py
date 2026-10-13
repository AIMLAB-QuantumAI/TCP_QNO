"""SeqPred: an unconstrained matrix predictor at the same budget.

The one-step matrix is a free identity-initialised base plus a rank-r
correction driven by the conditioning, with no structure keeping it a
probability law. Accuracy is scored after the conventional post-hoc repair,
clipping and column renormalisation; ``raw_propagator`` exposes the
pre-repair matrix so the validity cost of the unconstrained representation
can be measured rather than hidden by that repair.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from qtro.model.qtro import DEFAULT_COND_SCALE, N_COND_FEATURES


class SeqPred(nn.Module):
    """Comparable-budget raw matrix predictor with clip-and-renormalise inference."""

    def __init__(self, grid, n_cond_features: int = N_COND_FEATURES,
                 rank: int = 8, d_cond: int = 8, hidden: int = 64):
        super().__init__()
        M = grid.M
        self.grid = grid
        self.register_buffer(
            "cond_scale", torch.tensor(DEFAULT_COND_SCALE, dtype=torch.float32)
        )
        self.base = nn.Parameter(torch.eye(M))
        self.encoder = nn.Sequential(
            nn.Linear(n_cond_features, hidden), nn.Tanh(),
            nn.Linear(hidden, d_cond), nn.Tanh(),
        )
        self.out = nn.Parameter(0.01 * torch.randn(d_cond, rank, M))
        self.inp = nn.Parameter(0.01 * torch.randn(d_cond, rank, M))

    def set_conditioning_scale(self, scale: Tensor) -> None:
        scale = torch.as_tensor(scale, dtype=self.cond_scale.dtype,
                                device=self.cond_scale.device)
        if scale.shape != (N_COND_FEATURES,) or not bool(torch.all(scale > 0)):
            raise ValueError(
                f"conditioning scale must be {N_COND_FEATURES} positive values")
        self.cond_scale.copy_(scale)

    def raw_propagator(self, u: Tensor, xi: Tensor) -> Tensor:
        scale = self.cond_scale.to(device=u.device, dtype=u.dtype)
        cond = torch.cat([u, xi], dim=-1) / scale
        h = self.encoder(cond)
        out = torch.einsum("bc,crm->brm", h, self.out)
        inp = torch.einsum("bc,crn->brn", h, self.inp)
        return self.base.unsqueeze(0) + torch.einsum("brm,brn->bmn", out, inp)

    @staticmethod
    def repair(raw: Tensor) -> Tensor:
        clipped = raw.clamp_min(0)
        mass = clipped.sum(dim=1, keepdim=True)
        # A zero or numerically empty column has no defensible prediction.
        # Uniform is neutral and makes the fallback explicit without NaN.
        # The predicate must use the same threshold as the denominator:
        # dividing a 1e-38 column by a denominator clamped to 1e-30 leaves a
        # column sum of 1e-8 rather than one.
        uniform = torch.full_like(clipped, 1.0 / clipped.shape[1])
        threshold = 1.0e-30
        return torch.where(
            mass > threshold, clipped / mass.clamp_min(threshold), uniform)

    def propagator(self, u: Tensor, xi: Tensor) -> Tensor:
        return self.repair(self.raw_propagator(u, xi))

    def step(self, p: Tensor, u: Tensor, xi: Tensor) -> Tensor:
        return torch.einsum("bij,bj->bi", self.propagator(u, xi), p)
