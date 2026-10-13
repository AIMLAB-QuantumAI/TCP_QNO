"""CNTO, the conditioned neural transition operator: the decisive learned baseline.

Its columns are softmaxes, so it satisfies validity exactly, by the same kind
of construction QTRO uses; any advantage QTRO shows has to come from the
parameterisation and not from the constraint. The logits are a shared base
matrix plus a rank-r correction driven by the conditioning. A head emitting
all M^2 logits from the conditioning vector would need M^2 x d_hidden
parameters, some 34x QTRO's at M = 512, and would turn the comparison into a
capacity comparison; the low-rank form keeps the budgets comparable.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from qtro.model.qtro import DEFAULT_COND_SCALE, N_COND_FEATURES


class CNTO(nn.Module):
    """Neural transition operator whose columns are softmaxes, hence exactly valid."""

    def __init__(self, grid, n_cond_features: int, rank: int = 8,
                 d_cond: int = 8, hidden: int = 64):
        super().__init__()
        self.grid = grid
        self.register_buffer(
            "cond_scale", torch.tensor(DEFAULT_COND_SCALE, dtype=torch.float32)
        )
        M = grid.M
        # Zero init makes the initial propagator uniform over destinations,
        # the maximum-entropy starting point: the analogue of QTRO's
        # near-identity initialisation at small angles, in that neither starts
        # with a preferred transition structure.
        self.base = nn.Parameter(torch.zeros(M, M))
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
                f"conditioning scale must be {N_COND_FEATURES} positive values, "
                f"got shape={tuple(scale.shape)}")
        self.cond_scale.copy_(scale)

    def propagator(self, u: Tensor, xi: Tensor) -> Tensor:
        B = u.shape[0]
        logits = self.base.unsqueeze(0).expand(B, -1, -1)
        scale = self.cond_scale.to(device=u.device, dtype=u.dtype)
        cond = torch.cat([u, xi], dim=-1) / scale
        h = self.encoder(cond)
        out = torch.einsum("bc,crm->brm", h, self.out)
        inp = torch.einsum("bc,crn->brn", h, self.inp)
        logits = logits + torch.einsum("brm,brn->bmn", out, inp)
        # Softmax over dim=1, the destination axis: column j of the result is
        # P(next = . | now = j), matching QTRO.propagator.
        return torch.softmax(logits, dim=1)

    def step(self, p: Tensor, u: Tensor, xi: Tensor) -> Tensor:
        return torch.einsum("bij,bj->bi", self.propagator(u, xi), p)

    def rollout(self, p0: Tensor, u_seq: Tensor, xi: Tensor) -> Tensor:
        if u_seq.ndim != 3 or u_seq.shape[0] != p0.shape[0]:
            raise ValueError("u_seq must have shape (B, H, action_dim)")
        out, p = [], p0
        repeated = (u_seq.shape[1] <= 1 or torch.equal(
            u_seq[:, 1:], u_seq[:, :1].expand(-1, u_seq.shape[1] - 1, -1)))
        propagator = self.propagator(u_seq[:, 0], xi) if repeated else None
        for h in range(u_seq.shape[1]):
            matrix = (propagator if propagator is not None
                      else self.propagator(u_seq[:, h], xi))
            p = torch.bmm(matrix, p.unsqueeze(-1)).squeeze(-1)
            out.append(p)
        return torch.stack(out, dim=1)
