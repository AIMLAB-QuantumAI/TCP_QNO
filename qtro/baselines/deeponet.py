"""DeepONet propagator: a named architecture from the operator-learning literature.

The construction is the canonical one. DeepONet approximates an operator as
an inner product between a branch network of the input and a trunk network
of the query coordinate,

    G(v)(y) ~ sum_k b_k(v) t_k(y),

which here means: the input is the conditioning pair (u, xi), the query is a
(destination, source) cell pair, and the value is a transition logit. The
trunk therefore makes the kernel a smooth function of physical cell
coordinates rather than a free matrix, and that is the architectural claim
being tested.

Two choices decide whether the comparison is fair rather than rhetorical.

* The shared base matrix is kept. The conditioned neural transition operator
  has one, and removing it here would force DeepONet to express the entire
  kernel through a low-rank product while its comparator got a free matrix
  plus a correction. The remaining difference is then exactly the
  interesting one: a coordinate-driven nonlinear basis against a
  conditioning-driven low-rank outer product.
* The trunk sees displacements as well as endpoints. A transition kernel is
  about where mass moves, so withholding (w_d - w_s, q_d - q_s) would hide
  the variable the operator is a function of.

Columns are softmaxes over the destination axis, so this model satisfies the
validity constraint exactly, by construction; only accuracy is at stake.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from qtro.model.qtro import DEFAULT_COND_SCALE, N_COND_FEATURES

N_TRUNK_FEATURES = 8


class DeepONet(nn.Module):
    """Branch-trunk propagator whose columns are exactly stochastic."""

    def __init__(self, grid, n_cond_features: int, rank: int = 42,
                 hidden: int = 256, buffer_pkts: float = 90.0,
                 w_max: float = 64.0):
        super().__init__()
        self.grid = grid
        self.rank = int(rank)
        M = grid.M
        self.register_buffer(
            "cond_scale", torch.tensor(DEFAULT_COND_SCALE, dtype=torch.float32))

        # Trunk coordinates are fixed by the grid, so they are a buffer rather
        # than something recomputed per step. Physical units are normalised so
        # window and queue enter on comparable scales.
        w = grid.w_rep.float() / float(w_max)
        q = grid.q_rep.float() / float(buffer_pkts)
        phase = grid.phase_rep.float()
        dest = torch.stack([w, q, phase], dim=-1)          # (M, 3)
        pair = torch.cat([
            dest.unsqueeze(1).expand(M, M, 3),             # destination
            dest.unsqueeze(0).expand(M, M, 3),             # source
            (w.unsqueeze(1) - w.unsqueeze(0)).unsqueeze(-1),
            (q.unsqueeze(1) - q.unsqueeze(0)).unsqueeze(-1),
        ], dim=-1)                                          # (M, M, 8)
        self.register_buffer("trunk_input", pair.reshape(M * M, N_TRUNK_FEATURES))

        self.base = nn.Parameter(torch.zeros(M, M))
        # LayerNorm on both heads, and the inner product scaled by 1/sqrt(rank).
        #
        # Without this the branch-trunk product saturates the softmax and the
        # conditioning dies. Measured on a first version that omitted it: the
        # correction reached a standard deviation of 2274 against a base of
        # 0.14, logits ran to 6573, and only 0.9% of the correction responded
        # to u at all. The product of two unbounded heads is free to grow, and
        # once the softmax saturates the gradient reaching the u-dependent
        # part is gone. DeepONet does not meet this in its usual setting
        # because it regresses real values rather than logits.
        self.branch = nn.Sequential(
            nn.Linear(n_cond_features, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, self.rank), nn.LayerNorm(self.rank),
        )
        self.trunk = nn.Sequential(
            nn.Linear(N_TRUNK_FEATURES, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, self.rank), nn.LayerNorm(self.rank),
        )
        self.register_buffer("inner_scale",
                             torch.tensor(self.rank ** -0.5, dtype=torch.float32))

    def set_conditioning_scale(self, scale: Tensor) -> None:
        scale = torch.as_tensor(scale, dtype=self.cond_scale.dtype,
                                device=self.cond_scale.device)
        if scale.shape != (N_COND_FEATURES,) or not bool(torch.all(scale > 0)):
            raise ValueError(
                f"conditioning scale must be {N_COND_FEATURES} positive values, "
                f"got shape={tuple(scale.shape)}")
        self.cond_scale.copy_(scale)

    def propagator(self, u: Tensor, xi: Tensor) -> Tensor:
        M = self.grid.M
        scale = self.cond_scale.to(device=u.device, dtype=u.dtype)
        cond = torch.cat([u, xi], dim=-1) / scale
        b = self.branch(cond)                                   # (B, rank)
        # The trunk does not depend on the batch, so it is evaluated once per
        # forward and shared. This is what keeps the M^2 query set affordable.
        t = self.trunk(self.trunk_input.to(u))                  # (M*M, rank)
        logits = self.base.unsqueeze(0) + (
            torch.einsum("bk,pk->bp", b, t) * self.inner_scale
        ).reshape(-1, M, M)
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
