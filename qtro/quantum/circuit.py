"""The conditioned Givens ladder: the circuit the transport block evaluates."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from qtro.config import ModelConfig
from qtro.quantum.givens import apply_sweep
from qtro.quantum.pairs import build_sweeps


class ConditionedCircuit(nn.Module):
    """U_theta(u, xi) as a Givens ladder.

    Each layer applies a conditioned block V_l(u, xi) and then a free block
    W_l(theta_l). The conditioning is re-injected at every layer rather than
    only at the input: that is what lets one trained model represent the whole
    family of operators rather than a single member of it.
    """

    def __init__(self, grid, cfg: ModelConfig, n_cond_features: int):
        super().__init__()
        self.grid = grid
        self.d_env = cfg.d_env
        self.n_layers = cfg.n_layers
        self.n_joint = grid.M * cfg.d_env

        sweeps = build_sweeps(grid, cfg.d_env)
        self.offsets, offset = [], 0
        for _, lo, _ in sweeps:
            self.offsets.append((offset, lo.numel()))
            offset += lo.numel()
        self.n_pairs = offset

        for k, (_, lo, hi) in enumerate(sweeps):
            self.register_buffer(f"lo_{k}", lo)
            self.register_buffer(f"hi_{k}", hi)
            paired = torch.zeros(self.n_joint, dtype=torch.bool, device=lo.device)
            paired[lo] = True
            paired[hi] = True
            self.register_buffer(
                f"untouched_{k}", torch.arange(
                    self.n_joint, device=lo.device)[~paired], persistent=False)
        self.n_sweeps = len(sweeps)

        # The |j> tensor |0>_E input is constant across every query. Keeping a
        # non-persistent template avoids rebuilding and scattering into a
        # multi-megabyte zero tensor for every horizon step without enlarging
        # checkpoints or changing their state-dict keys.
        basis = torch.zeros(self.n_joint, self.grid.M, device=grid.device)
        cols = torch.arange(self.grid.M, device=grid.device)
        basis[cols * self.d_env, cols] = 1.0
        self.register_buffer("basis_columns", basis, persistent=False)

        self.encoder = nn.Sequential(
            nn.Linear(n_cond_features, 64), nn.Tanh(),
            nn.Linear(64, cfg.d_cond), nn.Tanh(),
        )
        self.trainable = nn.ParameterList([
            nn.Parameter(0.01 * torch.randn(self.n_pairs))
            for _ in range(cfg.n_layers)
        ])
        self.cond_heads = nn.ParameterList([
            nn.Parameter(0.01 * torch.randn(self.n_pairs, cfg.d_cond))
            for _ in range(cfg.n_layers)
        ])

    def _basis_columns(
            self, B: int, device, dtype,
            source_indices: Tensor | None = None) -> Tensor:
        basis = self.basis_columns.to(device=device, dtype=dtype)
        if source_indices is None:
            return basis.unsqueeze(0).expand(B, -1, -1)
        if source_indices.ndim != 2 or source_indices.shape[0] != B:
            raise ValueError("source_indices must have shape (B, K)")
        if source_indices.dtype != torch.long:
            raise ValueError("source_indices must have dtype torch.long")
        if source_indices.device != device:
            source_indices = source_indices.to(device)
        if bool(((source_indices < 0) |
                 (source_indices >= self.grid.M)).any()):
            raise ValueError("source_indices contains an invalid cell index")
        K = source_indices.shape[1]
        selected = torch.zeros(B, self.n_joint, K, device=device, dtype=dtype)
        rows = source_indices * self.d_env
        selected.scatter_(1, rows.unsqueeze(1), 1.0)
        return selected

    def forward(self, cond: Tensor,
                source_indices: Tensor | None = None) -> Tensor:
        B = cond.shape[0]
        h = self.encoder(cond)
        psi = self._basis_columns(
            B, cond.device, cond.dtype, source_indices=source_indices)
        for layer in range(self.n_layers):
            cond_angles = h @ self.cond_heads[layer].T
            train_angles = self.trainable[layer].unsqueeze(0).expand(B, -1)
            for angles in (cond_angles, train_angles):
                for k in range(self.n_sweeps):
                    start, count = self.offsets[k]
                    psi = apply_sweep(
                        psi, getattr(self, f"lo_{k}"), getattr(self, f"hi_{k}"),
                        angles[:, start:start + count],
                        getattr(self, f"untouched_{k}"),
                    )
        return psi
