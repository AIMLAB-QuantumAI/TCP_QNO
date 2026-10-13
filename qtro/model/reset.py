"""The protocol reset: learned event intensities, destinations fixed by TCP."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from qtro.config import ModelConfig


def reset_matrix(gamma: Tensor, targets: Tensor, M: int) -> Tensor:
    """The reset channel as a column-stochastic matrix.

    Mass leaves cell i at the learned rate gamma_z(i) and reappears at the
    target pi_z(i) that the protocol fixes. Completeness holds cell by cell, so
    pi_z may be many-to-one, which is exactly the entropy reduction a unitary
    cannot supply.
    """
    B = gamma.shape[0]
    stay = 1.0 - gamma.sum(-1)
    T = torch.diag_embed(stay)
    n_events = targets.shape[0]
    src = torch.arange(M, device=gamma.device).expand(B, M)
    for z in range(n_events):
        dst = targets[z].expand(B, M)
        T = T.index_put((torch.arange(B, device=gamma.device)
                         .unsqueeze(1).expand(B, M), dst, src),
                        gamma[:, :, z], accumulate=True)
    return T


class ResetBlock(nn.Module):
    """Reset block of the operator.

    Only the intensities are learned; the targets come from the protocol. The
    number of trainable outputs scales with the number of event types, not
    with M^2.
    """

    def __init__(self, grid, cfg: ModelConfig, n_cond_features: int):
        super().__init__()
        self.grid = grid
        self.register_buffer("targets", grid.reset_targets())
        feats = torch.stack([
            grid.w_rep / grid.w_rep.max(),
            grid.q_rep / grid.q_rep.max(),
            grid.phase_rep.float(),
            grid.rtt_rep / grid.rtt_rep.max(),
        ], dim=1)
        self.register_buffer("cell_features", feats)
        self.max_probability = float(cfg.reset_max_probability)
        if not 0.0 < self.max_probability <= 1.0:
            raise ValueError("reset_max_probability must lie in (0, 1]")
        self.net = nn.Sequential(
            nn.Linear(feats.shape[1] + n_cond_features, cfg.intensity_hidden),
            nn.Tanh(),
            nn.Linear(cfg.intensity_hidden, cfg.intensity_hidden), nn.Tanh(),
            nn.Linear(cfg.intensity_hidden, 3),
        )

    def _inputs(self, cond: Tensor) -> Tensor:
        B, M = cond.shape[0], self.grid.M
        cells = self.cell_features.unsqueeze(0).expand(B, M, -1)
        ctx = cond.unsqueeze(1).expand(B, M, cond.shape[1])
        return torch.cat([cells, ctx], dim=-1)

    def intensities(self, cond: Tensor) -> Tensor:
        """Per-cell loss and timeout probabilities, ``(B, M, 2)``."""
        logits = self.net(self._inputs(cond))
        return self.max_probability * torch.softmax(logits, dim=-1)[..., 1:]

    @staticmethod
    def _state_scale(mass: Tensor, values: Tensor) -> Tensor:
        """Broadcast ``(B, M)`` state mass over trailing value dimensions."""
        return mass.reshape(*mass.shape, *([1] * (values.ndim - 2)))

    def apply_channel(self, cond: Tensor, values: Tensor) -> Tensor:
        """Apply the reset channel without materialising its dense matrix.

        ``values`` is a batch of state vectors ``(B, M)`` or of matrices
        ``(B, M, C)`` whose destination axis is dimension one. The result is
        exactly ``self(cond) @ values``. TCP reset destinations are sparse, so
        the scatter-add costs ``O(B Z M C)`` instead of the dense
        ``O(B M^2 C)`` product.
        """
        if values.ndim < 2 or values.shape[0] != cond.shape[0] or \
                values.shape[1] != self.grid.M:
            raise ValueError(
                "values must have shape (B, M, ...) matching cond and grid")
        gamma = self.intensities(cond)
        stay = 1.0 - gamma.sum(-1)
        out = values * self._state_scale(stay, values)
        for event in range(gamma.shape[-1]):
            weighted = values * self._state_scale(gamma[:, :, event], values)
            out = out.index_add(1, self.targets[event], weighted)
        return out

    def forward(self, cond: Tensor) -> Tensor:
        return reset_matrix(self.intensities(cond), self.targets, self.grid.M)
