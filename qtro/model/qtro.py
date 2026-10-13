"""The QTRO operator: transport, then the protocol reset, on dephased inputs."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from qtro.config import ModelConfig
from qtro.model.reset import ResetBlock
from qtro.model.transport import TransportBlock

N_COND_FEATURES = 6
DEFAULT_COND_SCALE = (1.0, 100.0, 1.0e4, 0.1, 50.0, 0.01)


class QTRO(nn.Module):
    """Lambda_theta(u, xi) = D R T D: the dephased product of the protocol
    reset and the Givens transport.

    Because the dephasing projects onto the classical sector, the whole map
    reduces exactly to a product of column-stochastic matrices. The full
    density-matrix path is kept only in tests, where it verifies that this
    reduction is correct rather than assumed.
    """

    def __init__(self, grid, cfg: ModelConfig):
        super().__init__()
        self.grid = grid
        # Kept in the state dict so a checkpoint carries the exact feature
        # normalisation used during training. Training replaces the default
        # with maxima measured on the training split; held-out data must
        # never choose its own scale.
        self.register_buffer(
            "cond_scale", torch.tensor(DEFAULT_COND_SCALE, dtype=torch.float32)
        )
        self.transport = TransportBlock(grid, cfg, N_COND_FEATURES)
        self.reset = ResetBlock(grid, cfg, N_COND_FEATURES)

    def set_conditioning_scale(self, scale: Tensor) -> None:
        scale = torch.as_tensor(scale, dtype=self.cond_scale.dtype,
                                device=self.cond_scale.device)
        if scale.shape != (N_COND_FEATURES,) or not bool(torch.all(scale > 0)):
            raise ValueError(
                f"conditioning scale must be {N_COND_FEATURES} positive values, "
                f"got shape={tuple(scale.shape)}")
        self.cond_scale.copy_(scale)

    def _cond(self, u: Tensor, xi: Tensor) -> Tensor:
        scale = self.cond_scale.to(device=u.device, dtype=u.dtype)
        return torch.cat([u, xi], dim=-1) / scale

    def propagator_columns(self, u: Tensor, xi: Tensor,
                           source_indices: Tensor) -> Tensor:
        """Evaluate exact channel columns for selected source cells only.

        This preserves the learned transport and reset maps; it is an
        execution optimisation, not a separately trained approximation.
        Output shape is ``(B, M, K)`` for input indices ``(B, K)``.
        """
        if source_indices.ndim != 2 or source_indices.shape[0] != u.shape[0]:
            raise ValueError("source_indices must have shape (B, K)")
        source_indices = source_indices.to(device=u.device, dtype=torch.long)
        cond = self._cond(u, xi)
        transport = self.transport(cond, source_indices=source_indices)
        return self.reset.apply_channel(cond, transport)

    def propagator(self, u: Tensor, xi: Tensor) -> Tensor:
        cond = self._cond(u, xi)
        # The sparse TCP reset is applied once to the transport columns rather
        # than materialising and multiplying a dense reset matrix.
        return self.reset.apply_channel(cond, self.transport(cond))

    def step(self, p: Tensor, u: Tensor, xi: Tensor) -> Tensor:
        return torch.einsum("bij,bj->bi", self.propagator(u, xi), p)

    def rollout(self, p0: Tensor, u_seq: Tensor, xi: Tensor) -> Tensor:
        if u_seq.ndim != 3 or u_seq.shape[0] != p0.shape[0]:
            raise ValueError("u_seq must have shape (B, H, action_dim)")
        out, p = [], p0
        # The common control case scores one fixed action over the whole
        # horizon. The propagator depends on (u, xi), not on p, so building it
        # once is mathematically identical and keeps the complete gradient
        # graph.
        repeated = (u_seq.shape[1] <= 1 or torch.equal(
            u_seq[:, 1:], u_seq[:, :1].expand(-1, u_seq.shape[1] - 1, -1)))
        propagator = self.propagator(u_seq[:, 0], xi) if repeated else None
        for h in range(u_seq.shape[1]):
            matrix = (propagator if propagator is not None
                      else self.propagator(u_seq[:, h], xi))
            p = torch.bmm(matrix, p.unsqueeze(-1)).squeeze(-1)
            out.append(p)
        return torch.stack(out, dim=1)
