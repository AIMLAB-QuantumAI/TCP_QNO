"""Stinespring transport block: the Givens ladder traced over the environment."""
from __future__ import annotations

from torch import Tensor, nn

from qtro.config import ModelConfig
from qtro.quantum.circuit import ConditionedCircuit


class TransportBlock(nn.Module):
    """Stinespring transport reduced to its diagonal action.

    The input is diagonal and the output is dephased, so only
    T_ij = sum_e |<i,e| U |j,0>|^2 is ever needed. Tracing out the environment
    is what supplies irreversibility: a bare unitary on the system register
    alone is doubly stochastic and cannot concentrate mass.

    Displacement bound: each layer applies V_l then W_l, each a full sweep set
    whose even and odd sweeps move a cell index by at most one bin apiece, so
    probability moves at most 4 * n_layers bins along the window and queue
    axes.

    Known property, not a bug: n_w = n_q = 16 makes the axis only 15 bins
    wide, so at the default n_layers = 8 the bound is 4 * 8 = 32, which
    already exceeds the axis width. The bandedness prior is vacuous at that
    depth, and the tests pin the arithmetic so that a depth at which it binds
    has to be chosen deliberately.
    """

    def __init__(self, grid, cfg: ModelConfig, n_cond_features: int):
        super().__init__()
        self.grid = grid
        self.d_env = cfg.d_env
        self.circuit = ConditionedCircuit(grid, cfg, n_cond_features)

    def forward(self, cond: Tensor,
                source_indices: Tensor | None = None) -> Tensor:
        """Return all columns, or only the requested source columns.

        Selecting ``K`` columns before the Givens ladder changes the dominant
        circuit workspace from ``O(B M^2 d_env)`` to ``O(B M K d_env)``.
        The returned columns are otherwise exactly the same as gathering them
        from the full transport matrix.
        """
        psi = self.circuit(cond, source_indices=source_indices)
        B, _, K = psi.shape
        amp = psi.reshape(B, self.grid.M, self.d_env, K)
        return (amp ** 2).sum(dim=2)
