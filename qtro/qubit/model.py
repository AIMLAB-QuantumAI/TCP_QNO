"""QTRO with its transport swapped for a qubit-native circuit.

``QubitQTRO`` is ``QTRO`` with ``self.transport`` replaced. The reset factor,
the propagator assembly, the column path and the conditioning scale are
inherited, so what the comparison isolates is the gate set and nothing else.
"""
from __future__ import annotations

from torch import Tensor, nn

from qtro.config import ModelConfig
from qtro.model.qtro import N_COND_FEATURES, QTRO
from qtro.qubit.circuit import QubitCircuit


class QubitTransportBlock(nn.Module):
    """``T(i|j,c) = sum_e |<i,e| U_theta(c) |j,0>|^2`` from a qubit circuit."""

    def __init__(self, grid, cfg: ModelConfig, n_cond_features: int,
                 ansatz: str):
        super().__init__()
        self.grid = grid
        self.d_env = cfg.d_env
        self.circuit = QubitCircuit(grid, cfg, n_cond_features, ansatz)

    @staticmethod
    def trace_out(psi: Tensor, M: int, d_env: int) -> Tensor:
        """``psi`` is real ``(B, M*d_env, K, 2)``; returns ``(B, M, K)``."""
        B, _, K, _ = psi.shape
        prob = (psi ** 2).sum(dim=-1)                 # |amp|^2, (B, N, K)
        return prob.reshape(B, M, d_env, K).sum(dim=2)

    def forward(self, cond: Tensor,
                source_indices: Tensor | None = None) -> Tensor:
        psi = self.circuit(cond, source_indices=source_indices)
        return self.trace_out(psi, self.grid.M, self.d_env)


class QubitQTRO(QTRO):
    def __init__(self, grid, cfg: ModelConfig, ansatz: str):
        super().__init__(grid, cfg)
        # Replacing the submodule drops the Givens block's parameters and
        # buffers from the module tree, so the state dict carries only the
        # qubit circuit under the same "transport" prefix.
        self.transport = QubitTransportBlock(
            grid, cfg, N_COND_FEATURES, ansatz)
        self.ansatz = ansatz
