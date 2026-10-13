"""The conditioned unitary built from qubit-native gates.

The layer structure is the one ``qtro.quantum.circuit.ConditionedCircuit``
uses: each layer applies a conditioned block ``V_l(c)`` and then a free block
``W_l(theta_l)``, and the conditioning is re-read at every layer. What differs
is what a block is made of. The Givens ladder rotates cell-adjacent pairs of
the joint index, which is why one sweep moves a cell by at most one bin; here
a block is a circuit on the twelve qubits that binary-encode that index, and a
single two-qubit gate can move a cell by hundreds of bins.

Two ansaetze:

  hea    R_Z R_Y on every qubit, then a CNOT chain 0 -> 1 -> ... -> n-1.
         The textbook hardware-efficient layer; 2n free angles per block.
  dense  one SU(4) on every pair of qubits, 15 angles each. The most
         expressive block that is still a product of two-qubit gates.

All gate matrices for all layers are built before the ladder runs, in one
batched call per angle set, and the ladder is gradient-checkpointed layer by
layer. Building gates outside the checkpoint means the matrix exponentials
are computed once per step rather than once per layer per recompute, and
checkpointing means a dense depth-54 circuit keeps 54 states alive for the
backward pass rather than seven thousand.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from qtro.config import ModelConfig
from qtro.quantum.qubit import log2_exact
from qtro.qubit.simulator import (
    apply_1q,
    apply_2q,
    basis_columns,
    cnot_permutation,
    real_block,
    rz_ry,
    su4,
)

ANSATZE = ("hea", "dense")


def chain_links(n: int) -> list[tuple[int, int]]:
    """CNOT ``(control, target)`` links of the hea entangler: the ladder 0 -> 1 -> ... -> n-1."""
    return [(q, q + 1) for q in range(n - 1)]


def dense_pairs(n: int) -> list[tuple[int, int]]:
    """Every qubit pair, each carrying an SU(4) in the dense ansatz (66 at n = 12)."""
    return [(a, b) for a in range(n) for b in range(a)]


class QubitCircuit(nn.Module):
    def __init__(self, grid, cfg: ModelConfig, n_cond_features: int,
                 ansatz: str):
        super().__init__()
        if ansatz not in ANSATZE:
            raise ValueError(f"ansatz must be one of {ANSATZE}, got {ansatz!r}")
        self.grid = grid
        self.ansatz = ansatz
        self.d_env = cfg.d_env
        self.n_layers = cfg.n_layers
        self.n_sys = log2_exact(grid.M, "grid.M")
        self.n_env = log2_exact(cfg.d_env, "d_env")
        self.n = self.n_sys + self.n_env
        self.n_joint = grid.M * cfg.d_env
        init_scale = float(cfg.qubit_init_scale)

        if ansatz == "hea":
            self.links = chain_links(self.n)
            self.pairs = []
            self.n_angles = 2 * self.n
        else:
            self.links = []
            self.pairs = dense_pairs(self.n)
            self.n_angles = 15 * len(self.pairs)

        self.encoder = nn.Sequential(
            nn.Linear(n_cond_features, 64), nn.Tanh(),
            nn.Linear(64, cfg.d_cond), nn.Tanh(),
        )
        self.trainable = nn.ParameterList([
            nn.Parameter(init_scale * torch.randn(self.n_angles))
            for _ in range(cfg.n_layers)
        ])
        self.cond_heads = nn.ParameterList([
            nn.Parameter(init_scale * torch.randn(self.n_angles, cfg.d_cond))
            for _ in range(cfg.n_layers)
        ])
        self.register_buffer(
            "basis", basis_columns(grid.M, cfg.d_env, grid.device),
            persistent=False)
        # The entangler is a fixed permutation of basis states, so one gather
        # replaces all of its CNOTs in every hea block.
        self.register_buffer(
            "chain_perm", cnot_permutation(self.n, self.links)
            if ansatz == "hea" else torch.arange(self.n_joint),
            persistent=False)
        self.use_checkpoint = True

    def _gates(self, angles: Tensor) -> Tensor:
        """Real-block gate matrices for every layer from ``(B, L, n_angles)``."""
        B, L, _ = angles.shape
        if self.ansatz == "hea":
            n = self.n
            G = rz_ry(angles[..., :n], angles[..., n:])       # (B, L, n, 2, 2)
            return real_block(G)                              # (B, L, n, 4, 4)
        U = su4(angles.reshape(B, L, len(self.pairs), 15))    # (B, L, P, 4, 4)
        return real_block(U.to(torch.complex64))              # (B, L, P, 8, 8)

    def _block(self, psi: Tensor, gates: Tensor) -> Tensor:
        """One block; ``gates`` is ``(B, n, 4, 4)`` for hea, ``(B, P, 8, 8)`` dense."""
        n = self.n
        if self.ansatz == "hea":
            for q in range(n):
                psi = apply_1q(psi, gates[:, q], q, n)
            return psi[:, self.chain_perm]
        for k, (a, b) in enumerate(self.pairs):
            psi = apply_2q(psi, gates[:, k], a, b, n)
        return psi

    def _layer(self, psi: Tensor, cond_gates: Tensor,
               train_gates: Tensor) -> Tensor:
        psi = self._block(psi, cond_gates)      # V_l(c)
        psi = self._block(psi, train_gates)     # W_l(theta_l)
        return psi

    def _basis_columns(self, B: int, device,
                       source_indices: Tensor | None) -> Tensor:
        basis = self.basis.to(device)                          # (N, M, 2)
        if source_indices is None:
            return basis.unsqueeze(0).expand(B, -1, -1, -1)
        idx = source_indices.to(device)
        if idx.ndim == 1:
            idx = idx.unsqueeze(0).expand(B, -1)
        gathered = basis.unsqueeze(0).expand(B, -1, -1, -1).gather(
            2, idx[:, None, :, None].expand(-1, self.n_joint, -1, 2))
        return gathered

    def forward(self, cond: Tensor,
                source_indices: Tensor | None = None) -> Tensor:
        """Real ``psi`` of shape ``(B, M * d_env, K, 2)``, unit-norm columns."""
        B = cond.shape[0]
        h = self.encoder(cond)
        cond_angles = torch.stack(
            [h @ head.T for head in self.cond_heads], dim=1)   # (B, L, A)
        train_angles = torch.stack(list(self.trainable), dim=0)
        train_angles = train_angles.unsqueeze(0).expand(B, -1, -1)
        cond_gates = self._gates(cond_angles)
        train_gates = self._gates(train_angles)

        psi = self._basis_columns(B, cond.device, source_indices).contiguous()
        use_ckpt = (self.training and self.use_checkpoint
                    and torch.is_grad_enabled())
        for layer in range(self.n_layers):
            cg, tg = cond_gates[:, layer], train_gates[:, layer]
            if use_ckpt:
                psi = checkpoint(self._layer, psi, cg, tg, use_reentrant=False)
            else:
                psi = self._layer(psi, cg, tg)
        return psi
