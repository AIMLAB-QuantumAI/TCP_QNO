"""Diagonal encoding, dephasing, and the two statements the model half rests on.

A diagonal-preserving CPTP map can realise any column-stochastic T exactly,
and a unitary-only, no-environment circuit can only ever realise a doubly
stochastic T, so it can never decrease Shannon entropy or move mass off the
uniform distribution. ``tests/test_propositions.py`` checks both as
executable tests.

Real float32 throughout: every object here (density matrix, Kraus family,
stochastic matrix) is real-valued, since the circuit is real-orthogonal. No
complex dtype is used or needed.
"""
from __future__ import annotations

import torch
from torch import Tensor


def encode(p: Tensor) -> Tensor:
    """Probability vector to diagonal density matrix: (..., M) -> (..., M, M)."""
    return torch.diag_embed(p)


def decode(rho: Tensor) -> Tensor:
    """Diagonal of a density matrix: (..., M, M) -> (..., M)."""
    return torch.diagonal(rho, dim1=-2, dim2=-1)


def dephase(rho: Tensor) -> Tensor:
    """Dephasing: off-diagonal entries carry no cell-probability meaning."""
    return torch.diag_embed(decode(rho))


def kraus_from_stochastic(T: Tensor) -> Tensor:
    """The Kraus family K_ij = sqrt(T_ij) |i><j| of a column-stochastic T.

    Every column-stochastic matrix admits this family, so constraining the
    model to diagonal-preserving CPTP maps costs no expressive capacity at
    the level of classical Markov dynamics: every finite-state Markov chain
    is inside the class the model is restricted to. The operators are real,
    which is why a real-orthogonal dilation suffices in the transport block.

    Shape: (M, M) -> (M*M, M, M). This is M^2 operators, each an (M, M)
    matrix, M^4 floats in all: at M = 512 that is 256 GiB in float32, which
    no single machine materialises. The statement is a dimension-independent
    linear-algebra identity, so it is verified at small M in the tests.
    """
    M = T.shape[0]
    root = torch.sqrt(T.clamp(min=0.0))
    K = torch.zeros(M * M, M, M, dtype=T.dtype, device=T.device)
    i_idx = torch.arange(M, device=T.device).repeat_interleave(M)
    j_idx = torch.arange(M, device=T.device).repeat(M)
    K[torch.arange(M * M, device=T.device), i_idx, j_idx] = root[i_idx, j_idx]
    return K


def apply_kraus(K: Tensor, rho: Tensor) -> Tensor:
    """Phi(rho) = sum_k K_k @ rho @ K_k^T (K real, so K^dagger == K^T).

    Any map of this Kraus-sum form is completely positive, but that guarantee
    is only as good as the code actually being in that form, so the tests
    check complete positivity through the Choi matrix over all M^2 basis
    matrices rather than only positivity on diagonal inputs: plain
    transposition, which is positive but not completely positive, passes the
    diagonal-only check and fails the Choi one.
    """
    return torch.einsum("kij,jl,kml->im", K, rho, K)
