"""One sweep of disjoint real Givens rotations on a batch of state columns."""
from __future__ import annotations

import torch
from torch import Tensor


def apply_sweep(psi: Tensor, lo: Tensor, hi: Tensor, angles: Tensor,
                untouched: Tensor | None = None) -> Tensor:
    """Apply one sweep of disjoint real Givens rotations.

    psi is (B, N, C): B parameter settings, N joint-register amplitudes, C
    columns propagated together. Only M columns are ever needed because the
    input is diagonal and the output is dephased, so the whole transport block
    costs one pass over an (N, M) block rather than an (N, N) matrix.
    """
    cos = torch.cos(angles).unsqueeze(-1)
    sin = torch.sin(angles).unsqueeze(-1)
    a = psi[:, lo, :]
    b = psi[:, hi, :]
    # Cloning the whole (B, N, C) tensor and overwriting every paired row
    # writes the large tensor twice, and most sweeps cover nearly all rows. A
    # fresh output with a precomputed untouched set writes every row exactly
    # once and is the same differentiable map. The clone fallback remains for
    # callers that do not provide the precomputed complement.
    out = psi.clone() if untouched is None else torch.empty_like(psi)
    out[:, lo, :] = cos * a - sin * b
    out[:, hi, :] = sin * a + cos * b
    if untouched is not None and untouched.numel():
        out[:, untouched, :] = psi[:, untouched, :]
    return out


def sweep_matrix(n: int, lo: Tensor, hi: Tensor, angles: Tensor) -> Tensor:
    """Dense reference for one sweep. Test-only; independent of apply_sweep."""
    U = torch.eye(n, dtype=angles.dtype, device=angles.device)
    cos, sin = torch.cos(angles), torch.sin(angles)
    U[lo, lo] = cos
    U[hi, hi] = cos
    U[lo, hi] = -sin
    U[hi, lo] = sin
    return U
