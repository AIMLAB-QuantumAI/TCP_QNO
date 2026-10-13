"""The rotation pairs of each sweep: cell-adjacent along one axis, plus the environment."""
from __future__ import annotations

import torch
from torch import Tensor


def _joint(cell: Tensor, env: Tensor, d_env: int) -> Tensor:
    return cell * d_env + env


def build_sweeps(grid, d_env: int) -> list[tuple[str, Tensor, Tensor]]:
    """Enumerate the disjoint two-level rotation pairs of each sweep.

    Restricting entangling rotations to cell-adjacent pairs is what makes the
    depth-L displacement bound provable rather than approximate: with a binary
    qubit encoding, a two-qubit gate can move a cell index by many bins. Pairs
    within a sweep are disjoint, so a sweep applies as one batched update.
    """
    device = grid.device
    n_w, n_q, n_ph = grid.n_w, grid.n_q, grid.n_phase
    a = torch.arange(n_w, device=device)
    b = torch.arange(n_q, device=device)
    c = torch.arange(n_ph, device=device)
    e = torch.arange(d_env, device=device)

    def grid4(av, bv, cv, ev):
        A, B, C, E = torch.meshgrid(av, bv, cv, ev, indexing="ij")
        return A.reshape(-1), B.reshape(-1), C.reshape(-1), E.reshape(-1)

    sweeps: list[tuple[str, Tensor, Tensor]] = []

    for name, start in (("w_even", 0), ("w_odd", 1)):
        # max(..., start) guards n_w <= 1: torch.arange raises instead of
        # returning empty when stop < start (unlike start == stop, which is
        # fine), so e.g. n_w=1 would otherwise crash building w_odd.
        av = torch.arange(start, max(n_w - 1, start), 2, device=device)
        A, B, C, E = grid4(av, b, c, e)
        lo = _joint(grid.cell_index(A, B, C), E, d_env)
        hi = _joint(grid.cell_index(A + 1, B, C), E, d_env)
        sweeps.append((name, lo, hi))

    for name, start in (("q_even", 0), ("q_odd", 1)):
        bv = torch.arange(start, max(n_q - 1, start), 2, device=device)
        A, B, C, E = grid4(a, bv, c, e)
        lo = _joint(grid.cell_index(A, B, C), E, d_env)
        hi = _joint(grid.cell_index(A, B + 1, C), E, d_env)
        sweeps.append((name, lo, hi))

    A, B, C, E = grid4(a, b, torch.zeros(1, dtype=torch.long, device=device), e)
    lo = _joint(grid.cell_index(A, B, C), E, d_env)
    hi = _joint(grid.cell_index(A, B, C + 1), E, d_env)
    sweeps.append(("phase", lo, hi))

    for name, start in (("env_even", 0), ("env_odd", 1)):
        # Same guard as above: d_env=1 must yield two empty env sweeps, not
        # a crash from torch.arange(1, 0, 2).
        ev = torch.arange(start, max(d_env - 1, start), 2, device=device)
        A, B, C, E = grid4(a, b, c, ev)
        cell = grid.cell_index(A, B, C)
        sweeps.append((name, _joint(cell, E, d_env), _joint(cell, E + 1, d_env)))

    return sweeps
