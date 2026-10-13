"""The Givens ladder as a circuit on qubits.

The joint index ``j * d_env + e`` of the transport register is a binary
number, and its bits are qubits: the environment on the low bits, then the
phase, the queue and the window (:func:`registers`). With the shipped grid
that is nine system qubits and three environment qubits.

An even sweep along an axis rotates pairs ``(2m, 2m+1)`` that differ in that
axis's least significant qubit only, so it is one ``R_Y`` on that qubit whose
angle is selected by the other eleven: a uniformly controlled rotation with
``2**11`` angles, which is exactly the number of pairs the sweep has. An odd
sweep pairs ``(2m+1, 2m+2)``; the axis register's decrement maps those onto
the even pairs, so it is the same rotation conjugated by the decrement, a
permutation of basis states (an adder). The rotation reads the axis's own
high bits in either case, and the pair the decrement wraps from the top of
the axis to its bottom has angle zero, which is how the ladder has no such
pair.

:func:`apply_ucr` and :func:`register_shift_permutation` build a sweep from
these primitives on a complex statevector; :func:`qtro.quantum.givens.apply_sweep`
is the same linear map computed without forming gate matrices, and
``tests/test_qubit_reading.py`` checks that the two agree on the shipped grid.
The training and inference code keeps the sweep form because it is the
efficient simulation.

Gate count: a uniformly controlled rotation on ``n`` qubits compiles into
``2**(n-1)`` CNOTs and as many ``R_Y`` (Mottonen, Vartiainen, Bergholm and
Salomaa, 2004), so a sweep costs ``Theta(M d_E)`` gates. The model is exactly
simulable at the cost training already pays; nothing here claims a speedup.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

AXIS_OF = {"w": "window", "q": "queue", "phase": "phase", "env": "env"}


def log2_exact(value: int, what: str) -> int:
    bits = int(round(math.log2(value)))
    if 2 ** bits != value:
        raise ValueError(f"{what} must be a power of two for a qubit "
                         f"encoding, got {value}")
    return bits


def registers(grid, d_env: int) -> dict[str, list[int]]:
    """Which bits of the joint index belong to which coordinate.

    The joint index is ``j * d_env + e`` with ``j = (w * n_q + q) * n_phase
    + s``, so the environment is the low bits, then the phase, then the
    queue, then the window; each list runs least significant bit first.
    """
    widths = (("env", log2_exact(d_env, "d_env")),
              ("phase", log2_exact(grid.n_phase, "n_phase")),
              ("queue", log2_exact(grid.n_q, "n_q")),
              ("window", log2_exact(grid.n_w, "n_w")))
    out, lo = {}, 0
    for name, width in widths:
        out[name] = list(range(lo, lo + width))
        lo += width
    return out


def n_qubits(regs: dict[str, list[int]]) -> int:
    return sum(len(bits) for bits in regs.values())


def sweep_gate_layout(regs: dict[str, list[int]],
                      sweep: str) -> tuple[int, list[int]]:
    """Target qubit of a sweep and its control qubits, most significant first.

    The controls are every other qubit; their integer, read from the pair's
    lower index, is the position of the pair's angle in the order
    :func:`qtro.quantum.pairs.build_sweeps` enumerates that sweep.
    """
    axis = AXIS_OF[sweep.split("_")[0]]
    target = regs[axis][0]
    controls = sorted((b for bits in regs.values() for b in bits
                       if b != target), reverse=True)
    return target, controls


def register_shift_permutation(bits: list[int], delta: int,
                               n: int) -> Tensor:
    """Basis permutation of ``|r> -> |r + delta mod 2**w>`` on one register.

    ``bits`` are the register's qubits, least significant first. Returns
    ``perm`` with ``out[..., x, ...] = in[..., perm[x], ...]``: the amplitude
    that lands on register value ``r`` came from ``r - delta``.
    """
    x = torch.arange(2 ** n)
    r = torch.zeros_like(x)
    for i, b in enumerate(bits):
        r |= ((x >> b) & 1) << i
    src = (r - delta) % (2 ** len(bits))
    perm = x.clone()
    for i, b in enumerate(bits):
        perm &= ~(1 << b)
        perm |= ((src >> i) & 1) << b
    return perm


def apply_ucr(state: Tensor, gates: Tensor, target: int,
              controls: list[int], n: int) -> Tensor:
    """A uniformly controlled one-qubit gate on a ``(B, 2**n, K)`` state.

    ``gates`` is ``(B, 2**k, 2, 2)``, one gate per value of the ``k``
    controls (``controls[0]`` most significant). The control axes are moved
    next to the batch axis so every control value selects its own gate.
    """
    B, N, K = state.shape
    k = len(controls)
    if gates.shape[:2] != (B, 2 ** k):
        raise ValueError(f"gates must be (B, 2**{k}, 2, 2), got "
                         f"{tuple(gates.shape)}")
    if target in controls:
        raise ValueError("the target cannot also be a control")
    v = state.reshape(B, *([2] * n), K)
    dim_of = lambda q: 1 + (n - 1 - q)
    ctrl = [dim_of(q) for q in controls]
    tgt = dim_of(target)
    rest = [d for d in range(1, n + 1) if d not in ctrl and d != tgt]
    perm = [0] + ctrl + [tgt] + rest + [n + 1]
    v = v.permute(*perm).reshape(B * 2 ** k, 2, -1)
    out = torch.bmm(gates.reshape(B * 2 ** k, 2, 2), v)
    out = out.reshape(B, *([2] * k), 2, *([2] * len(rest)), K)
    inv = [0] * len(perm)
    for i, q in enumerate(perm):
        inv[q] = i
    return out.permute(*inv).reshape(B, N, K)


def r_y(theta: Tensor) -> Tensor:
    """``[[cos, -sin], [sin, cos]]`` on ``(...,)`` angles, the Givens rotation
    of :func:`qtro.quantum.givens.apply_sweep` as a one-qubit gate, shape
    ``(..., 2, 2)``."""
    c, s = torch.cos(theta), torch.sin(theta)
    return torch.stack([torch.stack([c, -s], -1),
                        torch.stack([s, c], -1)], -2)


def sweep_as_circuit(state: Tensor, name: str, lo: Tensor, angles: Tensor,
                     regs: dict[str, list[int]]) -> Tensor:
    """One Givens sweep from qubit gates: (decrement) UCR (increment).

    ``state`` is complex ``(B, 2**n, K)``; ``lo`` and ``angles`` (``(B, P)``)
    are the sweep's pair list and angles in ``build_sweeps`` order.
    """
    n = n_qubits(regs)
    target, controls = sweep_gate_layout(regs, name)
    slot = torch.zeros_like(lo)
    for bit in controls:
        slot = slot * 2 + ((lo >> bit) & 1)
    B = angles.shape[0]
    table = torch.zeros(B, 2 ** len(controls), dtype=angles.dtype,
                        device=angles.device)
    table[:, slot] = angles
    gates = r_y(table).to(state.dtype)
    odd = name.endswith("_odd")
    if odd:
        bits = regs[AXIS_OF[name.split("_")[0]]]
        state = state[:, register_shift_permutation(bits, -1, n)]
    state = apply_ucr(state, gates, target, controls, n)
    if odd:
        state = state[:, register_shift_permutation(bits, +1, n)]
    return state


def gate_count(regs: dict[str, list[int]], n_sweeps: int = 7) -> dict:
    """Two-qubit gate count of one block, from the Mottonen decomposition."""
    n = n_qubits(regs)
    per_sweep = 2 ** (n - 1)
    return {"qubits": n, "system_qubits": n - len(regs["env"]),
            "env_qubits": len(regs["env"]),
            "cnot_per_sweep": per_sweep, "ry_per_sweep": per_sweep,
            "cnot_per_block": per_sweep * n_sweeps}
