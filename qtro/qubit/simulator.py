"""Statevector primitives on a real ``(B, 2**n, K, 2)`` state.

The state is stored as a real tensor whose last axis is (re, im). Qubit ``q``
is the bit worth ``2**q`` of the joint index, so with ``joint = j * d_env + e``
the environment occupies the low qubits and the system the high ones,
matching ``ConditionedCircuit.basis_columns``.

A k-qubit gate is one real GEMM. Its complex ``(2**k x 2**k)`` matrix becomes
the real block matrix ``[[Re, -Im], [Im, Re]]`` interleaved on the (re, im)
axis, the target qubit axes and the (re, im) axis are permuted to the front so
the state reads ``(B, 2**(k+1), rest)``, and ``bmm`` does the rest. That is
two copies and one ``sgemm`` per gate.

Why not complex tensors: on the hardware the reported runs used, ``bmm`` of a
``(1, 2, 2)`` complex gate against a ``(1, 2, R)`` state ran at 1.08 ms for a
16 MB state, twenty times a plain copy, while the same arithmetic as a real
``4x4`` GEMM ran at 0.16 ms. With seven thousand gates in a deep forward that
ratio is the whole step time.

``B`` is the batch of conditioning vectors and ``K`` the number of source
columns evaluated at once; every gate is batched over ``B`` so the conditioned
blocks, whose angles depend on the batch element, cost no more than the free
ones.
"""
from __future__ import annotations

import torch
from torch import Tensor

# Pauli matrices, complex128 so that su4 is built in double and cast later.
_I = torch.eye(2, dtype=torch.complex128)
_X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
_Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
_Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)
_PAULI = (_I, _X, _Y, _Z)


def _pauli_pair_generators() -> Tensor:
    """The fifteen sigma_i (x) sigma_j with (i, j) != (0, 0), shape (15,4,4)."""
    gens = []
    for i in range(4):
        for j in range(4):
            if i == 0 and j == 0:
                continue
            gens.append(torch.kron(_PAULI[i], _PAULI[j]))
    return torch.stack(gens)


_GENERATORS = _pauli_pair_generators()


def su4(theta: Tensor) -> Tensor:
    """``exp(i sum_k theta_k G_k)`` over the Pauli-pair generators.

    ``theta`` is ``(..., 15)`` real; the result is ``(..., 4, 4)`` complex128.
    The generators are Hermitian, so the exponent is anti-Hermitian and the
    result is unitary for every ``theta``. Batch every gate of every layer
    into one call: ``matrix_exp`` on a few thousand 4x4 matrices costs about
    what it costs on sixty-six.
    """
    gens = _GENERATORS.to(theta.device)
    H = torch.einsum("...k,kij->...ij", theta.to(torch.complex128), gens)
    return torch.linalg.matrix_exp(1j * H)


def rz_ry(phi: Tensor, theta: Tensor) -> Tensor:
    """``R_Z(phi) R_Y(theta)`` as one 2x2 gate, shape ``(..., 2, 2)`` complex64.

    ``R_Y`` acts first. Fusing the pair halves the gate count of the
    hardware-efficient block without changing what it can express.
    """
    c = torch.cos(theta / 2)
    s = torch.sin(theta / 2)
    ry = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)
    ry = ry.to(torch.complex64)
    e = torch.exp(-0.5j * phi.to(torch.complex64))
    rz = torch.zeros_like(ry)
    rz[..., 0, 0] = e
    rz[..., 1, 1] = e.conj()
    return rz @ ry


def cnot() -> Tensor:
    """CNOT on ``(control, target)`` as a 4x4 in the ``(c, t)`` bit order."""
    U = torch.zeros(4, 4, dtype=torch.complex64)
    U[0, 0] = U[1, 1] = 1.0   # control 0: target unchanged
    U[2, 3] = U[3, 2] = 1.0   # control 1: target flipped
    return U


def cnot_permutation(n_qubits: int, links: list[tuple[int, int]]) -> Tensor:
    """Index permutation of a sequence of CNOTs on the computational basis.

    ``links`` is applied in order as ``(control, target)`` pairs. Any CNOT
    circuit is a fixed permutation of basis states, so a whole entangling
    block is one gather on the joint index rather than one gate per link.
    Returns ``perm`` with ``out[..., x, ...] = in[..., perm[x], ...]``.
    """
    N = 2 ** n_qubits
    x = torch.arange(N)
    img = x.clone()
    for c, t in links:
        control = (img >> c) & 1
        img = img ^ (control << t)
    inv = torch.empty_like(img)
    inv[img] = x
    return inv


def cnot_chain_permutation(n_qubits: int) -> Tensor:
    """The linear chain ``CNOT(0,1) CNOT(1,2) ... CNOT(n-2,n-1)``."""
    return cnot_permutation(n_qubits, [(q, q + 1) for q in range(n_qubits - 1)])


def real_block(G: Tensor) -> Tensor:
    """Complex ``(..., d, d)`` gate as the real ``(..., 2d, 2d)`` block matrix.

    Row and column index ``2 * i + r`` with ``r = 0`` real, ``1`` imaginary,
    so that it acts on a state whose (re, im) axis is interleaved after each
    qubit bit: ``[[Re, -Im], [Im, Re]]`` on every complex entry.
    """
    d = G.shape[-1]
    lead = G.shape[:-2]
    M = torch.empty(*lead, d, 2, d, 2, dtype=G.real.dtype, device=G.device)
    M[..., :, 0, :, 0] = G.real
    M[..., :, 0, :, 1] = -G.imag
    M[..., :, 1, :, 0] = G.imag
    M[..., :, 1, :, 1] = G.real
    return M.reshape(*lead, 2 * d, 2 * d)


def to_real(state_c: Tensor) -> Tensor:
    """``(B, N, K)`` complex -> ``(B, N, K, 2)`` real."""
    return torch.view_as_real(state_c).contiguous()


def to_complex(state_r: Tensor) -> Tensor:
    """``(B, N, K, 2)`` real -> ``(B, N, K)`` complex."""
    return torch.view_as_complex(state_r.contiguous())


def basis_columns(M: int, d_env: int, device) -> Tensor:
    """``|j> (x) |0>_E`` for every source cell, real, shape ``(M*d_env, M, 2)``."""
    cols = torch.zeros(M * d_env, M, 2, dtype=torch.float32, device=device)
    j = torch.arange(M, device=device)
    cols[j * d_env, j, 0] = 1.0
    return cols


def _split_k_outer(go: Tensor, v: Tensor) -> Tensor:
    """``sum_r go[b,i,r] v[b,j,r]`` as a batch of well-shaped GEMMs.

    Written as ``bmm(go, v^T)`` this is a ``(D, R) @ (R, D)`` product with
    ``D = 4`` or ``8`` and ``R`` in the millions, and cuBLAS answers it with a
    kernel built for large outputs: 20 ms for a 16 MB operand, a hundred times
    the copy time. Splitting ``R`` into ``R1`` batches of ``R / R1`` turns it
    into ``R1`` products each with a reasonable inner dimension, then one
    small sum. Measured at 0.17 ms for the same operand.
    """
    B, D, R = go.shape
    R1 = 256
    while R % R1:
        R1 //= 2
    R2 = R // R1
    a = go.reshape(B, D, R1, R2).permute(0, 2, 1, 3).reshape(B * R1, D, R2)
    b = v.reshape(B, D, R1, R2).permute(0, 2, 3, 1).reshape(B * R1, R2, D)
    return torch.bmm(a, b).reshape(B, R1, D, D).sum(1)


class _GateBmm(torch.autograd.Function):
    """``bmm(gate, v)`` with the gate gradient computed by ``_split_k_outer``.

    The forward and the state gradient are ordinary ``bmm`` calls of the same
    shape and are fast. Only the gate gradient has the pathological shape, so
    only it is routed elsewhere.
    """

    @staticmethod
    def forward(ctx, gate: Tensor, v: Tensor) -> Tensor:
        ctx.save_for_backward(gate, v)
        return torch.bmm(gate, v)

    @staticmethod
    def backward(ctx, go: Tensor):
        gate, v = ctx.saved_tensors
        grad_v = torch.bmm(gate.transpose(1, 2), go) \
            if ctx.needs_input_grad[1] else None
        grad_gate = _split_k_outer(go, v) \
            if ctx.needs_input_grad[0] else None
        return grad_gate, grad_v


def gate_bmm(gate: Tensor, v: Tensor) -> Tensor:
    return _GateBmm.apply(gate, v)


def apply_1q(state: Tensor, gate: Tensor, q: int, n_qubits: int) -> Tensor:
    """Apply a batched real-block 4x4 ``gate`` (``(B, 4, 4)``) to qubit ``q``."""
    B, N, K, _ = state.shape
    hi, lo = 2 ** (n_qubits - 1 - q), 2 ** q
    v = state.reshape(B, hi, 2, lo, K, 2).permute(0, 2, 5, 1, 3, 4)
    v = v.reshape(B, 4, hi * lo * K)
    out = gate_bmm(gate, v)
    out = out.reshape(B, 2, 2, hi, lo, K).permute(0, 3, 1, 4, 5, 2)
    return out.reshape(B, N, K, 2)


def apply_ucr(state: Tensor, gates: Tensor, target: int,
              controls: list[int], n_qubits: int) -> Tensor:
    """A uniformly controlled one-qubit gate: one gate per control value.

    ``gates`` is ``(B, 2**k, 4, 4)`` real-block, indexed by the integer the
    ``controls`` spell with ``controls[0]`` the most significant bit. The
    control axes are moved next to the batch axis so every control value is
    its own ``bmm`` batch; the gate that acts on a basis state is the one its
    control bits select. Two copies and one GEMM, like ``apply_1q``.
    """
    B, N, K, _ = state.shape
    k = len(controls)
    if gates.shape[:2] != (B, 2 ** k):
        raise ValueError(f"gates must be (B, 2**{k}, 4, 4), got "
                         f"{tuple(gates.shape)}")
    if target in controls:
        raise ValueError("the target cannot also be a control")
    v = state.reshape(B, *([2] * n_qubits), K, 2)
    dim_of = lambda q: 1 + (n_qubits - 1 - q)
    ctrl = [dim_of(q) for q in controls]
    tgt = dim_of(target)
    rest = [d for d in range(1, n_qubits + 1) if d not in ctrl and d != tgt]
    perm = [0] + ctrl + [tgt, n_qubits + 2] + rest + [n_qubits + 1]
    v = v.permute(*perm).reshape(B * 2 ** k, 4, -1)
    out = gate_bmm(gates.reshape(B * 2 ** k, 4, 4), v)
    out = out.reshape(B, *([2] * k), 2, 2, *([2] * len(rest)), K)
    inv = [0] * len(perm)
    for i, q in enumerate(perm):
        inv[q] = i
    return out.permute(*inv).reshape(B, N, K, 2)


def apply_2q(state: Tensor, gate: Tensor, a: int, b: int,
             n_qubits: int) -> Tensor:
    """Apply a batched real-block 8x8 ``gate`` (``(B, 8, 8)``) to qubits ``(a, b)``.

    The complex 4-index behind the gate is ``bit_a * 2 + bit_b``, so ``a`` is
    its high bit. ``a`` and ``b`` may come in either order; the state is
    viewed with the larger qubit index outermost and the gate is permuted to
    match.
    """
    B, N, K, _ = state.shape
    if a == b:
        raise ValueError("a two-qubit gate needs two distinct qubits")
    G = gate.reshape(B, 2, 2, 2, 2, 2, 2)     # [ia, ib, ri, ja, jb, rj]
    if a < b:
        a, b = b, a
        G = G.permute(0, 2, 1, 3, 5, 4, 6)
    G = G.reshape(B, 8, 8)
    hi = 2 ** (n_qubits - 1 - a)
    mid = 2 ** (a - 1 - b)
    lo = 2 ** b
    rest = hi * mid * lo * K
    v = state.reshape(B, hi, 2, mid, 2, lo, K, 2).permute(0, 2, 4, 7, 1, 3, 5, 6)
    v = v.reshape(B, 8, rest)
    out = gate_bmm(G, v)
    out = out.reshape(B, 2, 2, 2, hi, mid, lo, K).permute(0, 4, 1, 5, 2, 6, 7, 3)
    return out.reshape(B, N, K, 2)
