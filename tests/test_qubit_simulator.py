"""Qubit statevector primitives on the real (re, im) layout: unitarity, placement, hand cases."""
import math

import torch

from qtro.qubit.simulator import (
    apply_1q,
    apply_2q,
    basis_columns,
    cnot,
    cnot_chain_permutation,
    gate_bmm,
    real_block,
    rz_ry,
    su4,
    to_complex,
    to_real,
)


def _c(state_real):
    return to_complex(state_real)


def test_su4_is_unitary():
    theta = torch.randn(7, 15, dtype=torch.float64)
    U = su4(theta)
    eye = torch.eye(4, dtype=U.dtype).expand(7, 4, 4)
    assert torch.allclose(U.conj().transpose(-1, -2) @ U, eye, atol=1e-10)


def test_su4_zero_angles_is_identity():
    U = su4(torch.zeros(3, 15, dtype=torch.float64))
    assert torch.allclose(U, torch.eye(4, dtype=U.dtype).expand(3, 4, 4))


def test_rz_ry_zero_is_identity():
    G = rz_ry(torch.zeros(2, 5), torch.zeros(2, 5))
    assert G.shape == (2, 5, 2, 2)
    assert torch.allclose(G, torch.eye(2, dtype=G.dtype).expand(2, 5, 2, 2))


def test_real_block_reproduces_complex_product():
    torch.manual_seed(0)
    G = torch.randn(3, 4, 4, dtype=torch.complex64)
    z = torch.randn(3, 4, 7, dtype=torch.complex64)
    want = G @ z
    zr = torch.view_as_real(z).permute(0, 1, 3, 2).reshape(3, 8, 7)
    got = real_block(G) @ zr
    got = torch.view_as_complex(got.reshape(3, 4, 2, 7).permute(0, 1, 3, 2)
                                .contiguous())
    assert torch.allclose(got, want, atol=1e-5)


def test_apply_1q_ry_half_pi_splits_a_basis_state():
    state = torch.zeros(1, 2, 1, dtype=torch.complex64)
    state[0, 0, 0] = 1.0
    G = real_block(rz_ry(torch.zeros(1, 1), torch.full((1, 1), math.pi / 2))[:, 0])
    out = _c(apply_1q(to_real(state), G, q=0, n_qubits=1))
    assert torch.allclose(out.abs()[0, :, 0] ** 2, torch.tensor([0.5, 0.5]),
                          atol=1e-6)


def test_apply_1q_targets_the_right_qubit():
    state = torch.zeros(1, 8, 1, dtype=torch.complex64)
    state[0, 0, 0] = 1.0
    G = real_block(rz_ry(torch.zeros(1, 1), torch.full((1, 1), math.pi))[:, 0])
    out = _c(apply_1q(to_real(state), G, q=1, n_qubits=3))
    assert torch.argmax(out.abs()[0, :, 0]).item() == 2


def test_cnot_permutes_basis_states():
    state = torch.zeros(1, 4, 1, dtype=torch.complex64)
    state[0, 2, 0] = 1.0
    C = real_block(cnot().expand(1, 4, 4))
    out = _c(apply_2q(to_real(state), C, a=1, b=0, n_qubits=2))
    assert torch.argmax(out.abs()[0, :, 0]).item() == 3


def test_cnot_with_control_below_target():
    state = torch.zeros(1, 4, 1, dtype=torch.complex64)
    state[0, 1, 0] = 1.0
    C = real_block(cnot().expand(1, 4, 4))
    out = _c(apply_2q(to_real(state), C, a=0, b=1, n_qubits=2))
    assert torch.argmax(out.abs()[0, :, 0]).item() == 3


def test_apply_2q_su4_preserves_norm_on_batch():
    torch.manual_seed(0)
    state = torch.randn(3, 16, 5, dtype=torch.complex64)
    state = state / state.norm(dim=1, keepdim=True)
    U = real_block(su4(torch.randn(3, 15)).to(torch.complex64))
    out = _c(apply_2q(to_real(state), U, a=3, b=1, n_qubits=4))
    assert torch.allclose(out.norm(dim=1), torch.ones(3, 5), atol=1e-5)


def test_apply_2q_matches_dense_matrix_on_small_register():
    torch.manual_seed(2)
    U = su4(torch.randn(1, 15)).to(torch.complex64)[0]
    state = torch.randn(1, 8, 2, dtype=torch.complex64)
    out = _c(apply_2q(to_real(state), real_block(U.unsqueeze(0)),
                      a=2, b=0, n_qubits=3))
    full = torch.zeros(8, 8, dtype=torch.complex64)
    for x in range(8):
        b2, b1, b0 = (x >> 2) & 1, (x >> 1) & 1, x & 1
        for i2 in (0, 1):
            for i0 in (0, 1):
                y = i2 * 4 + b1 * 2 + i0
                full[y, x] = U[i2 * 2 + i0, b2 * 2 + b0]
    ref = torch.einsum("yx,bxk->byk", full, state)
    assert torch.allclose(out, ref, atol=1e-5)


def test_cnot_chain_permutation_matches_gate_by_gate():
    torch.manual_seed(1)
    n = 5
    state = torch.randn(2, 2 ** n, 3, dtype=torch.complex64)
    C = real_block(cnot().expand(2, 4, 4))
    ref = to_real(state)
    for q in range(n - 1):
        ref = apply_2q(ref, C, a=q, b=q + 1, n_qubits=n)
    perm = cnot_chain_permutation(n)
    assert torch.allclose(to_real(state)[:, perm], ref, atol=1e-6)


def test_basis_columns_places_source_j_at_joint_index_j_times_d_env():
    cols = basis_columns(M=4, d_env=2, device="cpu")
    assert cols.shape == (8, 4, 2)
    for j in range(4):
        assert cols[j * 2, j, 0] == 1
        assert cols[:, j].sum() == 1


def test_gate_bmm_gradients_match_autograd_bmm():
    torch.manual_seed(3)
    B, D, R = 2, 8, 4096
    gate = torch.randn(B, D, D, requires_grad=True)
    v = torch.randn(B, D, R, requires_grad=True)
    go = torch.randn(B, D, R)
    out = gate_bmm(gate, v)
    out.backward(go)
    g_gate, g_v = gate.grad.clone(), v.grad.clone()
    gate.grad = v.grad = None
    ref = torch.bmm(gate, v)
    ref.backward(go)
    assert torch.allclose(g_gate, gate.grad, rtol=1e-4, atol=1e-3)
    assert torch.allclose(g_v, v.grad, rtol=1e-5, atol=1e-5)


def test_gate_bmm_handles_R_not_divisible_by_256():
    torch.manual_seed(4)
    gate = torch.randn(1, 4, 4, requires_grad=True)
    v = torch.randn(1, 4, 6 * 7, requires_grad=True)     # R = 42
    out = gate_bmm(gate, v)
    out.sum().backward()
    ref_gate = torch.bmm(torch.ones_like(out), v.transpose(1, 2))
    assert torch.allclose(gate.grad, ref_gate, atol=1e-4)
