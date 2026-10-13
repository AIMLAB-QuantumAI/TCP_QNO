"""The two statements the model half rests on, as executable checks.

A diagonal-preserving CPTP map realises any column-stochastic matrix exactly,
and a unitary alone realises only doubly stochastic matrices, so it cannot
concentrate probability.
"""
import torch

from qtro.quantum.encoding import apply_kraus, decode, dephase, encode, kraus_from_stochastic


def _stochastic(M, seed):
    g = torch.Generator().manual_seed(seed)
    T = torch.rand(M, M, generator=g)
    return T / T.sum(0, keepdim=True)


def _simplex(M, seed):
    g = torch.Generator().manual_seed(seed)
    p = torch.rand(M, generator=g)
    return p / p.sum()


def test_encode_decode_round_trip():
    p = _simplex(8, 0)
    assert torch.allclose(decode(encode(p)), p, atol=1e-6)
    assert torch.allclose(encode(p), torch.diag(p), atol=1e-6)


def test_dephasing_kills_off_diagonal_entries():
    g = torch.Generator().manual_seed(1)
    rho = torch.rand(6, 6, generator=g)
    out = dephase(rho)
    assert torch.allclose(out, torch.diag(torch.diagonal(rho)), atol=1e-6)


def test_the_kraus_family_of_a_stochastic_matrix_is_trace_preserving():
    """sum_ij K_ij^T K_ij = I."""
    T = _stochastic(6, 2)
    K = kraus_from_stochastic(T)
    total = torch.einsum("kij,kil->jl", K, K)
    assert torch.allclose(total, torch.eye(6), atol=1e-5)


def test_the_diagonal_action_reproduces_the_stochastic_matrix():
    """Phi(diag(p)) = diag(T p)."""
    T = _stochastic(6, 3)
    p = _simplex(6, 4)
    out = apply_kraus(kraus_from_stochastic(T), encode(p))
    assert torch.allclose(decode(out), T @ p, atol=1e-5)
    assert torch.allclose(out, torch.diag(torch.diagonal(out)), atol=1e-5)


def test_the_output_is_a_valid_density_matrix():
    """Positivity of Phi(diag(p)) for a diagonal input only, not complete
    positivity: since Phi(diag(p)) = diag(T p) this reduces to T @ p >= 0,
    which plain transposition also satisfies. Complete positivity is pinned
    separately below."""
    T = _stochastic(6, 5)
    p = _simplex(6, 6)
    out = apply_kraus(kraus_from_stochastic(T), encode(p))
    assert torch.isclose(torch.trace(out), torch.tensor(1.0), atol=1e-5)
    assert torch.all(torch.linalg.eigvalsh(out) >= -1e-6)


def test_the_kraus_map_is_completely_positive():
    """Choi's theorem: Phi is completely positive iff its Choi matrix
    C(Phi) = sum_ij E_ij (x) Phi(E_ij) is PSD, E_ij the standard basis of
    M x M matrices. Building it applies Phi to every basis matrix, including
    the M(M-1) off-diagonal ones no other test here touches. Transposition,
    the standard positive-but-not-completely-positive map, has the SWAP
    operator as its Choi matrix, with eigenvalues -1, and fails this check."""
    M = 6
    T = _stochastic(M, 11)
    K = kraus_from_stochastic(T)
    basis = torch.eye(M * M).reshape(M * M, M, M)
    choi = torch.zeros(M * M, M * M)
    for k in range(M * M):
        i, j = divmod(k, M)
        choi[i * M:(i + 1) * M, j * M:(j + 1) * M] = apply_kraus(K, basis[k])
    assert torch.allclose(choi, choi.T, atol=1e-5)
    eigvals = torch.linalg.eigvalsh((choi + choi.T) / 2)
    assert torch.all(eigvals >= -1e-4)


def test_a_unitary_alone_is_doubly_stochastic():
    """D o U o D acts as T_ij = |U_ij|^2, which has unit row sums too."""
    g = torch.Generator().manual_seed(7)
    U, _ = torch.linalg.qr(torch.randn(10, 10, generator=g))
    T = U ** 2
    assert torch.allclose(T.sum(0), torch.ones(10), atol=1e-5)
    assert torch.allclose(T.sum(1), torch.ones(10), atol=1e-5)


def test_a_unitary_alone_cannot_decrease_entropy():
    g = torch.Generator().manual_seed(8)
    U, _ = torch.linalg.qr(torch.randn(10, 10, generator=g))
    T = U ** 2
    p = _simplex(10, 9)
    h = lambda v: -(v.clamp(min=1e-12) * v.clamp(min=1e-12).log()).sum()
    assert h(T @ p) >= h(p) - 1e-5


def test_a_unitary_alone_fixes_the_uniform_distribution():
    g = torch.Generator().manual_seed(10)
    U, _ = torch.linalg.qr(torch.randn(10, 10, generator=g))
    uniform = torch.full((10,), 0.1)
    assert torch.allclose((U ** 2) @ uniform, uniform, atol=1e-5)
