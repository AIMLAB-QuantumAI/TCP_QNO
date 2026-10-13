import torch

from qtro.quantum.givens import apply_sweep, sweep_matrix


def _sweep(n=8):
    lo = torch.tensor([0, 2, 4, 6])
    hi = torch.tensor([1, 3, 5, 7])
    return lo, hi


def test_matches_the_dense_reference():
    torch.manual_seed(0)
    n = 8
    lo, hi = _sweep(n)
    angles = torch.randn(1, lo.numel())
    psi = torch.randn(1, n, 3)
    got = apply_sweep(psi, lo, hi, angles)
    want = sweep_matrix(n, lo, hi, angles[0]) @ psi[0]
    assert torch.allclose(got[0], want, atol=1e-5)


def test_matches_the_dense_reference_on_a_nonsymmetric_batch():
    """A single-sample batch cannot show batch mixing. Check the batched kernel
    against the dense matrix per sample, with different angles per row, which
    is the shape the transport block calls it with."""
    torch.manual_seed(7)
    n = 8
    lo, hi = _sweep(n)
    B, C = 3, 4
    angles = torch.randn(B, lo.numel())
    psi = torch.randn(B, n, C)
    got = apply_sweep(psi, lo, hi, angles)
    for k in range(B):
        want = sweep_matrix(n, lo, hi, angles[k]) @ psi[k]
        assert torch.allclose(got[k], want, atol=1e-5), k


def test_dense_reference_is_orthogonal():
    torch.manual_seed(1)
    n = 8
    lo, hi = _sweep(n)
    U = sweep_matrix(n, lo, hi, torch.randn(lo.numel()))
    assert torch.allclose(U @ U.T, torch.eye(n), atol=1e-5)


def test_preserves_column_norms():
    torch.manual_seed(2)
    n, C = 8, 5
    lo, hi = _sweep(n)
    psi = torch.randn(2, n, C)
    out = apply_sweep(psi, lo, hi, torch.randn(2, lo.numel()))
    assert torch.allclose(out.pow(2).sum(1), psi.pow(2).sum(1), atol=1e-5)


def test_zero_angle_is_the_identity():
    n, C = 8, 4
    lo, hi = _sweep(n)
    psi = torch.randn(3, n, C)
    out = apply_sweep(psi, lo, hi, torch.zeros(3, lo.numel()))
    assert torch.allclose(out, psi, atol=1e-6)


def test_untouched_indices_are_unchanged():
    n, C = 8, 2
    lo = torch.tensor([0])
    hi = torch.tensor([1])
    psi = torch.randn(1, n, C)
    out = apply_sweep(psi, lo, hi, torch.full((1, 1), 0.7))
    assert torch.allclose(out[:, 2:], psi[:, 2:], atol=1e-6)


def test_each_batch_element_uses_its_own_angles():
    n, C = 8, 1
    lo, hi = _sweep(n)
    psi = torch.randn(1, n, C).expand(2, n, C).contiguous()
    angles = torch.stack([torch.zeros(lo.numel()),
                          torch.full((lo.numel(),), 0.9)])
    out = apply_sweep(psi, lo, hi, angles)
    assert torch.allclose(out[0], psi[0], atol=1e-6)
    assert not torch.allclose(out[1], psi[1], atol=1e-3)


def test_is_differentiable():
    n, C = 8, 2
    lo, hi = _sweep(n)
    angles = torch.randn(1, lo.numel(), requires_grad=True)
    psi = torch.randn(1, n, C)
    apply_sweep(psi, lo, hi, angles).sum().backward()
    assert angles.grad is not None and torch.isfinite(angles.grad).all()


def test_gradient_matches_finite_differences():
    """Verify the analytic gradient against a central-difference estimate, so
    a sign error in cos/sin (finite but wrong gradients) would be caught.

    The loss must not be sum(out**2): a Givens sweep is exactly orthogonal, so
    that loss is angle-invariant and its true gradient is zero everywhere.
    Weighting by a fixed non-symmetric tensor breaks the invariance; float64
    keeps cancellation noise far below the truncation error.
    """
    torch.manual_seed(3)
    n, C = 8, 2
    lo, hi = _sweep(n)
    psi = torch.randn(1, n, C, dtype=torch.float64)
    weight = torch.randn(1, n, C, dtype=torch.float64)
    angles0 = torch.randn(1, lo.numel(), dtype=torch.float64)

    def loss(a: torch.Tensor) -> torch.Tensor:
        return (apply_sweep(psi, lo, hi, a) * weight).sum()

    angles = angles0.clone().requires_grad_(True)
    loss(angles).backward()
    analytic = angles.grad.clone()

    eps = 1e-6
    numeric = torch.zeros_like(angles0)
    for i in range(angles0.numel()):
        flat = angles0.flatten().clone()
        flat[i] += eps
        plus = loss(flat.view_as(angles0))
        flat = angles0.flatten().clone()
        flat[i] -= eps
        minus = loss(flat.view_as(angles0))
        numeric[0, i] = (plus - minus) / (2 * eps)

    assert torch.allclose(analytic, numeric, atol=1e-6), (analytic, numeric)
    assert analytic.abs().max() > 0.1, "loss must not be angle-invariant"


def test_empty_sweep_does_not_crash():
    """An empty sweep (env_odd at d_env=2, or w_odd at n_w=2) must be a no-op,
    since apply_sweep is always called with whatever build_sweeps hands it."""
    n, B, C = 8, 2, 3
    lo = torch.empty(0, dtype=torch.long)
    hi = torch.empty(0, dtype=torch.long)
    angles = torch.empty(B, 0)
    psi = torch.randn(B, n, C)
    out = apply_sweep(psi, lo, hi, angles)
    assert torch.allclose(out, psi)
