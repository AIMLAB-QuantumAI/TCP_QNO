import torch

from qtro.config import load_config
from qtro.grid import Grid
from qtro.model.transport import TransportBlock


def _setup(n_layers=1):
    cfg = load_config(None, ["sim.device=cpu"])
    cfg.model.n_layers = n_layers
    grid = Grid(cfg.grid, cfg.net, device="cpu")
    torch.manual_seed(0)
    return grid, TransportBlock(grid, cfg.model, n_cond_features=6)


def test_output_is_column_stochastic():
    grid, block = _setup()
    T = block(torch.randn(3, 6))
    assert T.shape == (3, grid.M, grid.M)
    assert torch.all(T >= -1e-7)
    assert torch.allclose(T.sum(1), torch.ones(3, grid.M), atol=1e-5)


def test_selected_source_columns_match_full_transport_exactly():
    grid, block = _setup()
    cond = torch.randn(2, 6)
    indices = torch.tensor([[0, 3, 7], [2, 4, 1]])
    full = block(cond)
    selected = block(cond, source_indices=indices)
    expected = full.gather(
        2, indices.unsqueeze(1).expand(-1, grid.M, -1))
    assert selected.shape == (2, grid.M, 3)
    assert torch.allclose(selected, expected, atol=2e-6)
    assert torch.allclose(selected.sum(1), torch.ones(2, 3), atol=1e-5)


def test_depth_one_moves_probability_at_most_four_bins_per_axis():
    """The locality prior as a provable property of the Givens ladder.

    One layer applies V_l then W_l, and each of those is a full sweep set whose
    even and odd sweeps move a cell index by at most one bin each. The bound is
    therefore 4 bins per axis per layer, i.e. 4 * n_layers overall.
    """
    grid, block = _setup(n_layers=1)
    T = block(torch.randn(1, 6))[0]
    a, b, _ = grid.axis_bins()
    far = ((a[:, None] - a[None, :]).abs() > 4) | \
          ((b[:, None] - b[None, :]).abs() > 4)
    assert T[far].abs().max() < 1e-6


def test_depth_two_stays_within_eight_bins():
    grid, block = _setup(n_layers=2)
    T = block(torch.randn(1, 6))[0]
    a, b, _ = grid.axis_bins()
    far = ((a[:, None] - a[None, :]).abs() > 8) | \
          ((b[:, None] - b[None, :]).abs() > 8)
    assert T[far].abs().max() < 1e-6


def test_the_bound_is_not_vacuous():
    """Depth 1 must actually reach beyond the depth-0 diagonal."""
    grid, block = _setup(n_layers=1)
    with torch.no_grad():
        for p in block.parameters():
            p.mul_(50.0)
    T = block(torch.randn(1, 6))[0]
    a, _, _ = grid.axis_bins()
    off = (a[:, None] - a[None, :]).abs() >= 2
    assert T[off].abs().max() > 1e-4


def test_can_reduce_entropy_because_of_the_environment_register():
    """A bare unitary realises only doubly stochastic kernels; the dilation
    must not. A doubly stochastic matrix has unit row sums as well as unit
    column sums, and tracing out the environment breaks the row sums, which
    is precisely the irreversibility a reset needs."""
    grid, block = _setup(n_layers=4)
    with torch.no_grad():
        for p in block.parameters():
            p.mul_(50.0)
    T = block(torch.randn(1, 6))[0]
    assert torch.allclose(T.sum(0), torch.ones(grid.M), atol=1e-5)
    assert (T.sum(1) - 1.0).abs().max() > 1e-3


def test_is_differentiable():
    _, block = _setup()
    block(torch.randn(2, 6)).sum().backward()
    grads = [p.grad for p in block.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_reshape_matches_the_joint_index_definition_directly():
    """forward reshapes psi as (B, M, d_env, M) and sums squares over dim=2.
    Confirm independently of the circuit that this reproduces
    T[i, j] = sum_e psi[i * d_env + e, j]^2, so an axis swap cannot hide
    behind orthonormality."""
    torch.manual_seed(11)
    M, d_env, B = 5, 3, 2
    psi = torch.randn(B, M * d_env, M)
    amp = psi.reshape(B, M, d_env, M)
    got = (amp ** 2).sum(dim=2)

    want = torch.zeros(B, M, M)
    for i in range(M):
        for e in range(d_env):
            joint = i * d_env + e
            want[:, i, :] += psi[:, joint, :] ** 2
    assert torch.allclose(got, want, atol=1e-6)


def test_the_default_depth_bound_is_vacuous():
    """Known property, pinned: a depth-L ladder displaces a cell by at most
    4L bins per axis, and with n_w = n_q = 16 the axes are 15 bins wide, so
    at the default depth the bandedness prior restricts nothing. A default
    that binds has to be chosen deliberately rather than inherited."""
    cfg = load_config(None, [])
    grid = Grid(cfg.grid, cfg.net, device="cpu")
    n_layers = cfg.model.n_layers
    axis_width = grid.n_w - 1
    assert grid.n_w == grid.n_q == 16
    assert n_layers == 8
    assert axis_width == 15
    assert 4 * n_layers == 32
    assert 4 * n_layers >= axis_width
