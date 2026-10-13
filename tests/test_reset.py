import torch

from qtro.config import load_config
from qtro.grid import Grid
from qtro.model.reset import ResetBlock, reset_matrix


def _setup():
    cfg = load_config(None, ["sim.device=cpu"])
    grid = Grid(cfg.grid, cfg.net, device="cpu")
    torch.manual_seed(0)
    return grid, ResetBlock(grid, cfg.model, n_cond_features=6)


def test_intensities_lie_on_the_simplex_face():
    _, block = _setup()
    gamma = block.intensities(torch.randn(4, 6))
    assert torch.all(gamma >= 0.0)
    assert torch.all(gamma.sum(-1) <= 1.0 + 1e-6)


def test_reset_kraus_family_is_complete():
    """sum_i (A0i^T A0i + sum_z Az_i^T Az_i) = I."""
    grid, block = _setup()
    gamma = block.intensities(torch.randn(1, 6))[0]
    targets = grid.reset_targets()
    total = torch.zeros(grid.M)
    for i in range(grid.M):
        total[i] += 1.0 - float(gamma[i].sum().detach())
        for z in range(grid.n_events):
            total[i] += float(gamma[i, z].detach())
    assert torch.allclose(total, torch.ones(grid.M), atol=1e-5)
    assert targets.shape == (2, grid.M)


def test_reset_matrix_moves_gamma_to_the_protocol_targets():
    """[R p]_j = (1 - sum_z gamma_z(j)) p_j + sum_z sum_{i: pi_z(i)=j} gamma_z(i) p_i."""
    torch.manual_seed(1)
    M = 6
    targets = torch.tensor([[1, 2, 3, 4, 5, 5], [0, 0, 0, 0, 0, 0]])
    gamma = torch.rand(1, M, 2) * 0.4
    T = reset_matrix(gamma, targets, M)[0]
    p = torch.rand(M)
    p = p / p.sum()
    got = T @ p
    want = (1.0 - gamma[0].sum(-1)) * p
    for z in range(2):
        want = want.index_add(0, targets[z], gamma[0, :, z] * p)
    assert torch.allclose(got, want, atol=1e-6)


def test_sparse_reset_application_matches_dense_matrix_for_vectors_and_matrices():
    grid, block = _setup()
    cond = torch.randn(2, 6)
    dense = block(cond)
    vector = torch.rand(2, grid.M)
    matrix = torch.rand(2, grid.M, 7)
    assert torch.allclose(
        block.apply_channel(cond, vector),
        torch.bmm(dense, vector.unsqueeze(-1)).squeeze(-1), atol=1e-6)
    assert torch.allclose(
        block.apply_channel(cond, matrix), torch.bmm(dense, matrix), atol=1e-6)


def test_reset_matrix_is_column_stochastic():
    torch.manual_seed(2)
    M = 6
    targets = torch.tensor([[1, 2, 3, 4, 5, 5], [0, 0, 0, 0, 0, 0]])
    T = reset_matrix(torch.rand(3, M, 2) * 0.4, targets, M)
    assert torch.all(T >= 0.0)
    assert torch.allclose(T.sum(1), torch.ones(3, M), atol=1e-6)


def test_zero_intensity_gives_the_identity():
    M = 6
    targets = torch.tensor([[1, 2, 3, 4, 5, 5], [0, 0, 0, 0, 0, 0]])
    T = reset_matrix(torch.zeros(1, M, 2), targets, M)[0]
    assert torch.allclose(T, torch.eye(M), atol=1e-6)


def test_timeout_concentrates_mass_and_reduces_entropy():
    grid, _ = _setup()
    gamma = torch.zeros(1, grid.M, 2)
    gamma[:, :, grid.EVENT_TIMEOUT] = 1.0
    T = reset_matrix(gamma, grid.reset_targets(), grid.M)[0]
    p = torch.full((grid.M,), 1.0 / grid.M)
    h = lambda v: -(v.clamp(min=1e-12) * v.clamp(min=1e-12).log()).sum()
    assert h(T @ p) < h(p) - 1e-3


def test_block_output_is_column_stochastic():
    grid, block = _setup()
    T = block(torch.randn(3, 6))
    assert T.shape == (3, grid.M, grid.M)
    assert torch.all(T >= 0.0)
    assert torch.allclose(T.sum(1), torch.ones(3, grid.M), atol=1e-5)


def test_max_probability_caps_the_total_event_mass():
    cfg = load_config(None, ["sim.device=cpu", "model.reset_max_probability=0.3"])
    grid = Grid(cfg.grid, cfg.net, device="cpu")
    block = ResetBlock(grid, cfg.model, n_cond_features=6)
    gamma = block.intensities(torch.randn(3, 6))
    assert torch.all(gamma.sum(-1) <= 0.3 + 1e-6)
