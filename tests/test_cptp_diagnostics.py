import torch

from qtro.config import load_config
from qtro.eval.cptp import (
    dephased_choi_metrics,
    reset_kraus_metrics,
    stinespring_metrics,
    tcp_reset_target_metrics,
)
from qtro.grid import Grid
from qtro.model.reset import reset_matrix


def test_dephased_implicit_choi_matches_explicit_spectrum():
    torch.manual_seed(3)
    M = 4
    T = torch.rand(M, M, dtype=torch.float64)
    T /= T.sum(0, keepdim=True)
    metrics = dephased_choi_metrics(T)
    # Phi(E_ab)=0 for a!=b and diag(T[:,a]) for a=b.
    choi = torch.zeros(M * M, M * M, dtype=torch.float64)
    for a in range(M):
        choi[a*M:(a+1)*M, a*M:(a+1)*M] = torch.diag(T[:, a])
    explicit = torch.linalg.eigvalsh(choi)
    assert metrics["choi_min_eigenvalue"].item() == explicit.min().item()
    assert metrics["trace_preserving_residual"].item() < 1e-14
    assert metrics["choi_trace_residual"].item() < 1e-14


def test_negative_transition_is_detected_as_non_cp():
    T = torch.tensor([[1.2, -0.1], [-0.2, 1.1]], dtype=torch.float64)
    metrics = dephased_choi_metrics(T)
    assert metrics["choi_min_eigenvalue"].item() == -0.2
    assert metrics["choi_psd_violation"].item() == 0.2


def test_stinespring_metrics_check_completeness_and_choi_gram():
    torch.manual_seed(4)
    M, d_env = 5, 3
    q, _ = torch.linalg.qr(torch.randn(M*d_env, M, dtype=torch.float64))
    metrics = stinespring_metrics(q.unsqueeze(0), system_dim=M,
                                  environment_dim=d_env)
    assert metrics["kraus_completeness_max_residual"].item() < 1e-14
    assert metrics["choi_psd_violation"].item() == 0.0
    assert metrics["choi_trace_residual"].item() < 1e-13
    assert torch.allclose(
        metrics["transport"].sum(1),
        torch.ones(1, M, dtype=torch.float64), atol=1e-14)


def test_reset_kraus_and_tcp_semantics_are_exact():
    cfg = load_config(None, ["sim.device=cpu"])
    grid = Grid(cfg.grid, cfg.net, device="cpu")
    torch.manual_seed(5)
    gamma = torch.softmax(torch.randn(2, grid.M, 3, dtype=torch.float64), -1)[..., 1:]
    gamma *= 0.7
    targets = grid.reset_targets()
    reset = reset_matrix(gamma, targets, grid.M)
    metrics = reset_kraus_metrics(gamma, reset, targets)
    assert metrics["choi_min_weight"].min().item() >= 0.0
    assert metrics["choi_psd_violation"].max().item() == 0.0
    assert metrics["probability_bounds_violation"].max().item() == 0.0
    assert metrics["kraus_completeness_residual"].max().item() < 1e-14
    assert metrics["matrix_reconstruction_residual"].max().item() == 0.0
    assert metrics["protocol_routing_leakage"].max().item() == 0.0
    semantics = tcp_reset_target_metrics(grid, targets)
    assert all(value == 0 for key, value in semantics.items()
               if key.endswith("_count"))
    assert semantics["timeout_unique_targets"] == grid.n_q


def test_reset_cp_check_rejects_negative_stay_probability():
    cfg = load_config(None, ["sim.device=cpu"])
    grid = Grid(cfg.grid, cfg.net, device="cpu")
    gamma = torch.zeros(1, grid.M, 2, dtype=torch.float64)
    gamma[..., 0] = 0.8
    gamma[..., 1] = 0.3
    targets = grid.reset_targets()
    reset = reset_matrix(gamma, targets, grid.M)
    metrics = reset_kraus_metrics(gamma, reset, targets)
    assert torch.allclose(
        metrics["choi_min_weight"], torch.tensor([-0.1], dtype=torch.float64))
    assert torch.allclose(
        metrics["choi_psd_violation"], torch.tensor([0.1], dtype=torch.float64))
    assert torch.allclose(
        metrics["probability_bounds_violation"],
        torch.tensor([0.1], dtype=torch.float64))
