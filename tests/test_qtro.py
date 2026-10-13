from unittest.mock import patch

import torch

from qtro.config import load_config
from qtro.grid import Grid
from qtro.model.qtro import QTRO


def _setup(n_layers=2, device="cpu"):
    cfg = load_config(None, [f"sim.device={device}"])
    cfg.model.n_layers = n_layers
    grid = Grid(cfg.grid, cfg.net, device=device)
    torch.manual_seed(0)
    return grid, QTRO(grid, cfg.model).to(device)


def _inputs(B, device="cpu"):
    u = torch.rand(B, 2, device=device) * torch.tensor([0.5, 60.0],
                                                       device=device)
    xi = torch.tensor([[10000.0, 0.03, 10.0, 0.001]],
                      device=device).expand(B, 4)
    return u, xi


def test_propagator_is_column_stochastic():
    grid, model = _setup()
    u, xi = _inputs(3)
    T = model.propagator(u, xi)
    assert T.shape == (3, grid.M, grid.M)
    assert torch.all(T >= -1e-7)
    assert torch.allclose(T.sum(1), torch.ones(3, grid.M), atol=1e-5)


def test_selected_columns_match_the_full_channel():
    grid, model = _setup(n_layers=1)
    u, xi = _inputs(2)
    indices = torch.tensor([[0, 3, grid.M - 1], [2, 7, grid.M - 2]])
    full = model.propagator(u, xi)
    selected = model.propagator_columns(u, xi, indices)
    expected = full.gather(2, indices.unsqueeze(1).expand(-1, grid.M, -1))
    assert torch.allclose(selected, expected, atol=2.0e-6)


def test_state_dict_has_only_transport_reset_and_scale_keys():
    _, model = _setup(n_layers=1)
    keys = set(model.state_dict())
    assert "cond_scale" in keys
    assert all(k == "cond_scale" or k.startswith(("transport.circuit.", "reset."))
               for k in keys)


def test_validity_holds_over_a_hundred_step_rollout():
    """Validity is structural, so it survives any horizon.

    Under ``no_grad``: with autograd on, each step keeps its own propagator
    graph alive, and a hundred of them need about 25 GiB, which a CI runner
    does not have.
    """
    grid, model = _setup()
    u, xi = _inputs(2)
    p = torch.rand(2, grid.M)
    p = p / p.sum(-1, keepdim=True)
    with torch.no_grad():
        for _ in range(100):
            p = model.step(p, u, xi)
            assert float(p.min()) >= -1e-6
            assert abs(float(p.sum(-1).max()) - 1.0) < 1e-5


def test_validity_holds_for_arbitrary_parameter_values():
    grid, model = _setup()
    with torch.no_grad():
        for param in model.parameters():
            param.mul_(50.0)
    u, xi = _inputs(2)
    p = torch.rand(2, grid.M)
    p = p / p.sum(-1, keepdim=True)
    with torch.no_grad():
        for _ in range(20):
            p = model.step(p, u, xi)
    assert float(p.min()) >= -1e-6
    assert abs(float(p.sum(-1).max()) - 1.0) < 1e-5


def test_float64_agrees_with_float32():
    grid, model = _setup()
    u, xi = _inputs(1)
    p = torch.rand(1, grid.M)
    p = p / p.sum(-1, keepdim=True)
    a = model.step(p, u, xi)
    b = model.double().step(p.double(), u.double(), xi.double())
    assert torch.allclose(a.double(), b, atol=1e-5)


def test_rollout_shape_and_normalisation():
    grid, model = _setup()
    u, xi = _inputs(2)
    p0 = torch.rand(2, grid.M)
    p0 = p0 / p0.sum(-1, keepdim=True)
    u_seq = u.unsqueeze(1).expand(2, 5, 2)
    out = model.rollout(p0, u_seq, xi)
    assert out.shape == (2, 5, grid.M)
    assert torch.allclose(out.sum(-1), torch.ones(2, 5), atol=1e-5)


def test_rollout_composes_single_steps():
    grid, model = _setup()
    u, xi = _inputs(1)
    p0 = torch.rand(1, grid.M)
    p0 = p0 / p0.sum(-1, keepdim=True)
    out = model.rollout(p0, u.unsqueeze(1).expand(1, 3, 2), xi)
    manual = model.step(model.step(model.step(p0, u, xi), u, xi), u, xi)
    assert torch.allclose(out[:, -1], manual, atol=1e-5)


def test_constant_control_rollout_constructs_the_propagator_once():
    grid, model = _setup(n_layers=1)
    u, xi = _inputs(1)
    p0 = torch.rand(1, grid.M)
    p0 = p0 / p0.sum(-1, keepdim=True)
    u_seq = u.unsqueeze(1).expand(1, 5, 2)
    with patch.object(model, "propagator", wraps=model.propagator) as build:
        model.rollout(p0, u_seq, xi)
    assert build.call_count == 1


def test_rollout_with_varying_controls_composes_single_steps():
    grid, model = _setup(n_layers=1)
    u, xi = _inputs(1)
    p0 = torch.rand(1, grid.M)
    p0 = p0 / p0.sum(-1, keepdim=True)
    u_seq = torch.stack([u, u * torch.tensor([[0.8, 1.1]]), u], dim=1)
    out = model.rollout(p0, u_seq, xi)
    manual = p0
    for h in range(u_seq.shape[1]):
        manual = model.step(manual, u_seq[:, h], xi)
    assert torch.allclose(out[:, -1], manual, atol=1e-5)


def test_different_controls_give_different_predictions():
    grid, model = _setup()
    xi = torch.tensor([[10000.0, 0.03, 10.0, 0.001]]).expand(2, 4)
    u = torch.tensor([[0.01, 10.0], [0.9, 90.0]])
    # A uniform input can be invariant under the transport's initially
    # near-unital local rotations, hiding genuine condition dependence.
    p = torch.zeros(2, grid.M)
    p[:, grid.M // 3] = 1.0
    out = model.step(p, u, xi)
    assert float((out[0] - out[1]).abs().max().detach()) > 1e-8


def test_is_differentiable_through_the_rollout():
    grid, model = _setup()
    u, xi = _inputs(1)
    p0 = torch.full((1, grid.M), 1.0 / grid.M)
    model.rollout(p0, u.unsqueeze(1).expand(1, 3, 2), xi).sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
