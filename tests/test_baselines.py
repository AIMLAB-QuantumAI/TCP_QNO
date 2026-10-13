import torch

from qtro.baselines.cnto import CNTO
from qtro.baselines.deeponet import DeepONet
from qtro.baselines.empirical import EmpiricalPropagator
from qtro.baselines.seqpred import SeqPred
from qtro.config import load_config
from qtro.grid import Grid
from qtro.model.qtro import N_COND_FEATURES


def _grid():
    cfg = load_config(None, ["sim.device=cpu", "grid.n_w=2", "grid.n_q=2"])
    return Grid(cfg.grid, cfg.net, device="cpu")


def _inputs():
    u = torch.tensor([[0.2, 10.0], [0.05, 30.0]])
    xi = torch.tensor([[2000.0, .03, 4.0, 0.0], [1000.0, .02, 2.0, 0.001]])
    return u, xi


def test_empirical_uses_the_nearest_normalised_configuration():
    grid = _grid()
    eye = torch.eye(grid.M)
    flip = torch.flip(eye, dims=(0,))
    features = torch.tensor([[0.1, 10, 1000, .02, 2, 0],
                             [0.4, 20, 4000, .05, 8, .001]])
    model = EmpiricalPropagator(features, torch.stack([eye, flip]),
                                features.abs().amax(0).clamp_min(1e-6))
    u, xi = features[1:, :2], features[1:, 2:]
    got = model.propagator(u, xi)[0]
    assert torch.allclose(got, flip, atol=1e-5)
    assert model.substitution_report()["n_substituted"] == 0


def test_seqpred_repairs_columns_but_exposes_raw_violations():
    grid = _grid()
    model = SeqPred(grid, rank=2, d_cond=2, hidden=4)
    with torch.no_grad():
        model.base[0, 0] = -1.0
        model.base[1, 0] = 2.0
    u = torch.tensor([[0.2, 10.0]])
    xi = torch.tensor([[2000.0, .03, 4.0, 0.0]])
    raw = model.raw_propagator(u, xi)
    repaired = model.propagator(u, xi)
    assert float(raw.detach().min()) < 0
    assert torch.all(repaired >= 0)
    assert torch.allclose(repaired.sum(1), torch.ones_like(repaired.sum(1)))


def test_seqpred_repair_uses_uniform_for_numerically_empty_column():
    raw = torch.zeros(1, 4, 4, dtype=torch.float64)
    raw[0, 0, 1] = 2.0e-38
    repaired = SeqPred.repair(raw)
    assert torch.allclose(repaired.sum(1), torch.ones(1, 4, dtype=raw.dtype))
    assert torch.equal(repaired[0, :, 1], torch.full((4,), 0.25,
                                                     dtype=raw.dtype))


def test_cnto_is_column_stochastic_and_responds_to_the_condition():
    grid = _grid()
    torch.manual_seed(0)
    model = CNTO(grid, N_COND_FEATURES, rank=3, d_cond=4, hidden=8)
    u, xi = _inputs()
    T = model.propagator(u, xi)
    assert T.shape == (2, grid.M, grid.M)
    assert torch.all(T >= 0.0)
    assert torch.allclose(T.sum(1), torch.ones(2, grid.M), atol=1e-6)
    assert not torch.allclose(T[0], T[1])
    assert set(model.state_dict()) == {"cond_scale", "base", "encoder.0.weight",
                                       "encoder.0.bias", "encoder.2.weight",
                                       "encoder.2.bias", "out", "inp"}


def test_deeponet_is_column_stochastic_and_shares_one_trunk():
    grid = _grid()
    torch.manual_seed(0)
    model = DeepONet(grid, N_COND_FEATURES, rank=4, hidden=8)
    u, xi = _inputs()
    T = model.propagator(u, xi)
    assert T.shape == (2, grid.M, grid.M)
    assert torch.allclose(T.sum(1), torch.ones(2, grid.M), atol=1e-6)
    assert model.trunk_input.shape == (grid.M * grid.M, 8)


def test_baselines_reject_a_malformed_conditioning_scale():
    grid = _grid()
    for model in (CNTO(grid, N_COND_FEATURES), SeqPred(grid), DeepONet(grid, N_COND_FEATURES)):
        for bad in (torch.ones(5), torch.tensor([1., 1., 1., 1., 1., 0.])):
            try:
                model.set_conditioning_scale(bad)
            except ValueError:
                continue
            raise AssertionError(f"{type(model).__name__} accepted {bad}")
