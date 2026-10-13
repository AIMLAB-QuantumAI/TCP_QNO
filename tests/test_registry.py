import pytest
import torch

from qtro.config import load_config
from qtro.grid import Grid
from qtro.model.qtro import QTRO
from qtro.qubit.model import QubitQTRO
from qtro.qubit.ucr import UCRQTRO
from qtro.registry import MODEL_NAMES, build_model, canonical_name, parameter_count


def _setup():
    cfg = load_config(None, ["sim.device=cpu", "model.n_layers=1", "model.d_env=2"])
    grid = Grid(cfg.grid, cfg.net, device="cpu")
    torch.manual_seed(0)
    return cfg, grid


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_registered_model_is_column_stochastic(name):
    cfg, grid = _setup()
    model = build_model(name, grid, cfg, {"rank": 3})
    model.set_conditioning_scale(torch.ones(6))
    u = torch.tensor([[0.1, 20.0], [0.3, 34.0]])
    xi = torch.tensor([[2000.0, 0.03, 4.0, 0.0], [1000.0, 0.02, 2.0, 0.001]])
    with torch.no_grad():
        P = model.propagator(u, xi)
    assert P.shape == (2, grid.M, grid.M)
    assert float(P.min()) >= -1e-7
    assert torch.allclose(P.sum(1), torch.ones(2, grid.M), atol=1e-5)
    assert parameter_count(model) > 0


def test_legacy_names_map_to_the_paper_terms():
    cfg, grid = _setup()
    assert canonical_name("qkfp") == "qtro"
    assert canonical_name("softmax") == "cnto"
    assert canonical_name("seqpred") == "seqpred"
    assert isinstance(build_model("qkfp", grid, cfg), QTRO)
    assert type(build_model("softmax", grid, cfg, {"rank": 2})).__name__ == "CNTO"


def test_qubit_names_select_the_ansatz_and_depth():
    cfg, grid = _setup()
    cfg.model.n_layers = 3
    dense = build_model("qubit_dense", grid, cfg)
    assert isinstance(dense, QubitQTRO)
    assert dense.transport.circuit.n_layers == 3
    assert dense.transport.circuit.ansatz == "dense"
    cfg.model.qubit_controls = 2
    ucr = build_model("qubit_ucr", grid, cfg)
    assert isinstance(ucr, UCRQTRO)
    assert ucr.transport.circuit.n_controls == 2


def test_unknown_names_are_refused():
    cfg, grid = _setup()
    with pytest.raises(ValueError, match="unknown model"):
        build_model("qkfp_pending2", grid, cfg)
