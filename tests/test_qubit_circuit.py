"""The qubit-native circuits and the models built on them."""
import pytest
import torch

from qtro.config import load_config
from qtro.grid import Grid
from qtro.quantum.qubit import registers
from qtro.qubit.circuit import QubitCircuit, chain_links, dense_pairs
from qtro.qubit.model import QubitQTRO, QubitTransportBlock


def _cfg(n_layers=1, d_env=8):
    cfg = load_config(None, ["sim.device=cpu"])
    cfg.model.n_layers = n_layers
    cfg.model.d_env = d_env
    grid = Grid(cfg.grid, cfg.net, device="cpu")
    torch.manual_seed(0)
    return cfg, grid


@pytest.mark.parametrize("ansatz", ["hea", "dense"])
def test_circuit_columns_are_unit_norm(ansatz):
    cfg, grid = _cfg()
    circ = QubitCircuit(grid, cfg.model, n_cond_features=6, ansatz=ansatz)
    psi = circ(torch.randn(2, 6))
    assert psi.shape == (2, grid.M * 8, grid.M, 2)
    norms = (psi ** 2).sum(dim=(1, 3))
    assert torch.allclose(norms, torch.ones(2, grid.M), atol=1e-5)


def test_zero_angles_hea_is_a_permutation_and_dense_is_identity():
    cfg, grid = _cfg()
    for ansatz, expect_identity in (("hea", False), ("dense", True)):
        circ = QubitCircuit(grid, cfg.model, n_cond_features=6, ansatz=ansatz)
        with torch.no_grad():
            for p in circ.parameters():
                p.zero_()
        T = QubitTransportBlock.trace_out(circ(torch.zeros(1, 6)), grid.M, 8)[0]
        assert torch.allclose(T.max(0).values, torch.ones(grid.M), atol=1e-6)
        assert torch.allclose(T.sum(0), torch.ones(grid.M), atol=1e-6)
        if expect_identity:
            assert torch.allclose(T, torch.eye(grid.M), atol=1e-6)


@pytest.mark.parametrize("ansatz", ["hea", "dense"])
def test_transport_block_is_column_stochastic_and_selects_columns(ansatz):
    cfg, grid = _cfg()
    block = QubitTransportBlock(grid, cfg.model, n_cond_features=6,
                                ansatz=ansatz)
    cond = torch.randn(2, 6)
    full = block(cond)
    assert full.shape == (2, grid.M, grid.M)
    assert torch.all(full >= -1e-7)
    assert torch.allclose(full.sum(1), torch.ones(2, grid.M), atol=1e-5)
    idx = torch.tensor([[0, 3, 7], [2, 4, 1]])
    sel = block(cond, source_indices=idx)
    expected = full.gather(2, idx.unsqueeze(1).expand(-1, grid.M, -1))
    assert torch.allclose(sel, expected, atol=2e-6)


def test_checkpointed_training_forward_matches_eval_forward():
    cfg, grid = _cfg(n_layers=2)
    block = QubitTransportBlock(grid, cfg.model, n_cond_features=6,
                                ansatz="hea")
    cond = torch.randn(1, 6)
    block.eval()
    with torch.no_grad():
        ref = block(cond)
    block.train()
    out = block(cond)
    assert torch.allclose(out, ref, atol=1e-6)


def test_model_propagator_is_column_stochastic_and_trains():
    cfg, grid = _cfg(n_layers=2)
    model = QubitQTRO(grid, cfg.model, "hea")
    model.set_conditioning_scale(torch.ones(6))
    u = torch.rand(2, 2)
    xi = torch.rand(2, 4)
    P = model.propagator(u, xi)
    assert P.shape == (2, grid.M, grid.M)
    assert torch.allclose(P.sum(1), torch.ones(2, grid.M), atol=1e-5)
    P.sum().backward()
    grads = [p.grad for p in model.transport.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)


def test_model_state_dict_carries_the_qubit_circuit_under_the_transport_prefix():
    cfg, grid = _cfg(n_layers=3)
    model = QubitQTRO(grid, cfg.model, "dense")
    assert model.transport.circuit.n_layers == 3
    assert model.transport.circuit.ansatz == "dense"
    keys = set(model.state_dict())
    assert all(k == "cond_scale" or k.startswith(("transport.circuit.", "reset."))
               for k in keys)
    assert not any(k.startswith("transport.circuit.lo_") for k in keys)


def test_registers_partition_the_twelve_bits_in_layout_order():
    cfg, grid = _cfg()
    r = registers(grid, 8)
    assert r == {"env": [0, 1, 2], "phase": [3], "queue": [4, 5, 6, 7],
                 "window": [8, 9, 10, 11]}


def test_chain_links_and_dense_pairs_cover_the_register():
    assert chain_links(12) == [(i, i + 1) for i in range(11)]
    pairs = dense_pairs(12)
    assert len(pairs) == 66
    assert all(a > b for a, b in pairs)
