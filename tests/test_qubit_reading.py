"""The Givens ladder is a twelve-qubit circuit: sweep by sweep, on the shipped grid."""
import torch

from qtro.config import load_config
from qtro.grid import Grid
from qtro.quantum.circuit import ConditionedCircuit
from qtro.quantum.givens import apply_sweep
from qtro.quantum.pairs import build_sweeps
from qtro.quantum.qubit import (
    apply_ucr,
    gate_count,
    n_qubits,
    r_y,
    register_shift_permutation,
    registers,
    sweep_as_circuit,
    sweep_gate_layout,
)


def _grid(**overrides):
    cfg = load_config(None, ["sim.device=cpu"])
    for key, value in overrides.items():
        setattr(cfg.grid, key, value)
    return cfg, Grid(cfg.grid, cfg.net, device="cpu")


def test_the_shipped_grid_is_nine_system_and_three_environment_qubits():
    _, grid = _grid()
    regs = registers(grid, 8)
    assert regs == {"env": [0, 1, 2], "phase": [3], "queue": [4, 5, 6, 7],
                    "window": [8, 9, 10, 11]}
    counts = gate_count(regs)
    assert counts["qubits"] == 12
    assert counts["system_qubits"] == 9 and counts["env_qubits"] == 3
    assert counts["cnot_per_sweep"] == 2048 == grid.M * 8 // 2


def test_a_sweep_has_one_angle_per_control_value():
    _, grid = _grid()
    regs = registers(grid, 8)
    for name, lo, _ in build_sweeps(grid, 8):
        target, controls = sweep_gate_layout(regs, name)
        assert len(controls) == 11 and target not in controls
        slot = torch.zeros_like(lo)
        for bit in controls:
            slot = slot * 2 + ((lo >> bit) & 1)
        # Distinct pairs read distinct control values, and the values come
        # in the order build_sweeps enumerates the pairs.
        assert slot.unique().numel() == lo.numel()
        assert torch.equal(slot.argsort(), torch.arange(lo.numel()))


def test_register_shift_is_an_adder():
    perm = register_shift_permutation([0, 1], +1, 3)
    # |r> -> |r+1 mod 4> on bits (0, 1); bit 2 untouched.
    x = torch.arange(8)
    r = x & 3
    assert torch.equal(perm & 3, (r - 1) % 4)
    assert torch.equal(perm >> 2, x >> 2)
    back = register_shift_permutation([0, 1], -1, 3)
    assert torch.equal(perm[back], x)


def test_ucr_selects_the_gate_by_the_control_value():
    torch.manual_seed(0)
    n = 3
    state = torch.randn(1, 8, 2, dtype=torch.complex64)
    angles = torch.tensor([[0.3, -1.1, 0.7, 2.0]])
    gates = r_y(angles).to(torch.complex64)
    out = apply_ucr(state, gates, target=0, controls=[2, 1], n=n)
    for hi in range(4):                      # control value = bits (2, 1)
        lo0, lo1 = hi * 2, hi * 2 + 1
        pair = torch.stack([state[0, lo0], state[0, lo1]])
        expect = gates[0, hi] @ pair
        assert torch.allclose(out[0, lo0], expect[0], atol=1e-6)
        assert torch.allclose(out[0, lo1], expect[1], atol=1e-6)


def test_every_sweep_equals_its_qubit_circuit_on_the_shipped_grid():
    torch.manual_seed(0)
    _, grid = _grid()
    d_env = 8
    regs = registers(grid, d_env)
    N, M = grid.M * d_env, grid.M
    psi = torch.zeros(1, N, M)
    psi[0, torch.arange(M) * d_env, torch.arange(M)] = 1.0
    state = psi.to(torch.complex64)
    for name, lo, hi in build_sweeps(grid, d_env):
        angles = 0.8 * torch.randn(1, lo.numel())
        psi = apply_sweep(psi, lo, hi, angles)
        state = sweep_as_circuit(state, name, lo, angles, regs)
        assert torch.allclose(state.imag, torch.zeros_like(state.imag))
        assert torch.allclose(state.real, psi, atol=1e-5), name


def test_the_conditioned_circuit_is_the_composition_of_those_gates():
    torch.manual_seed(0)
    cfg, grid = _grid()
    cfg.model.n_layers, cfg.model.d_env = 2, 8
    circ = ConditionedCircuit(grid, cfg.model, n_cond_features=6)
    cond = torch.randn(2, 6)
    with torch.no_grad():
        ref = circ(cond)
        regs = registers(grid, 8)
        h = circ.encoder(cond)
        N, M = grid.M * 8, grid.M
        state = torch.zeros(2, N, M, dtype=torch.complex64)
        state[:, torch.arange(M) * 8, torch.arange(M)] = 1.0
        sweeps = build_sweeps(grid, 8)
        for layer in range(circ.n_layers):
            cond_angles = h @ circ.cond_heads[layer].T
            train_angles = circ.trainable[layer].unsqueeze(0).expand(2, -1)
            for angles in (cond_angles, train_angles):
                for k, (name, lo, _) in enumerate(sweeps):
                    start, count = circ.offsets[k]
                    state = sweep_as_circuit(
                        state, name, lo, angles[:, start:start + count], regs)
    assert torch.allclose(state.real, ref, atol=1e-4)
    assert n_qubits(regs) == 12
