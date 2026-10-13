"""The Givens ladder as uniformly controlled rotations, and its control dial."""
import pytest
import torch

from qtro.config import load_config
from qtro.grid import Grid
from qtro.model.qtro import QTRO
from qtro.quantum.circuit import ConditionedCircuit
from qtro.quantum.givens import apply_sweep
from qtro.quantum.qubit import register_shift_permutation, registers
from qtro.qubit.simulator import apply_ucr, real_block, rz_ry
from qtro.qubit.ucr import UCRQTRO, angle_map, sweep_controls


def _cfg(n_layers=1, d_env=8, controls=-1, **grid_overrides):
    cfg = load_config(None, ["sim.device=cpu"])
    cfg.model.n_layers = n_layers
    cfg.model.d_env = d_env
    cfg.model.qubit_controls = controls
    for key, value in grid_overrides.items():
        setattr(cfg.grid, key, value)
    grid = Grid(cfg.grid, cfg.net, device="cpu")
    torch.manual_seed(0)
    return cfg, grid


def test_all_controls_is_the_givens_ladder_parameter_for_parameter():
    cfg, grid = _cfg()
    amap, counts = angle_map(grid, 8, -1)
    ladder = ConditionedCircuit(grid, cfg.model, n_cond_features=6)
    assert counts == [count for _, count in ladder.offsets]
    assert sum(counts) == ladder.n_pairs == 13312
    assert torch.equal(amap, torch.arange(ladder.n_pairs))


def test_slot_counts_double_per_control_until_the_axis_runs_out():
    _, grid = _cfg()
    assert angle_map(grid, 8, 0)[1] == [1] * 7
    assert angle_map(grid, 8, 1)[1] == [2] * 7
    # Three controls: the pair's own axis. Odd sweeps have one pair fewer
    # (no wrap-around pair), so one slot fewer.
    assert angle_map(grid, 8, 3)[1] == [8, 7, 8, 7, 8, 8, 6]


def test_control_order_is_own_axis_then_phase_then_other_axis_then_env():
    _, grid = _cfg()
    by_name = {sw["name"]: sw for sw in sweep_controls(grid, 8, -1)}
    regs = registers(grid, 8)
    assert by_name["w_even"]["target"] == regs["window"][0]
    assert by_name["q_odd"]["target"] == regs["queue"][0]
    assert by_name["env_even"]["target"] == regs["env"][0]
    assert by_name["phase"]["target"] == regs["phase"][0]
    four = {sw["name"]: sw["controls"] for sw in sweep_controls(grid, 8, 4)}
    assert four["w_even"] == sorted(regs["window"][1:] + regs["phase"],
                                    reverse=True)
    assert four["q_even"] == sorted(regs["queue"][1:] + regs["phase"],
                                    reverse=True)


def _ucr_sweep(state, sw, params, regs, n):
    """One sweep from qubit primitives: (decrement) UCR (increment)."""
    odd = sw["name"].endswith("_odd")
    axis = {"w": "window", "q": "queue", "env": "env"}.get(
        sw["name"].split("_")[0])
    # An odd sweep's rotation always reads the axis's own high bits, so that
    # the wrapped pair (top of the axis, bottom of the axis) stays untouched.
    controls = sorted(set(sw["controls"])
                      | (set(regs[axis][1:]) if odd else set()), reverse=True)
    slot = torch.zeros_like(sw["lo"])
    for bit in controls:
        slot = slot * 2 + ((sw["lo"] >> bit) & 1)
    table = torch.zeros(2 ** len(controls))
    table[slot] = params[sw["index"]]
    gates = real_block(rz_ry(torch.zeros_like(table), 2 * table))[None]
    if odd:
        bits = regs[axis]
        state = state[:, register_shift_permutation(bits, -1, n)]
    state = apply_ucr(state, gates, sw["target"], controls, n)
    if odd:
        state = state[:, register_shift_permutation(bits, +1, n)]
    return state


@pytest.mark.parametrize("controls", [0, 1, 2, 3, -1])
def test_ucr_from_qubit_gates_matches_the_givens_sweep(controls):
    cfg, grid = _cfg(d_env=2, n_w=4, n_q=4, n_phase=2)
    d_env = 2
    regs = registers(grid, d_env)
    n = sum(len(v) for v in regs.values())
    N, M = grid.M * d_env, grid.M
    sweeps = sweep_controls(grid, d_env, controls)
    n_slots = sum(sw["n_slots"] for sw in sweeps)
    params = 0.7 * torch.randn(n_slots)
    offsets = torch.cumsum(torch.tensor(
        [0] + [sw["n_slots"] for sw in sweeps[:-1]]), 0)

    psi = torch.zeros(1, N, M)
    psi[0, torch.arange(M) * d_env, torch.arange(M)] = 1.0
    state = torch.zeros(1, N, M, 2)
    state[..., 0] = psi[0]
    for sw, off in zip(sweeps, offsets.tolist()):
        theta = params[sw["index"] + off]
        psi = apply_sweep(psi, sw["lo"], sw["hi"], theta[None])
        state = _ucr_sweep(state, sw, params[off:off + sw["n_slots"]],
                           regs, n)
    assert torch.allclose(state[..., 1], torch.zeros(1, N, M), atol=1e-6)
    assert torch.allclose(state[..., 0], psi, atol=1e-5)


def test_k_minus_one_loads_a_qtro_state_dict_and_agrees():
    cfg, grid = _cfg(n_layers=2)
    qtro = QTRO(grid, cfg.model)
    ucr = UCRQTRO(grid, cfg.model)
    assert {k: v.shape for k, v in ucr.state_dict().items()} == \
        {k: v.shape for k, v in qtro.state_dict().items()}
    ucr.load_state_dict(qtro.state_dict(), strict=True)
    for m in (qtro, ucr):
        m.set_conditioning_scale(torch.ones(6))
    u, xi = torch.rand(2, 2), torch.rand(2, 4)
    assert torch.allclose(ucr.propagator(u, xi), qtro.propagator(u, xi),
                          atol=1e-6)


@pytest.mark.parametrize("controls", [0, 2, 5])
def test_fewer_controls_is_column_stochastic_and_trains(controls):
    cfg, grid = _cfg(n_layers=2, controls=controls)
    model = UCRQTRO(grid, cfg.model)
    model.set_conditioning_scale(torch.ones(6))
    assert model.transport.circuit.n_controls == controls
    P = model.propagator(torch.rand(2, 2), torch.rand(2, 4))
    assert torch.all(P >= -1e-7)
    assert torch.allclose(P.sum(1), torch.ones(2, grid.M), atol=1e-5)
    P.sum().backward()
    grads = [p.grad for p in model.transport.circuit.trainable]
    assert all(g is not None and g.shape == (model.transport.circuit.n_slots,)
               for g in grads)


def test_zero_controls_shares_one_angle_per_sweep():
    cfg, grid = _cfg(controls=0)
    circ = UCRQTRO(grid, cfg.model).transport.circuit
    assert circ.n_slots == 7
    expanded = circ.trainable[0][circ.angle_map]
    for k, (start, count) in enumerate(circ.offsets):
        block = expanded[start:start + count]
        assert torch.allclose(block, block[0].expand_as(block))
