import pytest
import torch

from qtro.config import load_config
from qtro.eval.metrics import (
    column_values,
    configuration_column_aggregates,
    structural_validity,
    summarise_columns,
)
from qtro.grid import Grid


def _fixture():
    cfg = load_config(None, ["sim.device=cpu", "grid.n_w=2", "grid.n_q=2"])
    grid = Grid(cfg.grid, cfg.net, device="cpu")
    ref = torch.eye(grid.M).unsqueeze(0)
    pred = ref.clone()
    counts = torch.arange(1, grid.M + 1).float().unsqueeze(0)
    noise_f = torch.full_like(counts, 0.9)
    noise_tv = torch.full_like(counts, 0.1)
    return cfg, grid, pred, ref, counts, noise_f, noise_tv


def test_metrics_are_exact_on_a_perfect_prediction():
    _, grid, pred, ref, counts, nf, nt = _fixture()
    values = column_values(pred, ref, counts, nf, nt, grid, c_min=1)
    out = summarise_columns([values])
    assert out["fidelity"]["uniform"] == 1.0
    assert out["tv"]["uniform"] == 0.0
    assert out["mean_queue_abs_error"]["uniform"] == 0.0
    assert out["overflow_abs_error"]["uniform"] == 0.0
    assert out["queue_w1"]["uniform"] == 0.0
    assert out["cwnd_w1"]["uniform"] == 0.0
    assert abs(out["noise_fidelity"]["uniform"] - 0.9) < 1e-6


def test_axis_wasserstein_separates_queue_and_window_motion():
    cfg, grid, _, ref, counts, nf, nt = _fixture()
    pred = ref.clone()
    # Source cell 0 originally stays at (w-bin 0, q-bin 0, phase 0).
    # Move it only to q-bin 1 while preserving its window bin.
    destination = grid.cell_index(torch.tensor(0), torch.tensor(1),
                                  torch.tensor(0))
    pred[0, :, 0] = 0.0
    pred[0, destination, 0] = 1.0
    values = column_values(pred, ref, counts, nf, nt, grid, c_min=1)
    assert values["queue_w1"][0, 0] == pytest.approx(1.0)
    assert values["cwnd_w1"][0, 0] == pytest.approx(0.0)


def test_validity_detects_negative_mass_and_bad_column_sum():
    P = torch.eye(3).unsqueeze(0)
    P[0, 0, 0] = -0.2
    out = structural_validity(P, [1])['1']
    assert out["max_negative_mass"] == pytest.approx(0.2)
    assert out["max_column_mass_deviation"] == pytest.approx(1.2)
    assert out["has_nonfinite"] is False


def test_validity_reports_nonfinite_rollout_as_infinite_violation():
    P = torch.eye(2).unsqueeze(0)
    P[0, 0, 0] = float("nan")
    out = structural_validity(P, [1])["1"]
    assert out["has_nonfinite"] is True
    assert out["max_negative_mass"] == float("inf")
    assert out["max_column_mass_deviation"] == float("inf")


def test_configuration_column_aggregates_retain_one_value_per_configuration():
    score = torch.tensor([[1.0, .5, .25, 0.0], [.2, .8, .4, .6]])
    valid = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
    occupancy = torch.tensor([[1., 1., 2., 6.], [1., 3., 0., 0.]])
    tail = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=torch.bool)
    values = {
        "fidelity": score,
        "overflow_abs_error": 1.0 - score,
        "valid": valid,
        "occupancy": occupancy,
        "source_tail": tail,
    }
    out = configuration_column_aggregates(values)
    assert torch.allclose(out["fidelity"]["uniform"],
                          torch.tensor([.4375, .5]))
    assert torch.allclose(out["fidelity"]["occupancy_weighted"],
                          torch.tensor([.2, .65]))
    assert out["fidelity"]["tail_only"][0] == pytest.approx(.125)
    assert torch.isnan(out["fidelity"]["tail_only"][1])
