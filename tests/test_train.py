"""The training loop: memory guard, minibatch stream, schedules, budgets, learning."""
from __future__ import annotations

import pytest
import torch
from helpers import write_shards
from torch import nn

from qtro.config import load_config
from qtro.data import SweepData
from qtro.grid import Grid
from qtro.losses import total_loss
from qtro.train import (
    batch_index_generator,
    check_memory,
    estimate_activation_bytes,
    parse_lr_milestones,
    train,
    validated_lr_schedule,
)


def _cfg(**over):
    cfg = load_config(None, ["sim.device=cpu"])
    cfg.model.n_layers, cfg.model.d_env = 1, 2
    cfg.train.n_steps, cfg.train.batch_configs = 6, 2
    cfg.train.h0 = 2
    cfg.data.horizons_train = [1, 2]
    cfg.data.c_min = 1
    for k, v in over.items():
        section, _, field = k.partition(".")
        setattr(getattr(cfg, section), field, v)
    return cfg


class _ToyPropagator(nn.Module):
    """A column-stochastic matrix with free logits: the loop needs nothing more."""

    def __init__(self, M: int):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(M, M))

    def propagator(self, u, xi):
        return torch.softmax(self.logits, dim=0).unsqueeze(0).expand(u.shape[0], -1, -1)


def test_memory_estimate_matches_the_measured_points():
    """Anchored to the measured activation sizes, within 30%."""
    for batch, layers, denv, measured_gib in ((16, 8, 4, 6.59),
                                              (4, 8, 4, 1.67),
                                              (64, 8, 4, 26.28)):
        est = estimate_activation_bytes(batch, layers, denv) / 2**30
        assert abs(est - measured_gib) / measured_gib < 0.30, \
            (batch, layers, denv, est, measured_gib)


def test_memory_guard_is_a_noop_without_cuda():
    cfg = _cfg()
    info = check_memory(cfg, "cpu")
    assert info["checked"] is False


def test_sample_offset_continues_the_same_minibatch_stream():
    seed, n_data, batch, offset = 17, 23, 4, 7
    whole = batch_index_generator(seed, n_data, batch)
    expected = None
    for _ in range(offset + 1):
        expected = torch.randint(n_data, (batch,), generator=whole)
    resumed = batch_index_generator(seed, n_data, batch, offset)
    got = torch.randint(n_data, (batch,), generator=resumed)
    assert torch.equal(got, expected)


def test_sample_offset_rejects_negative_steps():
    with pytest.raises(ValueError, match="non-negative"):
        batch_index_generator(0, 10, 2, -1)


def test_lr_milestone_parser_and_validator_are_unambiguous():
    expected = {2000: 0.001, 4000: 0.0003}
    assert parse_lr_milestones("2000:0.001,4000:0.0003") == expected
    assert validated_lr_schedule(expected) == expected
    for bad in ("2000", "-1:0.1", "2:0", "2:nan", "2:0.1,2:0.2"):
        with pytest.raises(ValueError, match="learning-rate"):
            parse_lr_milestones(bad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_memory_guard_refuses_an_oversized_batch_before_the_first_step():
    cfg = _cfg()
    cfg.model.n_layers, cfg.model.d_env = 8, 4
    cfg.train.batch_configs = 4096          # far past any real card
    with pytest.raises(RuntimeError, match="Refusing to start"):
        check_memory(cfg, "cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_memory_guard_error_names_a_batch_that_would_fit():
    cfg = _cfg()
    cfg.model.n_layers, cfg.model.d_env = 8, 4
    cfg.train.batch_configs = 4096
    with pytest.raises(RuntimeError, match=r"batch_configs<=\d+"):
        check_memory(cfg, "cuda")


def _fixed_batch_loss(model, data, grid, cfg):
    """Loss on the whole dataset, so before/after are the same measurement."""
    idx = torch.arange(len(data))
    batch = data.batch(idx, "cpu")
    horizons = [h for h in cfg.data.horizons_train if h <= cfg.train.h0]
    with torch.no_grad():
        P = model.propagator(batch["u"], batch["xi"])
        loss, _ = total_loss(P, {h: batch["targets"][h] for h in horizons},
                             batch["counts"][horizons[0]], grid, cfg)
    return float(loss)


@pytest.mark.parametrize("build", ["qtro", "cnto"])
def test_training_lowers_a_fixed_batch_loss(tmp_path, build):
    """Both models learn, and both go through the one shared loop."""
    write_shards(tmp_path, banded=True)
    cfg = _cfg()
    cfg.train.n_steps, cfg.train.lr = 120, 0.05
    data = SweepData(tmp_path, horizons=[1, 2])
    grid = Grid(cfg.grid, cfg.net, device="cpu")
    if build == "qtro":
        from qtro.model.qtro import QTRO
        model = QTRO(grid, cfg.model)
    else:
        from qtro.baselines.cnto import CNTO
        from qtro.model.qtro import N_COND_FEATURES
        model = CNTO(grid, N_COND_FEATURES)
    before = _fixed_batch_loss(model, data, grid, cfg)
    train(model, data, grid, cfg, seed=0)
    after = _fixed_batch_loss(model, data, grid, cfg)
    assert after < 0.9 * before, (build, before, after)


def test_h0_above_the_available_horizons_is_an_error(tmp_path):
    write_shards(tmp_path, horizons=(1,))
    cfg = _cfg()
    cfg.data.horizons_train = [1]
    cfg.train.h0 = 0
    data = SweepData(tmp_path, horizons=[1])
    grid = Grid(cfg.grid, cfg.net, device="cpu")
    with pytest.raises(ValueError, match="excludes every training horizon"):
        train(_ToyPropagator(grid.M), data, grid, cfg)


def test_run_reports_what_the_column_threshold_dropped(tmp_path):
    write_shards(tmp_path)
    cfg = _cfg()
    cfg.data.c_min = 400          # discard most columns
    data = SweepData(tmp_path, horizons=[1, 2])
    grid = Grid(cfg.grid, cfg.net, device="cpu")
    out = train(_ToyPropagator(grid.M), data, grid, cfg,
                sample_offset_steps=11,
                lr_schedule={11: 0.01, 14: 0.001})
    drop = out["columns_dropped"]
    assert drop["n_dropped"] > 0
    assert 0.0 < drop["occupancy_share_dropped"] < 1.0
    assert out["sample_offset_steps"] == 11
    assert out["completed_steps"] == 11 + cfg.train.n_steps
    assert out["history"][0]["step"] == 11
    assert out["history"][0]["lr"] == 0.01
    assert out["history"][3]["lr"] == 0.001
    assert out["lr_schedule"] == {11: 0.01, 14: 0.001}


def test_wall_budget_stops_on_a_completed_update(tmp_path):
    write_shards(tmp_path)
    cfg = _cfg()
    cfg.train.n_steps = 20
    data = SweepData(tmp_path, horizons=[1, 2])
    grid = Grid(cfg.grid, cfg.net, device="cpu")
    out = train(_ToyPropagator(grid.M), data, grid, cfg, max_wall_s=1.0e-12)
    assert out["stopped_by_wall"] is True
    assert out["completed_steps"] == 1
    assert out["wall_s"] >= out["max_wall_s"]
    assert out["wall_overshoot_s"] >= 0.0


@pytest.mark.parametrize("budget", [0.0, -1.0, float("inf")])
def test_invalid_wall_budget_is_rejected(tmp_path, budget):
    write_shards(tmp_path)
    cfg = _cfg()
    data = SweepData(tmp_path, horizons=[1, 2])
    grid = Grid(cfg.grid, cfg.net, device="cpu")
    with pytest.raises(ValueError, match="max_wall_s"):
        train(_ToyPropagator(grid.M), data, grid, cfg, max_wall_s=budget)


def test_history_records_the_loss_parts_and_the_learning_rate(tmp_path):
    write_shards(tmp_path)
    cfg = _cfg()
    data = SweepData(tmp_path, horizons=[1, 2])
    grid = Grid(cfg.grid, cfg.net, device="cpu")
    out = train(_ToyPropagator(grid.M), data, grid, cfg, seed=3)
    assert out["seed"] == 3
    first = out["history"][0]
    assert {"step", "loss", "dist", "obs", "wasserstein", "multitail", "reg", "lr"} <= set(first)
    assert out["final"] == out["history"][-1]
