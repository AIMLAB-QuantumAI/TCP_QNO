import json

import numpy as np
import pytest
import torch
from helpers import write_shards

from qtro.baselines.cnto import CNTO
from qtro.checkpoint import save_checkpoint
from qtro.config import load_config
from qtro.data import SweepData
from qtro.eval.panel import (
    parse_model_specs,
    score_checkpoint_directory,
    score_kernels,
    score_run_directory,
    split_half_floor,
)
from qtro.grid import Grid
from qtro.model.qtro import N_COND_FEATURES


def _panel(tmp_path):
    write_shards(tmp_path / "data", n_conditions=4, horizons=(1,), splits=("ns3_heldout",))
    cfg = load_config(None, ["sim.device=cpu"])
    grid = Grid(cfg.grid, cfg.net, device="cpu")
    panel = SweepData(tmp_path / "data", horizons=[1], splits=("ns3_heldout",),
                      want_refs=True, require_targets=False)
    return cfg, grid, panel


def test_score_kernels_returns_one_value_per_condition_and_aggregation(tmp_path):
    cfg, grid, panel = _panel(tmp_path)
    identity = torch.eye(grid.M)

    def kernels_for(batch):
        return {1: identity.expand(batch["u"].shape[0], -1, -1)}

    scores = score_kernels(kernels_for, panel, grid, cfg.data.c_min, "cpu", (1,),
                           ("fidelity", "tv"))
    assert set(scores) == {1} and set(scores[1]) == {"fidelity", "tv"}
    for aggregation in ("uniform", "occupancy_weighted", "tail_only"):
        values = scores[1]["fidelity"][aggregation]
        assert isinstance(values, np.ndarray) and values.shape == (4,)
        assert np.all((values >= 0) & (values <= 1))


def test_split_half_floor_reports_fidelity_and_tv_per_horizon(tmp_path):
    cfg, _, panel = _panel(tmp_path)
    floor = split_half_floor(panel, (1,), cfg.data.c_min)
    assert set(floor) == {1}
    fid, tv = floor[1]
    assert 0.0 <= fid <= 1.0 and 0.0 <= tv <= 1.0


def test_checkpoints_are_scored_by_seed_from_either_convention(tmp_path):
    cfg, grid, panel = _panel(tmp_path)
    model = CNTO(grid, N_COND_FEATURES, rank=2)
    run = tmp_path / "run"
    checkpoint = save_checkpoint(run / "checkpoints" / "cnto_rank2_seed3.pt", model, cfg,
                                 model_name="cnto", model_kwargs={"rank": 2},
                                 metadata={"seed": 3})
    (run / "train_cnto_rank2.json").write_text(
        json.dumps([{"checkpoint": str(checkpoint), "seed": 3}]), encoding="utf-8")
    by_dir = score_checkpoint_directory(run, panel, "cpu", (1,), ("fidelity",))
    by_manifest = score_run_directory(run, panel, "cpu", (1,), ("fidelity",))
    assert set(by_dir) == set(by_manifest) == {3}
    assert np.array_equal(by_dir[3][1]["fidelity"]["occupancy_weighted"],
                          by_manifest[3][1]["fidelity"]["occupancy_weighted"])
    assert score_checkpoint_directory(run, panel, "cpu", (1,), ("fidelity",), seeds={0}) == {}


def test_model_specs_keep_their_order_and_reject_empty_entries():
    assert list(parse_model_specs("a=x,b=y").items()) == [("a", "x"), ("b", "y")]
    with pytest.raises(ValueError, match="name=path"):
        parse_model_specs("a=")
