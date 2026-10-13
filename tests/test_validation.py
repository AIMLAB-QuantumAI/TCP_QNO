"""Validation at composed horizons, and the ranking the depth screen uses."""
from __future__ import annotations

import pytest
import torch

from qtro.config import load_config
from qtro.eval.validation import (
    depth_screen_score,
    rank_depth_variants,
    validation_metrics,
    validation_metrics_multihorizon,
)
from qtro.grid import Grid


class _FakeData:
    """Two conditions whose refs are exact powers of a known matrix."""

    def __init__(self, M: int, horizons, stay: float = 0.0):
        g = torch.Generator().manual_seed(7)
        P = torch.rand(2, M, M, generator=g) + 1e-3
        P = P / P.sum(dim=1, keepdim=True)
        if stay:
            # A dense random kernel mixes in about two steps, so every
            # perturbation collapses into the same stationary law and long
            # horizons stop discriminating. Adding self-transition mass puts
            # |lambda_2| near the real kernel's 0.976.
            eye = torch.eye(M).unsqueeze(0).expand(2, -1, -1)
            P = stay * eye + (1.0 - stay) * P
            P = P / P.sum(dim=1, keepdim=True)
        self.P = P
        self.u = torch.rand(2, 2, generator=g)
        self.xi = torch.rand(2, 4, generator=g)
        self.horizons = list(horizons)
        self.targets = {h: torch.linalg.matrix_power(self.P, h) for h in horizons}
        self.counts = {h: torch.full((2, M), 1000.0) for h in horizons}
        self.refs, self.ref_counts = {}, {}
        self.noise_fidelity, self.noise_tv = {}, {}
        for h in horizons:
            self.refs[h] = torch.linalg.matrix_power(self.P, h)
            self.ref_counts[h] = torch.full((2, M), 1000.0)
            self.noise_fidelity[h] = torch.ones(2, M)
            self.noise_tv[h] = torch.zeros(2, M)

    def __len__(self):
        return 2

    def batch(self, idx, device):
        return {
            "u": self.u[idx].to(device),
            "xi": self.xi[idx].to(device),
            "targets": {h: v[idx].to(device).float() for h, v in self.targets.items()},
            "counts": {h: v[idx].to(device) for h, v in self.counts.items()},
            "refs": {h: v[idx].to(device).float() for h, v in self.refs.items()},
            "ref_counts": {h: v[idx].to(device)
                           for h, v in self.ref_counts.items()},
            "noise_fidelity": {h: v[idx].to(device)
                               for h, v in self.noise_fidelity.items()},
            "noise_tv": {h: v[idx].to(device)
                         for h, v in self.noise_tv.items()},
        }


class _ExactModel(torch.nn.Module):
    def __init__(self, P):
        super().__init__()
        self.register_buffer("P", P)

    def propagator(self, u, xi):
        return self.P[: u.shape[0]]


def _grid_and_cfg():
    cfg = load_config(None, ["sim.device=cpu", "grid.n_w=4", "grid.n_q=4",
                             "grid.n_phase=2"])
    grid = Grid(cfg.grid, cfg.net, device="cpu")
    return grid, cfg


def test_a_model_that_is_the_kernel_scores_perfectly_at_every_horizon():
    """P^h must be compared against ref_h, not against ref_1."""
    grid, cfg = _grid_and_cfg()
    data = _FakeData(grid.M, (1, 5, 10, 20))
    model = _ExactModel(data.P)
    out = validation_metrics_multihorizon(
        model, data, grid, cfg, "cpu", horizons=(1, 5, 10, 20))
    assert set(out) == {"h1", "h5", "h10", "h20"}
    for key in out:
        got = out[key]["fidelity"]["occupancy_weighted"]
        assert got > 0.999, f"{key} fidelity {got}"


def test_the_default_reproduces_the_single_horizon_call():
    grid, cfg = _grid_and_cfg()
    data = _FakeData(grid.M, (1,))
    model = _ExactModel(data.P)
    out = validation_metrics_multihorizon(model, data, grid, cfg, "cpu")
    assert set(out) == {"h1"}


def test_composition_reveals_error_that_h1_hides():
    """An operator can look near-exact at h=1 and be badly wrong at h=20. A
    slow-mixing kernel is the relevant regime; a fast-mixing one washes any
    perturbation out into the stationary law and hides it again."""
    grid, cfg = _grid_and_cfg()
    data = _FakeData(grid.M, (1, 5, 10, 20), stay=0.97)
    smeared = data.P + 1e-3
    smeared = smeared / smeared.sum(dim=1, keepdim=True)
    out = validation_metrics_multihorizon(
        _ExactModel(smeared), data, grid, cfg, "cpu",
        horizons=(1, 5, 10, 20))
    f = {k: out[k]["fidelity"]["occupancy_weighted"] for k in out}
    assert f["h1"] > 0.99, "the perturbation must be invisible at h=1"
    assert f["h20"] < f["h1"] - 0.05, "composition must expose it"
    assert f["h1"] > f["h5"] > f["h10"] > f["h20"]


def test_one_step_validation_scores_against_the_fitting_targets():
    grid, cfg = _grid_and_cfg()
    cfg.data.c_min = 1
    data = _FakeData(grid.M, (1,))
    out = validation_metrics(_ExactModel(data.P), data, grid, cfg, "cpu")
    assert out["fidelity"]["occupancy_weighted"] > 0.999
    assert out["tv"]["uniform"] < 1e-5


def _record(variant, seed, f1, flong, tv=None):
    block = {f"h{h}": {"fidelity": {"occupancy_weighted": v},
                       "tv": {"occupancy_weighted": (1.0 - v) if tv is None else tv}}
             for h, v in (("1", f1), ("5", flong), ("10", flong), ("20", flong))}
    return {"variant": variant, "seed": seed, "validation_multihorizon": block}


def test_the_summary_ranks_on_the_long_horizons():
    """Selecting on h=1 alone is the error the screen exists to avoid."""
    records = [_record("8x4", 0, 0.99, 0.30), _record("2x16", 0, 0.90, 0.50)]
    ranked = rank_depth_variants(records)
    assert ranked[0]["variant"] == "2x16", "h=1 must not decide the ranking"


def test_seeds_are_averaged_and_counted():
    ranked = rank_depth_variants([_record("8x4", 0, 0.4, 0.4, tv=0.0),
                                  _record("8x4", 1, 0.6, 0.6, tv=0.0)])
    assert ranked[0]["n_seeds"] == 2
    assert abs(ranked[0]["score"] - 0.5) < 1e-9
    assert depth_screen_score(_record("x", 0, 0.1, 0.7, tv=0.1)) == pytest.approx(0.6)


def test_a_record_without_multihorizon_metrics_is_refused():
    with pytest.raises(KeyError):
        rank_depth_variants([{"variant": "8x4", "seed": 0,
                              "validation_multihorizon": None}])
