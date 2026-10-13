from dataclasses import asdict

import pytest
import torch

from qtro.baselines.cnto import CNTO
from qtro.checkpoint import load_checkpoint, save_checkpoint, upgrade_config_dict
from qtro.config import load_config
from qtro.grid import Grid
from qtro.model.qtro import N_COND_FEATURES, QTRO


def _small():
    cfg = load_config(None, ["sim.device=cpu", "grid.n_w=2", "grid.n_q=2",
                             "model.n_layers=1", "model.d_env=2"])
    return cfg, Grid(cfg.grid, cfg.net, device="cpu")


def test_checkpoint_round_trip_keeps_resolved_config_and_conditioning_scale(tmp_path):
    cfg, grid = _small()
    model = QTRO(grid, cfg.model)
    scale = torch.tensor([.4, 24, 4000, .05, 8, .001])
    model.set_conditioning_scale(scale)
    path = save_checkpoint(tmp_path / "model.pt", model, cfg,
                           model_name="qtro", metadata={"seed": 7})
    got, got_grid, got_cfg, payload = load_checkpoint(path, "cpu")
    assert got_grid.M == grid.M
    assert got_cfg.model.n_layers == 1
    assert payload["metadata"]["seed"] == 7
    assert torch.equal(got.cond_scale, scale)
    for key, value in model.state_dict().items():
        assert torch.equal(value, got.state_dict()[key])


def test_conditioning_scale_rejects_zero_or_wrong_length():
    cfg, grid = _small()
    model = QTRO(grid, cfg.model)
    for bad in (torch.ones(5), torch.tensor([1., 1., 1., 1., 1., 0.])):
        with pytest.raises(ValueError):
            model.set_conditioning_scale(bad)


def test_state_dict_keys_are_transport_reset_and_scale(tmp_path):
    cfg, grid = _small()
    model = QTRO(grid, cfg.model)
    path = save_checkpoint(tmp_path / "m.pt", model, cfg, model_name="qtro")
    got, _, _, _ = load_checkpoint(path, "cpu")
    keys = set(got.state_dict())
    assert keys == set(model.state_dict())
    assert all(k == "cond_scale" or k.startswith(("transport.circuit.", "reset."))
               for k in keys)


def _legacy_payload(model, cfg, model_name, model_kwargs=None):
    config = asdict(cfg)
    config["model"].update({
        "n_transport_experts": 1, "transport_gate_init": 0.5,
        "transport_residual_mode": "none", "transport_move_init": 0.1,
        "reset_init_probability": -1.0, "reset_neighborhood_radius": 0,
        "base_channel_mode": "none", "base_channel_radius": 1,
        "base_channel_allow_phase": True, "base_channel_identity_init": 0.9,
        "base_channel_init": "identity", "base_gate_init": 0.5,
        "local_channel_mode": "none", "local_init_probability": 0.02,
        "local_max_probability": 0.25, "local_gate_init": 0.1,
        "qubit_topology": "chain"})
    config["train"].update({
        "lambda_tv": 0.0, "lambda_entropy": 0.0, "entropy_bias_correct": True,
        "macro_balanced_distribution": False, "balance_memory_modes": False,
        "lambda_action_contrast": 0.0, "action_contrast_risks": [0.0, 1.0],
        "action_rank_weight": 0.25, "action_rank_margin": 0.25,
        "action_min_signal": 0.5, "grad_checkpoint": False})
    config["aqm"].update({"p_max": 0.1, "min_th": 20.0})
    config["net"].update({"n_flows": 4, "p_err": 0.0})
    config["sim"].update({"n_snapshots": 1000, "warmup_s": 8.0,
                          "n_replications": 16384, "rto_s": 0.2, "seed": 0})
    config["data"]["horizons_eval"] = [1, 5, 10, 20]
    return {"format_version": 1, "model_name": model_name,
            "model_kwargs": dict(model_kwargs or {}), "config": config,
            "state_dict": model.state_dict(), "metadata": {}, "training": {}}


def test_a_research_checkpoint_loads_under_the_new_names(tmp_path):
    cfg, grid = _small()
    torch.manual_seed(0)
    model = QTRO(grid, cfg.model)
    path = tmp_path / "qkfp_L1_dE2_seed0.pt"
    torch.save(_legacy_payload(model, cfg, "qkfp"), path)
    loaded, _, loaded_cfg, payload = load_checkpoint(path, "cpu")
    assert payload["model_name"] == "qtro" and isinstance(loaded, QTRO)
    assert not hasattr(loaded_cfg.model, "base_channel_mode")
    assert loaded_cfg.sim.device == "cpu"
    u = torch.tensor([[0.1, 20.0]])
    xi = torch.tensor([[2000.0, 0.03, 4.0, 0.0]])
    with torch.no_grad():
        assert torch.equal(loaded.propagator(u, xi), model.propagator(u, xi))


def test_a_research_cnto_checkpoint_maps_softmax_to_cnto(tmp_path):
    cfg, grid = _small()
    model = CNTO(grid, N_COND_FEATURES, rank=3)
    path = tmp_path / "softmax_rank3_seed0.pt"
    torch.save(_legacy_payload(model, cfg, "softmax", {"rank": 3}), path)
    loaded, _, _, payload = load_checkpoint(path, "cpu")
    assert payload["model_name"] == "cnto" and isinstance(loaded, CNTO)


def test_a_checkpoint_that_used_a_removed_feature_is_refused(tmp_path):
    cfg, grid = _small()
    payload = _legacy_payload(QTRO(grid, cfg.model), cfg, "qkfp")
    payload["config"]["model"]["local_channel_mode"] = "compose"
    path = tmp_path / "local.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="local_channel_mode"):
        load_checkpoint(path, "cpu")


def test_upgrade_treats_negative_reset_init_as_inert_and_positive_as_live():
    assert upgrade_config_dict({"model": {"reset_init_probability": -1.0, "d_env": 4}}) == {
        "model": {"d_env": 4}}
    with pytest.raises(ValueError, match="reset_init_probability"):
        upgrade_config_dict({"model": {"reset_init_probability": 0.02}})


def test_save_writes_canonical_names_and_loads_with_weights_only(tmp_path):
    cfg, grid = _small()
    model = CNTO(grid, N_COND_FEATURES, rank=2)
    path = save_checkpoint(tmp_path / "m.pt", model, cfg, model_name="softmax",
                           model_kwargs={"rank": 2}, metadata={"seed": 0},
                           training={"history": [{"step": 0, "loss": 1.0}]})
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["model_name"] == "cnto"
