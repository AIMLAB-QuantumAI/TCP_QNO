"""Behaviour of the ns-3 drivers that the --help test cannot see."""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

from qtro.config import load_config
from qtro.grid import Grid

ROOT = Path(__file__).resolve().parents[1]
COND = {"capacity_pps": 2000.0, "rtt_prop_s": 0.03, "n_flows": 4, "p_err": 0.0,
        "p_max": 0.1, "min_th": 20.0, "max_th_frac": 0.5, "buffer_pkts": 90.0,
        "w_q": 1e-4}


def _load(script: Path):
    spec = importlib.util.spec_from_file_location(script.stem, script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _unreachable_ns3(tmp_path):
    env = {**os.environ, "NS3_DIR": str(tmp_path / "no-such-ns3"), "NS3_MODE": "native"}
    env.pop("NS3_USER", None)
    return env


def test_domain_controllers_receive_the_same_objective_settings():
    closed_loop = _load(ROOT / "scripts" / "control" / "closed_loop.py")
    cfg = load_config(None, ["sim.device=cpu"])
    grid = Grid(cfg.grid, cfg.net, device="cpu")
    args = argparse.Namespace(models="", with_domain=True, horizon=5, q_target=20.0,
                              lam=1.0, risk="threshold", q95_limit=35.0, lam_move=0.02,
                              init_p_max=0.05, init_min_th=26.0, seed_index=0)
    controllers = closed_loop.make_controllers(args, cfg, grid, "cpu")
    assert set(controllers) == {"fixed", "fluid", "diffusion"}
    for label in ("fluid", "diffusion"):
        controller = controllers[label]
        assert (controller.risk, controller.q95_limit, controller.lam_move) == (
            "threshold", 35.0, 0.02)


def test_generate_dataset_fails_loudly_when_no_run_returns_data(tmp_path):
    conditions = tmp_path / "conditions.json"
    conditions.write_text(json.dumps([COND]), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "data" / "generate_dataset.py"),
         "--conditions", str(conditions), "--out", str(tmp_path / "out"),
         "--replications", "2", "--samples", "5", "--workers", "1", "--timeout-s", "60"],
        cwd=tmp_path, capture_output=True, text=True, env=_unreachable_ns3(tmp_path),
        timeout=300)
    assert result.returncode != 0
    assert "no ns-3 run returned data" in result.stderr or "cannot start" in result.stderr


def test_generate_load_steps_fails_loudly_when_no_run_returns_data(tmp_path):
    condition = {**COND, "name": "probe", "warmup_s": 1.0, "sample_interval_s": 0.01,
                 "samples": 5, "surge_flows": 2, "surge_start_s": 0.02}
    conditions = tmp_path / "steps.json"
    conditions.write_text(json.dumps([condition]), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "data" / "generate_load_steps.py"),
         "--conditions", str(conditions), "--out", str(tmp_path / "out"),
         "--replications", "2", "--workers", "1", "--timeout-s", "60"],
        cwd=tmp_path, capture_output=True, text=True, env=_unreachable_ns3(tmp_path),
        timeout=300)
    assert result.returncode != 0
    assert "no ns-3 run returned data" in result.stderr or "cannot start" in result.stderr
