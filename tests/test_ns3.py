import argparse

import numpy as np
import pytest

from qtro.config import load_config
from qtro.grid import Grid
from qtro.ns3 import (
    Ns3Runner,
    add_ns3_arguments,
    condition_flags,
    csv_rows,
    digitise,
    extra_flags,
)

NS3 = "/opt/ns-allinone-3.42/ns-3.42"
COND = {"capacity_pps": 2000.0, "rtt_prop_s": 0.03, "n_flows": 4, "p_err": 0.0,
        "p_max": 0.1, "min_th": 20.0, "max_th_frac": 0.5, "buffer_pkts": 90.0,
        "w_q": 1e-4}


def test_condition_flags_follow_the_documented_order():
    assert condition_flags(COND) == (
        "--capacityPps=2000.0 --rttPropS=0.03 --nFlows=4 --pErr=0.0 --pMax=0.1 "
        "--minTh=20.0 --maxThFrac=0.5 --bufferPkts=90.0 --qw=0.0001")


def test_condition_values_must_be_numbers():
    # Condition values are interpolated into a shell command, as root in WSL.
    with pytest.raises(ValueError, match="capacity_pps"):
        condition_flags({**COND, "capacity_pps": '2000"; rm -rf /; echo "'})
    with pytest.raises(ValueError, match="n_flows"):
        condition_flags({**COND, "n_flows": True})


def test_extra_flags_accept_only_plain_names_and_numbers():
    assert extra_flags({"burstFlows": 2, "burstShape": 1.5}) == " --burstFlows=2 --burstShape=1.5"
    assert extra_flags({}) == ""
    with pytest.raises(ValueError, match="burst flows"):
        extra_flags({"burst flows": 2})
    with pytest.raises(ValueError, match="burstFlows"):
        extra_flags({"burstFlows": "2 && echo"})


def test_paths_are_quoted_and_program_names_checked():
    runner = Ns3Runner("/opt/ns 3", mode="wsl", user="ns3")
    assert runner.wrapper_command("qtro-dumbbell", "--x=1") == (
        "cd '/opt/ns 3' && ./ns3 run 'qtro-dumbbell --x=1' --no-build")
    assert runner.binary_command("qtro-closedloop", "--x=1") == (
        "LD_LIBRARY_PATH='/opt/ns 3/build/lib' '/opt/ns 3/build/scratch/"
        "ns3.42-qtro-closedloop-optimized' --x=1")
    with pytest.raises(ValueError, match="program"):
        runner.wrapper_command("qtro-dumbbell; rm -rf /", "--x=1")


def test_wsl_wrapper_argv_runs_as_the_build_owner():
    runner = Ns3Runner(NS3, mode="wsl", distro="Ubuntu-24.04", user="ns3")
    assert runner.wrapper_argv("qtro-dumbbell", "--samples=600 --runSeed=1") == [
        "wsl", "-d", "Ubuntu-24.04", "--user", "root", "--", "bash", "-lc",
        "su ns3 -c \"cd /opt/ns-allinone-3.42/ns-3.42 && ./ns3 run "
        "'qtro-dumbbell --samples=600 --runSeed=1' --no-build\" 2>/dev/null"]


def test_wsl_binary_argv_can_close_stdin_for_a_probe():
    runner = Ns3Runner(NS3, mode="wsl", user="ns3")
    argv = runner.binary_argv("qtro-closedloop", "--samples=1200", close_stdin=True)
    assert argv[-1] == (
        "su ns3 -c 'LD_LIBRARY_PATH=/opt/ns-allinone-3.42/ns-3.42/build/lib "
        "/opt/ns-allinone-3.42/ns-3.42/build/scratch/ns3.42-qtro-closedloop-optimized "
        "--samples=1200 < /dev/null' 2>/dev/null")


def test_wsl_without_a_user_runs_the_command_as_given():
    runner = Ns3Runner(NS3, mode="wsl")
    assert runner.binary_argv("qtro-closedloop", "--x=1")[-1] == (
        "LD_LIBRARY_PATH=/opt/ns-allinone-3.42/ns-3.42/build/lib "
        "/opt/ns-allinone-3.42/ns-3.42/build/scratch/ns3.42-qtro-closedloop-optimized --x=1")


def test_native_mode_runs_bash_on_this_machine():
    runner = Ns3Runner("/home/me/ns-3.42")
    assert runner.wrapper_argv("qtro-dumbbell", "--x=1") == [
        "bash", "-lc", "cd /home/me/ns-3.42 && ./ns3 run 'qtro-dumbbell --x=1' --no-build"]


def test_missing_ns3_dir_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("NS3_DIR", raising=False)
    parser = argparse.ArgumentParser()
    add_ns3_arguments(parser)
    with pytest.raises(SystemExit, match="NS3_DIR"):
        Ns3Runner.from_args(parser.parse_args([]))


def test_environment_supplies_the_defaults(monkeypatch):
    monkeypatch.setenv("NS3_DIR", "/x")
    monkeypatch.setenv("NS3_MODE", "wsl")
    monkeypatch.setenv("NS3_USER", "ns3")
    parser = argparse.ArgumentParser()
    add_ns3_arguments(parser)
    runner = Ns3Runner.from_args(parser.parse_args([]))
    assert (runner.ns3_dir, runner.mode, runner.user) == ("/x", "wsl", "ns3")


def test_csv_rows_keep_data_rows_or_protocol_rows():
    text = "ns-3 banner\n0.01,1,2,0\nS,0.02,3,4,0,0,0,5.5,2\nCTRL,1,2\n"
    assert csv_rows(text) == [["0.01", "1", "2", "0"]]
    assert csv_rows(text, "S,")[0][1] == "0.02"


def test_digitise_maps_rows_onto_grid_cells():
    cfg = load_config(None, ["sim.device=cpu"])
    grid = Grid(cfg.grid, cfg.net, device="cpu")
    cells = digitise(np.array([[1.0, 0.0, 0.0], [64.0, 89.0, 1.0]]), grid)
    assert cells.shape == (2,) and cells[0] == 0 and cells[1] == grid.M - 1
