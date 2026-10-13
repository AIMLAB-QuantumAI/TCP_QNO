"""Launch the ns-3 programs under ``ns3/`` and parse what they print.

Where ns-3 lives and how it is started are configuration, not code: ``NS3_DIR``
(or ``--ns3-dir``) names the ns-3.42 tree whose ``scratch/`` holds the three
programs, and the mode is ``native`` (run ``bash`` on this machine) or ``wsl``
(run inside a WSL distribution, optionally as the user who owns the build).
The three run styles the experiments need are kept apart: the build wrapper
for the open-loop scenarios; the optimised binary called directly for the
closed loop, because the ``./ns3`` wrapper does not forward stdin and the
control channel needs it; and the same binary with stdin closed for the
state probes.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import os
import subprocess
from dataclasses import dataclass

import numpy as np
import torch

MODES = ("native", "wsl")
# Flags every scenario shares, in the order the programs document them.
CONDITION_FLAGS = (
    ("capacityPps", "capacity_pps"), ("rttPropS", "rtt_prop_s"),
    ("nFlows", "n_flows"), ("pErr", "p_err"), ("pMax", "p_max"),
    ("minTh", "min_th"), ("maxThFrac", "max_th_frac"),
    ("bufferPkts", "buffer_pkts"), ("qw", "w_q"),
)


def condition_flags(condition: dict) -> str:
    return " ".join(f"--{flag}={condition[key]}" for flag, key in CONDITION_FLAGS)


def add_ns3_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--ns3-dir", default=os.environ.get("NS3_DIR"),
        help="ns-3.42 tree with the qtro programs built in scratch/ (default: $NS3_DIR)")
    parser.add_argument(
        "--ns3-mode", default=os.environ.get("NS3_MODE", "native"), choices=MODES,
        help="run ns-3 on this machine or inside WSL (default: $NS3_MODE or native)")
    parser.add_argument(
        "--ns3-distro", default=os.environ.get("NS3_DISTRO", "Ubuntu-24.04"),
        help="WSL distribution (wsl mode; default: $NS3_DISTRO or Ubuntu-24.04)")
    parser.add_argument(
        "--ns3-user", default=os.environ.get("NS3_USER"),
        help="user who owns the ns-3 build inside WSL, entered with su (default: $NS3_USER)")


@dataclass(frozen=True)
class Ns3Runner:
    ns3_dir: str
    mode: str = "native"
    distro: str = "Ubuntu-24.04"
    user: str | None = None
    version: str = "3.42"

    @classmethod
    def from_args(cls, args) -> "Ns3Runner":
        if not args.ns3_dir:
            raise SystemExit("ns-3 location unknown: set NS3_DIR or pass --ns3-dir")
        return cls(args.ns3_dir, args.ns3_mode, args.ns3_distro, args.ns3_user)

    def wrapper_command(self, program: str, flags: str) -> str:
        """Through the build wrapper, as the open-loop datasets are generated."""
        return f"cd {self.ns3_dir} && ./ns3 run '{program} {flags}' --no-build"

    def binary_command(self, program: str, flags: str, close_stdin: bool = False) -> str:
        """The optimised binary directly, as the closed loop needs."""
        binary = f"{self.ns3_dir}/build/scratch/ns{self.version}-{program}-optimized"
        command = f"LD_LIBRARY_PATH={self.ns3_dir}/build/lib {binary} {flags}"
        return command + " < /dev/null" if close_stdin else command

    def _argv(self, inner: str, quote: str) -> list[str]:
        if self.mode == "native":
            return ["bash", "-lc", inner]
        if self.user:
            # ns-3 writes its own progress to stderr; the data is on stdout.
            inner = f"su {self.user} -c {quote}{inner}{quote} 2>/dev/null"
        return ["wsl", "-d", self.distro, "--user", "root", "--", "bash", "-lc", inner]

    def wrapper_argv(self, program: str, flags: str) -> list[str]:
        return self._argv(self.wrapper_command(program, flags), quote='"')

    def binary_argv(self, program: str, flags: str, close_stdin: bool = False) -> list[str]:
        return self._argv(self.binary_command(program, flags, close_stdin), quote="'")

    def run(self, argv: list[str], timeout_s: int) -> str | None:
        """Captured stdout, or None when the run timed out."""
        try:
            result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return None
        return result.stdout

    def popen(self, argv: list[str]) -> subprocess.Popen:
        """A line-buffered two-way channel for the closed loop."""
        return subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                text=True, bufsize=1)


def csv_rows(stdout: str, prefix: str | None = None) -> list[list[str]]:
    """Comma-separated rows: data rows start with a digit, protocol rows with ``prefix``."""
    lines = stdout.splitlines()
    if prefix is None:
        return [line.split(",") for line in lines if line and line[0].isdigit()]
    return [line.split(",") for line in lines if line.startswith(prefix)]


def digitise(trajectory: np.ndarray, grid) -> np.ndarray:
    """Rows of (cwnd, queue, phase, ...) to cell indices on ``grid``."""
    w = torch.tensor(trajectory[:, 0], dtype=torch.float32)
    q = torch.tensor(trajectory[:, 1], dtype=torch.float32)
    phase = torch.tensor(trajectory[:, 2], dtype=torch.float32)
    return grid.digitize(w, q, phase).numpy()


def run_closedloop_silent(runner: Ns3Runner, condition: dict, run_seed: int,
                          samples: int, warmup: float, timeout_s: int) -> np.ndarray | None:
    """One ``qtro-closedloop`` run with no controller attached.

    Returns ``(samples + 1, 4)`` rows of cwnd, queue, phase and RED's averaged
    queue, or None when the run failed or timed out.
    """
    flags = (f"{condition_flags(condition)} --samples={samples} --warmup={warmup} "
             f"--runSeed={run_seed} --controlIntervalS=1000 "
             f"--surgeFlows={condition.get('surge_flows', 0)} "
             f"--surgeStartS={condition.get('surge_start_s', -1.0)}")
    stdout = runner.run(runner.binary_argv("qtro-closedloop", flags, close_stdin=True),
                        timeout_s)
    if stdout is None:
        return None
    rows = csv_rows(stdout, "S,")
    if len(rows) < samples + 1:
        return None
    return np.array([[float(r[2]), float(r[3]), float(r[4]), float(r[7])]
                     for r in rows[:samples + 1]], dtype=np.float64)


def gather_closedloop_silent(runner: Ns3Runner, condition: dict, seeds, samples: int,
                             warmup: float, workers: int, timeout_s: int) -> list[np.ndarray]:
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(run_closedloop_silent, runner, condition, s, samples,
                               warmup, timeout_s) for s in seeds]
        out = [f.result() for f in futures]
    return [o for o in out if o is not None]
