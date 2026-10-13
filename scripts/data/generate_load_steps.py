"""Measure the queue's response to a load step in ns-3.

The AQM literature judges a controller by its step response rather than by
its stationary queue, so this driver runs the dumbbell with extra sessions
switched on mid-run, collects an ensemble of independent replications, and
reports the response the way that work does: peak, time to peak, settling
time, and the queue quantiles at every instant rather than only a mean.
Session counts stay inside the training range {2, 4, 6, 8} on purpose: a
larger step would be a more dramatic figure and would also make every model
comparison an extrapolation, which is a different experiment.

Arrivals only, by default. An arriving session is in slow start and overshoots
whatever its eventual share is, so a load increase saturates this 90-packet
buffer at every step size tried; that is the system's behaviour, and the drop
cost and settling time still separate the RED settings. The scenario can also
release sessions and the driver will summarise that direction, but it is left
out of the default set. Stopping a sender closes its socket gracefully, so TCP
keeps delivering whatever is already buffered; at the send-buffer size the
stationary datasets were generated with, the released sessions never actually
leave. Shrinking the buffer does make them leave, but then the drain is a
measurement of the socket buffer rather than of TCP. See the comment in
ns3/qtro-dumbbell.cc.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path

import numpy as np
import torch

from qtro.config import load_config
from qtro.grid import Grid
from qtro.ns3 import Ns3Runner, add_ns3_arguments, condition_flags, csv_rows, extra_flags


def run_one(runner: Ns3Runner, condition: dict, run_seed: int,
            timeout_s: int) -> np.ndarray | None:
    """Return ``(samples + 1, 5)`` of cwnd, queue, phase, n_active, drops."""
    flags = condition_flags(condition) + extra_flags({
        "surgeFlows": condition.get("surge_flows", 0),
        "surgeStartS": condition.get("surge_start_s", -1.0),
        "surgeStopS": condition.get("surge_stop_s", -1.0),
        "releaseFlows": condition.get("release_flows", 0),
        "releaseStopS": condition.get("release_stop_s", -1.0),
        "releaseRestartS": condition.get("release_restart_s", -1.0),
        "samples": condition["samples"],
        "sampleInterval": condition["sample_interval_s"],
        "warmup": condition["warmup_s"],
        "runSeed": run_seed,
    })
    stdout = runner.run(runner.wrapper_argv("qtro-dumbbell", flags), timeout_s)
    if stdout is None:
        return None
    rows = csv_rows(stdout)
    if len(rows) < condition["samples"] + 1:
        return None
    return np.array([[float(r[1]), float(r[2]), float(r[3]), float(r[4]),
                      float(r[5])] for r in rows[:condition["samples"] + 1]],
                    dtype=np.float64)


def settling_index(mean_queue: np.ndarray, step: int, end: int,
                   band: float) -> int:
    """First index after the step from which the mean stays inside the band.

    The band is measured around the post-step level, taken as the mean over
    the last quarter of the interval the load change lasts. Returning the
    *first* index from which the trace never leaves again -- rather than the
    first entry -- is what makes this a settling time and not a first-crossing
    time.

    Returns -1 when the trace never settles. The final index trivially
    satisfies "never leaves again", so it is excluded: reporting it would turn
    "did not settle" into a settling time equal to the window length, which is
    the same number for every unsettled condition and reads like a
    measurement.
    """
    window = mean_queue[step:end]
    if window.size < 4:
        return -1
    level = float(window[3 * window.size // 4:].mean())
    inside = np.abs(mean_queue - level) <= band
    for index in range(step, end - 1):
        if inside[index:end].all():
            return index
    return -1


def summarise(cells: np.ndarray, condition: dict, band_fraction: float,
              rng: np.random.Generator, n_resamples: int) -> dict:
    """Ensemble statistics for one operating point.

    The extremum follows the step direction: a load increase is summarised by
    the highest mean queue it reaches and a release by the lowest, because the
    same statistic under the other direction would only report the pre-step
    level back.
    """
    queue = cells[:, :, 1]
    drops = cells[:, :, 4]
    dt = condition["sample_interval_s"]
    increase = condition.get("surge_flows", 0) > 0
    start_s = (condition["surge_start_s"] if increase
               else condition["release_stop_s"])
    end_s = (condition.get("surge_stop_s", -1.0) if increase
             else condition.get("release_restart_s", -1.0))
    step = int(round(start_s / dt))
    end = int(round(end_s / dt)) if end_s >= 0 else queue.shape[1]

    mean_queue = queue.mean(axis=0)
    boot = rng.integers(0, queue.shape[0], size=(n_resamples, queue.shape[0]))
    resampled = queue[boot].mean(axis=1)
    low, high = np.percentile(resampled, [2.5, 97.5], axis=0)

    pre = float(mean_queue[max(step - 20, 0):step].mean())
    inner = mean_queue[step:end]
    peak_index = int((np.argmax(inner) if increase else np.argmin(inner))
                     + step)
    peak = float(mean_queue[peak_index])
    band = band_fraction * abs(peak - pre)
    settle = settling_index(mean_queue, step, end, band)
    ceiling = float(condition["buffer_pkts"]) - 1.0

    # Drops are cumulative, so the surge cost is a difference. Per-replication
    # so the interval reflects run-to-run spread rather than one trace.
    surge_drops = drops[:, min(end, drops.shape[1] - 1)] - drops[:, step]
    return {
        "n_replications": int(cells.shape[0]),
        "step_index": step, "step_time_s": step * dt,
        "direction": "increase" if increase else "release",
        "pre_step_mean_queue": pre,
        "extreme_mean_queue": peak,
        "extreme_time_s": (peak_index - step) * dt,
        "excursion_packets": peak - pre,
        # A response that reaches the buffer is censored, and its extremum is
        # then a property of the buffer rather than of the dynamics.
        "buffer_saturated": bool((queue[:, step:end] >= ceiling).any()),
        "saturated_sample_fraction": float(
            (queue[:, step:end] >= ceiling).mean()),
        "settling_time_s": (settle - step) * dt if settle >= 0 else None,
        "settling_band_packets": band,
        "surge_drops_mean": float(surge_drops.mean()),
        "surge_drops_sd": float(surge_drops.std(ddof=1)),
        "mean_queue": mean_queue.tolist(),
        "mean_queue_ci_low": low.tolist(),
        "mean_queue_ci_high": high.tolist(),
        # The distribution, not only its mean: this is the quantity the
        # operator predicts, and a mean trace cannot show it.
        "queue_quantiles": {
            str(q): np.percentile(queue, q, axis=0).tolist()
            for q in (5, 25, 50, 75, 95)},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="directory for results.json and the cells")
    ap.add_argument("--replications", type=int, default=64)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--timeout-s", type=int, default=1800)
    ap.add_argument("--band-fraction", type=float, default=0.1)
    ap.add_argument("--resamples", type=int, default=4000)
    ap.add_argument("--bootstrap-seed", type=int, default=20260826)
    ap.add_argument("--conditions", default="",
                    help="JSON list of step conditions (default: the six built in)")
    ap.add_argument("--config", default=None,
                    help="YAML config (default: the built-in values)")
    add_ns3_arguments(ap)
    args = ap.parse_args()

    runner = Ns3Runner.from_args(args)
    # The digitised ensemble is what a model can be scored against; the
    # summary statistics below cannot be turned back into a distribution.
    cfg = load_config(args.config, ["sim.device=cpu"])
    grid = Grid(cfg.grid, cfg.net, device="cpu")

    if args.conditions:
        conditions = json.loads(
            Path(args.conditions).read_text(encoding="utf-8"))
    else:
        base = {"capacity_pps": 2000.0, "rtt_prop_s": 0.030, "p_err": 0.0,
                "max_th_frac": 0.5, "buffer_pkts": 90.0, "w_q": 1e-4,
                "warmup_s": 8.0, "sample_interval_s": 0.010, "samples": 300}
        arrive = {"surge_start_s": 0.5, "surge_stop_s": 1.5}
        # Three RED settings rather than two: the control setting moves
        # accuracy far more than the model choice does, so a transient
        # reported at one setting would say very little.
        red = (("mild", 0.05, 16.0), ("mid", 0.1, 20.0), ("firm", 0.2, 30.0))
        conditions = []
        for label, p_max, min_th in red:
            control = {"p_max": p_max, "min_th": min_th}
            # 2 -> 8 and 4 -> 8, both ends inside the training range
            # {2, 4, 6, 8}. Two step sizes separate the size of the step from
            # the RED setting.
            conditions.append({**base, **arrive, **control,
                               "name": f"arrive_2_to_8_red_{label}",
                               "n_flows": 2, "surge_flows": 6})
            conditions.append({**base, **arrive, **control,
                               "name": f"arrive_4_to_8_red_{label}",
                               "n_flows": 4, "surge_flows": 4})

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.bootstrap_seed)
    report = {"simulator": "ns-3.42", "scenario": "qtro-dumbbell.cc",
              "replications": args.replications,
              "band_fraction": args.band_fraction, "conditions": []}

    for condition in conditions:
        runs: list[np.ndarray] = []
        with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
            futures = [pool.submit(run_one, runner, condition, seed + 1,
                                   args.timeout_s)
                       for seed in range(args.replications)]
            for future in concurrent.futures.as_completed(futures):
                value = future.result()
                if value is not None:
                    runs.append(value)
        if len(runs) < 8:
            print(json.dumps({"event": "condition_failed",
                              "name": condition["name"],
                              "runs": len(runs)}), flush=True)
            continue
        stacked = np.stack(runs)
        summary = summarise(stacked, condition, args.band_fraction,
                            rng, args.resamples)
        cells = grid.digitize(
            torch.tensor(stacked[:, :, 0], dtype=torch.float32),
            torch.tensor(stacked[:, :, 1], dtype=torch.float32),
            torch.tensor(stacked[:, :, 2], dtype=torch.float32)).numpy()
        np.savez_compressed(
            out / f"cells_{condition['name']}.npz",
            cells=cells.astype(np.int16),
            n_active=stacked[:, :, 3].astype(np.int16),
            raw=stacked.astype(np.float32))
        summary["cells_file"] = f"cells_{condition['name']}.npz"
        report["conditions"].append({**condition, **summary})
        (out / "results.json").write_text(
            json.dumps(report, indent=1), encoding="utf-8")
        print(json.dumps({
            "event": "condition_done", "name": condition["name"],
            "direction": summary["direction"],
            "runs": summary["n_replications"],
            "pre": round(summary["pre_step_mean_queue"], 2),
            "extreme": round(summary["extreme_mean_queue"], 2),
            "extreme_time_s": summary["extreme_time_s"],
            "settling_time_s": summary["settling_time_s"],
            "saturated": summary["buffer_saturated"],
            "step_drops": round(summary["surge_drops_mean"], 1)}), flush=True)

    if not report["conditions"]:
        raise SystemExit("no ns-3 run returned data: check NS3_DIR, NS3_MODE and "
                         "NS3_USER, and that the qtro programs are built (ns3/README.md)")
    print(json.dumps({"event": "complete",
                      "conditions": len(report["conditions"])}))


if __name__ == "__main__":
    main()
