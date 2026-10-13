"""Would an extended state fix the load-step under-prediction?

The state probe showed that the instantaneous queue's departure from RED's
average carries transient information the base state does not, and that queue
resolution does not account for it. That was measured on one-step kernels. It
does not follow that a model built on the extended state predicts the step
better, because a rollout compounds and because splitting the state halves the
transitions behind every column.

This tests the fix before paying for it, and without training anything. Two
empirical kernels are estimated from the *same* stationary runs at the
post-step load:

  base      the 512-cell state;
  delta2    (queue above its own average) * 512 + base cell.

Both are then rolled forward from the measured distribution at the step and
scored against the measured transient. An empirical kernel is the best any
model of that representation could fit, so if delta2 does not beat base here,
no amount of training on the extended state will help.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from qtro.config import load_config
from qtro.grid import Grid
from qtro.ns3 import Ns3Runner, add_ns3_arguments, digitise, gather_closedloop_silent


def labelled(body_list, grid, extended: bool):
    """Digitised trajectories, optionally with the delta bit prepended."""
    cells = np.stack([digitise(b, grid) for b in body_list])
    if not extended:
        return cells
    q = np.stack([b[:, 1] for b in body_list])
    a = np.stack([b[:, 3] for b in body_list])
    bit = (q > a).astype(np.int64)
    return bit * grid.M + cells


def kernel(cells, M, start=0):
    counts = np.zeros((M, M), dtype=np.int64)
    src = cells[:, start:-1].reshape(-1)
    dst = cells[:, start + 1:].reshape(-1)
    np.add.at(counts, (dst, src), 1)
    total = counts.sum(axis=0)
    return counts / np.maximum(total, 1), total


def occupancy(cells_column, M):
    counts = np.bincount(cells_column, minlength=M).astype(np.float64)
    return counts / max(counts.sum(), 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None,
                    help="YAML config (default: the built-in values)")
    ap.add_argument("--capacity", type=float, default=2000.0)
    ap.add_argument("--rtt", type=float, default=0.030)
    ap.add_argument("--flows", type=int, default=8)
    ap.add_argument("--base-flows", type=int, default=2,
                    help="load before the step; the load-step set uses both "
                         "2 -> 8 and 4 -> 8")
    ap.add_argument("--p-max", type=float, default=0.1)
    ap.add_argument("--min-th", type=float, default=20.0)
    ap.add_argument("--replications", type=int, default=64)
    ap.add_argument("--samples", type=int, default=1200)
    ap.add_argument("--warmup", type=float, default=12.0)
    ap.add_argument("--surge-samples", type=int, default=200)
    ap.add_argument("--horizon", type=int, default=100)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--out", required=True)
    add_ns3_arguments(ap)
    args = ap.parse_args()

    runner = Ns3Runner.from_args(args)
    cfg = load_config(args.config, ["sim.device=cpu"])
    grid = Grid(cfg.grid, cfg.net, device="cpu")
    Mb = grid.M
    q_base = grid.q_rep.numpy().astype(np.float64)
    tail_base = grid.tail_mask.numpy().astype(np.float64)

    base_cond = {"capacity_pps": args.capacity, "rtt_prop_s": args.rtt,
                 "n_flows": args.flows, "p_err": 0.0, "p_max": args.p_max,
                 "min_th": args.min_th, "max_th_frac": 0.5,
                 "buffer_pkts": float(cfg.grid.buffer_pkts), "w_q": 1e-4}
    seeds = list(range(1, args.replications + 1))

    print(f"stationary at {args.flows} flows ...", flush=True)
    stat = gather_closedloop_silent(runner, base_cond, seeds, args.samples,
                                    args.warmup, args.workers, args.timeout)
    surge = dict(base_cond)
    surge["n_flows"] = args.base_flows
    surge["surge_flows"] = args.flows - args.base_flows
    surge["surge_start_s"] = 0.5
    print(f"transient {args.base_flows} -> {args.flows} flows ...", flush=True)
    tr = gather_closedloop_silent(runner, surge, seeds, args.surge_samples,
                                  args.warmup, args.workers, args.timeout)
    if not stat or not tr:
        raise SystemExit("ns-3 returned nothing")
    print(f"  {len(stat)} stationary, {len(tr)} transient runs", flush=True)

    step = int(0.5 / 0.01)
    report = {"condition": base_cond, "horizon": args.horizon,
              "replications": {"stationary": len(stat), "transient": len(tr)},
              "representations": {}}

    for name, extended in (("base", False), ("delta2", True)):
        M = Mb * (2 if extended else 1)
        q = np.tile(q_base, 2) if extended else q_base
        tail = np.tile(tail_base, 2) if extended else tail_base

        stat_cells = labelled(stat, grid, extended)
        tr_cells = labelled(tr, grid, extended)
        P, counts = kernel(stat_cells, M)
        usable = int((counts >= 32).sum())

        # Start from the measured distribution at the step and roll forward.
        p = occupancy(tr_cells[:, step], M)
        pred_mean, pred_tail = [], []
        for _ in range(args.horizon):
            p = P @ p
            s = p.sum()
            if s > 0:
                p = p / s
            pred_mean.append(float(p @ q))
            pred_tail.append(float(p @ tail))

        ref_mean, ref_tail = [], []
        for k in range(1, args.horizon + 1):
            t = step + k
            if t >= tr_cells.shape[1]:
                break
            r = occupancy(tr_cells[:, t], M)
            ref_mean.append(float(r @ q))
            ref_tail.append(float(r @ tail))

        n = min(len(pred_mean), len(ref_mean))
        mean_err = np.abs(np.array(pred_mean[:n]) - np.array(ref_mean[:n]))
        tail_err = np.abs(np.array(pred_tail[:n]) - np.array(ref_tail[:n]))
        peak = int(np.argmax(ref_mean[:n]))
        report["representations"][name] = {
            "states": M, "usable_columns": usable,
            "mean_queue_error": float(mean_err.mean()),
            "mean_queue_error_at_peak": float(mean_err[peak]),
            "tail_error": float(tail_err.mean()),
            "reference_peak": float(np.max(ref_mean[:n])),
            "predicted_at_peak": float(pred_mean[peak]),
            "peak_index": peak,
        }
        r = report["representations"][name]
        print(f"\n{name}: {M} states, {usable} usable columns")
        print(f"  reference peaks at {r['reference_peak']:.1f} packets "
              f"(sample {peak}); prediction there is "
              f"{r['predicted_at_peak']:.1f}")
        print(f"  mean queue error {r['mean_queue_error']:.2f} packets "
              f"(at the peak {r['mean_queue_error_at_peak']:.2f})")
        print(f"  high-queue probability error {r['tail_error']:.4f}")

    # ---- the remaining candidate: fit on transient data instead ----------
    # The kernels above come from stationary runs, so the test so far is
    # "does the extended state rescue an equilibrium fit". The separate
    # question is whether the fit has to see steps at all, and whether the
    # extra state then earns its place. Half the transient replications
    # supply the fit and the other half the reference, so the comparison is
    # out of sample.
    half = len(tr) // 2
    print("")
    print("fitted on transient replications instead of stationary")
    for name, extended in (("base", False), ("delta2", True)):
        M = Mb * (2 if extended else 1)
        q = np.tile(q_base, 2) if extended else q_base
        tail = np.tile(tail_base, 2) if extended else tail_base
        tr_cells = labelled(tr, grid, extended)
        P, counts = kernel(tr_cells[:half], M)
        usable = int((counts >= 32).sum())

        p_vec = occupancy(tr_cells[half:, step], M)
        pred_mean, pred_tail = [], []
        for _ in range(args.horizon):
            p_vec = P @ p_vec
            t = p_vec.sum()
            if t > 0:
                p_vec = p_vec / t
            pred_mean.append(float(p_vec @ q))
            pred_tail.append(float(p_vec @ tail))
        ref_mean, ref_tail = [], []
        for k in range(1, args.horizon + 1):
            t = step + k
            if t >= tr_cells.shape[1]:
                break
            r = occupancy(tr_cells[half:, t], M)
            ref_mean.append(float(r @ q))
            ref_tail.append(float(r @ tail))
        n = min(len(pred_mean), len(ref_mean))
        mean_err = np.abs(np.array(pred_mean[:n]) - np.array(ref_mean[:n]))
        tail_err = np.abs(np.array(pred_tail[:n]) - np.array(ref_tail[:n]))
        peak = int(np.argmax(ref_mean[:n]))
        report["representations"][name + "_transient_fit"] = {
            "states": M, "usable_columns": usable,
            "mean_queue_error": float(mean_err.mean()),
            "mean_queue_error_at_peak": float(mean_err[peak]),
            "tail_error": float(tail_err.mean()),
            "reference_peak": float(np.max(ref_mean[:n])),
            "predicted_at_peak": float(pred_mean[peak]),
        }
        r = report["representations"][name + "_transient_fit"]
        print(f"  {name:>7}: {usable:>3} usable columns | "
              f"peak ref {r['reference_peak']:5.1f} pred "
              f"{r['predicted_at_peak']:5.1f} | mean queue error "
              f"{r['mean_queue_error']:5.2f} (peak "
              f"{r['mean_queue_error_at_peak']:5.2f}) | tail "
              f"{r['tail_error']:.4f}")

    b = report["representations"]["base"]
    d = report["representations"]["delta2"]
    gain = b["mean_queue_error"] - d["mean_queue_error"]
    print(f"\ndelta2 {'reduces' if gain > 0 else 'increases'} the mean queue "
          f"error by {abs(gain):.2f} packets "
          f"({100 * gain / b['mean_queue_error']:+.1f}%)")
    print(f"at the peak: {b['mean_queue_error_at_peak']:.1f} -> "
          f"{d['mean_queue_error_at_peak']:.1f} packets")
    report["gain_mean_queue_error"] = float(gain)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
