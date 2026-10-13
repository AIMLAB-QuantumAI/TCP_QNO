"""Is there a router observable the operator's state is missing after a load step?

The operator is fitted to stationary ensembles at fixed (u, xi), and the
load-step and closed-loop experiments ask it about the stretch just after
something changed. If that is what limits it, there has to be a state
variable that distinguishes "in equilibrium at this (u, xi)" from "heading
there". RED's own averaged queue is the first candidate: it lags the
instantaneous queue by seconds and is what the drop decision is computed
from. This probe tests that and the other observables the router has, in two
parts, without building anything on top of them:

  spread     At fixed (u, xi) and fixed source cell, does the one-step
             destination distribution depend on the candidate variable? If
             not, it is not a missing state variable.

  transfer   Take the kernel measured during a load-step transient. Is it
             closer to a stationary kernel measured at the *same* value of
             the candidate, or to the stationary kernel at the same (u, xi)?
             This is the decisive test, because it is exactly the
             substitution an operator conditioned on that variable would
             make.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from qtro.config import load_config
from qtro.grid import Grid
from qtro.ns3 import Ns3Runner, add_ns3_arguments, digitise, gather_closedloop_silent


def kernel(cells, mask, M):
    """Column-normalised one-step kernel over the transitions mask selects."""
    counts = np.zeros((M, M), dtype=np.int64)
    src = cells[:, :-1][mask[:, :-1]]
    dst = cells[:, 1:][mask[:, :-1]]
    np.add.at(counts, (dst, src), 1)
    total = counts.sum(axis=0)
    return counts / np.maximum(total, 1), total


def column_tv(a, b, counts_a, counts_b, c_min):
    """Mean total variation over columns both kernels actually observed."""
    live = (counts_a >= c_min) & (counts_b >= c_min)
    if not live.any():
        return np.nan, 0
    tv = 0.5 * np.abs(a[:, live] - b[:, live]).sum(axis=0)
    return float(tv.mean()), int(live.sum())


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
    ap.add_argument("--c-min", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--out", required=True)
    add_ns3_arguments(ap)
    args = ap.parse_args()

    runner = Ns3Runner.from_args(args)
    cfg = load_config(args.config, ["sim.device=cpu"])
    grid = Grid(cfg.grid, cfg.net, device="cpu")
    M = grid.M
    base = {"capacity_pps": args.capacity, "rtt_prop_s": args.rtt,
            "n_flows": args.flows, "p_err": 0.0, "p_max": args.p_max,
            "min_th": args.min_th, "max_th_frac": 0.5,
            "buffer_pkts": float(cfg.grid.buffer_pkts), "w_q": 1e-4}
    seeds = list(range(1, args.replications + 1))

    # ---- stationary run at the post-step load ----------------------------
    print(f"stationary at {args.flows} flows ...", flush=True)
    stat = gather_closedloop_silent(runner, base, seeds, args.samples, args.warmup,
                                    args.workers, args.timeout)
    if not stat:
        raise SystemExit("no stationary runs returned")
    cells = np.stack([digitise(b, grid) for b in stat])
    qavg = np.stack([b[:, 3] for b in stat])
    print(f"  {len(stat)} runs, qAvg range [{qavg.min():.1f}, {qavg.max():.1f}], "
          f"mean {qavg.mean():.1f}", flush=True)

    # ---- candidate state variables ---------------------------------------
    # qAvg barely moves during a step, so the probe tests every observable
    # the router has that could separate "settled at this load" from "still
    # responding to it".
    def features(body_list):
        out = {}
        q = np.stack([b[:, 1] for b in body_list])
        a = np.stack([b[:, 3] for b in body_list])
        out["qavg"] = a
        # How far the instantaneous queue sits above its own average. Zero on
        # average in equilibrium, large and positive while a step is building.
        out["q_minus_qavg"] = q - a
        # Recent trend, over one RTT at the calibrated point.
        trend = np.zeros_like(q)
        trend[:, 3:] = q[:, 3:] - q[:, :-3]
        out["dq_3"] = trend
        # The tagged flow's own window relative to the pipe it should occupy.
        w = np.stack([b[:, 0] for b in body_list])
        out["cwnd"] = w
        # Control. A queue bin is several packets wide, so q - qAvg partly
        # reveals where inside its bin q sits, and that is information about
        # queue resolution rather than about RED's averaged state. If this
        # control separates the kernel as strongly, the fix is finer bins and
        # not an extra state variable.
        centres = 0.5 * (grid.q_edges[:-1] + grid.q_edges[1:]).numpy()
        edges_np = grid.q_edges.numpy()
        idx = np.clip(np.searchsorted(edges_np, q, side="right") - 1,
                      0, centres.size - 1)
        out["q_subbin"] = q - centres[idx]
        return out

    # The transient the two experiments actually see: the same load arriving
    # as a step rather than being there from the start.
    surge = dict(base)
    surge["n_flows"] = args.base_flows
    surge["surge_flows"] = args.flows - args.base_flows
    surge["surge_start_s"] = 0.5
    print(f"transient {args.base_flows} -> {args.flows} flows ...", flush=True)
    tr = gather_closedloop_silent(runner, surge, seeds, args.surge_samples, args.warmup,
                                  args.workers, args.timeout)
    if not tr:
        raise SystemExit("no transient runs returned")
    tr_cells = np.stack([digitise(b, grid) for b in tr])
    print(f"  {len(tr)} runs", flush=True)

    stat_f = features(stat)
    tr_f = features(tr)

    report = {"condition": base, "replications": len(stat), "features": {}}
    print("")
    print(f"{'feature':>16}{'spread TV':>12}{'floor':>9}{'ratio':>8}"
          f"{'to equil.':>12}{'to matched':>12}{'gain':>9}")

    half = len(stat) // 2
    step = int(0.5 / 0.01)

    for name, values in stat_f.items():
        edges = np.quantile(values, [0.0, 1 / 3, 2 / 3, 1.0])
        bins = []
        for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            mask = ((values >= lo) & (values <= hi)) if i == 2 else (
                (values >= lo) & (values < hi))
            k, c = kernel(cells, mask, M)
            bins.append({"lo": float(lo), "hi": float(hi), "kernel": k,
                         "counts": c, "mask": mask})

        mid = bins[1]
        ka, ca = kernel(cells[:half], mid["mask"][:half], M)
        kb, cb = kernel(cells[half:], mid["mask"][half:], M)
        floor, _ = column_tv(ka, kb, ca, cb, args.c_min)
        lo_hi, lh_cols = column_tv(bins[0]["kernel"], bins[2]["kernel"],
                                   bins[0]["counts"], bins[2]["counts"],
                                   args.c_min)

        # The transient kernel, and the two stationary kernels it might match.
        tv = tr_f[name]
        window = np.zeros_like(tv, dtype=bool)
        window[:, step:] = True
        tr_kernel, tr_counts = kernel(tr_cells, window, M)
        tr_mean = float(tv[:, step:].mean())
        nearest = min(bins, key=lambda b: abs(0.5 * (b["lo"] + b["hi"]) - tr_mean))
        d_eq, _ = column_tv(tr_kernel, mid["kernel"], tr_counts,
                            mid["counts"], args.c_min)
        d_ma, _ = column_tv(tr_kernel, nearest["kernel"], tr_counts,
                            nearest["counts"], args.c_min)

        report["features"][name] = {
            "spread_tv": lo_hi, "split_half_floor": floor,
            "spread_over_floor": (lo_hi / floor) if floor else None,
            "transient_mean": tr_mean,
            "distance_to_equilibrium": d_eq,
            "distance_to_matched": d_ma,
            "gain": (d_eq - d_ma) if not (np.isnan(d_eq) or np.isnan(d_ma)) else None,
            "bin_edges": [float(e) for e in edges],
        }
        print(f"{name:>16}{lo_hi:>12.4f}{floor:>9.4f}"
              f"{(lo_hi / floor if floor else float('nan')):>8.2f}"
              f"{d_eq:>12.4f}{d_ma:>12.4f}{(d_eq - d_ma):>9.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
