"""Design the main dataset's condition list.

The design is a full cross of the two control axes (p_max, min_th) against a
spread of plant points (capacity, RTT, session count, link-error rate), so
the control setting can be separated from the network it acts on. Within
each control cell the held-out plant points are disjoint from the training
plant points, which is what makes the held-out split held out. The settings
that reach the queue tail get more replications, because that is where the
risk claim has to be measured.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

# Plant axes. The loss rate stays small enough to be a link error rather than
# a second congestion signal.
CAPACITY = (1000.0, 1500.0, 2000.0, 3000.0, 4000.0)
RTT = (0.020, 0.030, 0.040, 0.050)
FLOWS = (2, 4, 6, 8)
P_ERR = (0.0, 0.001)

# Control axes, fully crossed. min_th 26 and above is where the tail becomes
# reachable at all.
P_MAX = (0.05, 0.1, 0.2, 0.3)
MIN_TH = (16.0, 20.0, 26.0, 30.0, 34.0)
TAIL_MIN_TH = 26.0

FIXED = {"max_th_frac": 0.5, "buffer_pkts": 90.0, "w_q": 1e-4}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-per-cell", type=int, default=12)
    ap.add_argument("--heldout-per-cell", type=int, default=3)
    ap.add_argument("--replications", type=int, default=64)
    ap.add_argument("--tail-replications", type=int, default=128)
    ap.add_argument("--seed", type=int, default=20260826)
    args = ap.parse_args()

    plant = list(itertools.product(CAPACITY, RTT, FLOWS, P_ERR))
    per_cell = args.train_per_cell + args.heldout_per_cell
    if per_cell > len(plant):
        raise SystemExit(
            f"{per_cell} plant points per control cell requested but the "
            f"plant grid has {len(plant)}")

    rng = np.random.default_rng(args.seed)
    conditions = []
    for p_max, min_th in itertools.product(P_MAX, MIN_TH):
        order = rng.permutation(len(plant))[:per_cell]
        for rank, index in enumerate(order):
            capacity, rtt, flows, p_err = plant[index]
            conditions.append({
                "capacity_pps": capacity, "rtt_prop_s": rtt,
                "n_flows": int(flows), "p_err": p_err,
                "p_max": p_max, "min_th": min_th, **FIXED,
                "split": ("train" if rank < args.train_per_cell
                          else "ns3_heldout"),
                "replications": (args.tail_replications
                                 if min_th >= TAIL_MIN_TH
                                 else args.replications),
            })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(conditions, indent=1), encoding="utf-8")

    train = [c for c in conditions if c["split"] == "train"]
    held = [c for c in conditions if c["split"] != "train"]
    runs = sum(c["replications"] for c in conditions)
    print(json.dumps({
        "file": str(out),
        "control_cells": len(P_MAX) * len(MIN_TH),
        "conditions": len(conditions),
        "train": len(train), "heldout": len(held),
        "tail_conditions": sum(1 for c in conditions
                               if c["min_th"] >= TAIL_MIN_TH),
        "p_err_varied": sorted({c["p_err"] for c in conditions}),
        "total_ns3_runs": runs,
        "estimated_hours_at_1.9_runs_per_s": round(runs / 1.9 / 3600, 1),
    }, indent=1))


if __name__ == "__main__":
    main()
