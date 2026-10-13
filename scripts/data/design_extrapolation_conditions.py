"""Design the out-of-envelope condition list.

The held-out split of the main dataset varies plant points inside each
control cell, so it measures interpolation. This list puts coordinates
outside the range the operator was trained on, in two blocks:

``one_factor``
    exactly one coordinate outside its training range, the rest at values the
    training set contains. This says which axis generalisation breaks on.
``joint``
    two or three coordinates outside at once. This says whether the failures
    compound.

No retraining is involved. The dataset is scored with the checkpoints trained
on the main dataset, so it is an evaluation set in the strict sense and
carries a split name the training script will not load.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# The training envelope of configs/conditions/main.json.
IN = {
    "capacity_pps": (1000.0, 1500.0, 2000.0, 3000.0, 4000.0),
    "rtt_prop_s": (0.020, 0.030, 0.040, 0.050),
    "n_flows": (2, 4, 6, 8),
    "p_err": (0.0, 0.001),
    "p_max": (0.05, 0.1, 0.2, 0.3),
    "min_th": (16.0, 20.0, 26.0, 30.0, 34.0),
}
# Outside it. min_th stays below max_th = 0.5 * 90 = 45 so RED keeps a gentle
# region; a threshold above it would test the AQM configuration rather than
# the operator.
OUT = {
    "capacity_pps": (500.0, 6000.0),
    "rtt_prop_s": (0.012, 0.070),
    "n_flows": (1, 12),
    "p_err": (0.003,),
    "p_max": (0.02, 0.45),
    "min_th": (10.0, 40.0),
}
CENTRE = {"capacity_pps": 2000.0, "rtt_prop_s": 0.030, "n_flows": 4,
          "p_err": 0.0, "p_max": 0.1, "min_th": 20.0}
FIXED = {"max_th_frac": 0.5, "buffer_pkts": 90.0, "w_q": 1e-4}
TAIL_MIN_TH = 26.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-cell", type=int, default=4,
                    help="in-range plant points per out-of-range value")
    ap.add_argument("--joint", type=int, default=24)
    ap.add_argument("--replications", type=int, default=64)
    ap.add_argument("--tail-replications", type=int, default=128)
    ap.add_argument("--seed", type=int, default=20260827)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    conditions = []

    def emit(condition: dict, block: str) -> None:
        outside = sorted(k for k, v in condition.items()
                         if k in IN and v not in IN[k])
        conditions.append({
            **condition, **FIXED, "split": "ns3_extrap", "block": block,
            "outside": outside,
            "replications": (args.tail_replications
                             if condition["min_th"] >= TAIL_MIN_TH
                             else args.replications)})

    # One factor at a time. The other coordinates are drawn from the training
    # values rather than pinned at the centre, so the block does not measure
    # one operating point repeatedly.
    others = [k for k in IN]
    for axis, values in OUT.items():
        for value in values:
            for _ in range(args.per_cell):
                condition = {k: (value if k == axis
                                 else IN[k][rng.integers(len(IN[k]))])
                             for k in others}
                condition["n_flows"] = int(condition["n_flows"])
                emit(condition, "one_factor")

    # Two or three axes outside at once.
    axes = list(OUT)
    for _ in range(args.joint):
        k = int(rng.integers(2, 4))
        chosen = rng.choice(len(axes), size=k, replace=False)
        condition = {a: IN[a][rng.integers(len(IN[a]))] for a in others}
        for index in chosen:
            axis = axes[index]
            condition[axis] = OUT[axis][rng.integers(len(OUT[axis]))]
        condition["n_flows"] = int(condition["n_flows"])
        emit(condition, "joint")

    # Distinct shard names are what stops two conditions collapsing onto one
    # file, which would silently lose conditions from the dataset.
    def name(c: dict) -> tuple:
        return (c["capacity_pps"], c["rtt_prop_s"], c["n_flows"], c["p_err"],
                c["p_max"], c["min_th"])
    seen, unique = set(), []
    for c in conditions:
        if name(c) in seen:
            continue
        seen.add(name(c))
        unique.append(c)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(unique, indent=1), encoding="utf-8")
    runs = sum(c["replications"] for c in unique)
    print(json.dumps({
        "file": str(out), "conditions": len(unique),
        "dropped_duplicates": len(conditions) - len(unique),
        "one_factor": sum(1 for c in unique if c["block"] == "one_factor"),
        "joint": sum(1 for c in unique if c["block"] == "joint"),
        "total_ns3_runs": runs,
        "estimated_hours_at_1.9_runs_per_s": round(runs / 1.9 / 3600, 1)},
        indent=1))


if __name__ == "__main__":
    main()
