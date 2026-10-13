"""Per-condition statistics of a dataset: transitions, column coverage, visitation share.

The dataset description in the paper quotes, for the held-out split of the main
dataset, the observed one-step transitions per condition, the median sample count
of a visited source column, how many columns per condition clear the grading
threshold, and the share of visitation those columns carry.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True, help="dataset directory of shard_*.npz files")
    ap.add_argument("--split", default="ns3_heldout")
    ap.add_argument("--horizons", default="1,5,10,20")
    ap.add_argument("--c-min", type=int, default=32)
    ap.add_argument("--out", default=None, help="write the report as JSON here too")
    args = ap.parse_args()
    horizons = [int(h) for h in args.horizons.split(",") if h.strip()]

    transitions, medians = [], []
    columns = {h: [] for h in horizons}
    share = {h: [] for h in horizons}
    for path in sorted(glob.glob(os.path.join(args.panel, "shard_*.npz"))):
        z = np.load(path)
        for c in range(z["split"].shape[0]):
            if str(z["split"][c]) != args.split:
                continue
            occupancy = np.asarray(z["count_h1"][c], dtype=np.float64)
            total = occupancy.sum()
            if total <= 0:
                continue
            transitions.append(total)
            medians.append(float(np.median(occupancy[occupancy > 0])))
            for h in horizons:
                valid = np.asarray(z[f"count_ref_h{h}"][c]) >= args.c_min
                columns[h].append(int(valid.sum()))
                share[h].append(float(occupancy[valid].sum() / total))

    report = {
        "panel": args.panel, "split": args.split, "conditions": len(transitions),
        "c_min": args.c_min,
        "fitting_half_transitions_per_condition": float(np.mean(transitions)),
        "median_count_of_a_visited_column_mean_over_conditions": float(np.mean(medians)),
        "per_horizon": {
            str(h): {"graded_columns_mean": float(np.mean(columns[h])),
                     "visitation_share_mean": float(np.mean(share[h])),
                     "visitation_share_worst_condition": float(np.min(share[h]))}
            for h in horizons},
    }
    print(json.dumps(report, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)


if __name__ == "__main__":
    main()
