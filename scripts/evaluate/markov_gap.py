"""The Markov gap of a dataset, pooled the way the accuracy script pools the floor.

The gap between the measured h-step kernel and the h-th power of the measured
one-step kernel, pooled over every graded (condition, column) pair exactly as
the accuracy script pools the split-half floor, so the two are on one scale.
It feeds the sentence that puts the composition error next to the measurement
floor.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np


def identity_filled(K: np.ndarray, count: np.ndarray) -> np.ndarray:
    """The one-step kernel with never-left source cells held in place."""
    K = np.asarray(K, dtype=np.float64).copy()
    dead = np.asarray(count) <= 0
    if dead.any():
        K[:, dead] = 0.0
        idx = np.where(dead)[0]
        K[idx, idx] = 1.0
    s = K.sum(axis=0)
    g = s > 0
    K[:, g] /= s[g]
    return K


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True, help="dataset directory of shard_*.npz files")
    ap.add_argument("--split", default="ns3_heldout")
    ap.add_argument("--horizons", default="1,5,10,20")
    ap.add_argument("--c-min", type=int, default=32)
    ap.add_argument("--out", default=None, help="write the summary as JSON here too")
    args = ap.parse_args()
    horizons = [int(h) for h in args.horizons.split(",") if h.strip()]

    pool_gap = {h: [] for h in horizons}
    pool_noise = {h: [] for h in horizons}
    pool_fid = {h: [] for h in horizons}
    per_cond = {h: [] for h in horizons}
    n_cond = 0

    for path in sorted(glob.glob(os.path.join(args.panel, "shard_*.npz"))):
        z = np.load(path)
        splits = z["split"]
        for c in range(splits.shape[0]):
            if str(splits[c]) != args.split:
                continue
            n_cond += 1
            K1 = identity_filled(z["target_h1"][c], z["count_h1"][c])
            P = np.eye(K1.shape[0], dtype=np.float64)
            step = 0
            for h in horizons:
                while step < h:
                    P = K1 @ P
                    step += 1
                Kh = np.asarray(z[f"target_h{h}"][c], dtype=np.float64)
                s = Kh.sum(axis=0)
                Kn = Kh.copy()
                Kn[:, s > 0] /= s[s > 0]
                valid = np.asarray(z[f"count_ref_h{h}"][c]) >= args.c_min
                if not valid.any():
                    continue
                tv = 0.5 * np.abs(P - Kn).sum(axis=0)
                pool_gap[h].append(tv[valid])
                pool_noise[h].append(np.asarray(z[f"noise_tv_h{h}"][c])[valid])
                pool_fid[h].append(np.asarray(z[f"noise_fidelity_h{h}"][c])[valid])
                per_cond[h].append(float(tv[valid].mean()))

    print(f"{args.split} conditions: {n_cond}   minimum grading count: {args.c_min}")
    print("pooled over every valid (condition, column) pair\n")
    print("=" * 78)
    print("%-4s %10s %12s %7s %10s  %s"
          % ("h", "gap TV", "split-half", "ratio", "n columns", "per-condition range"))
    print("=" * 78)
    summary = {}
    for h in horizons:
        g = np.concatenate(pool_gap[h])
        n = np.concatenate(pool_noise[h])
        f = np.concatenate(pool_fid[h])
        pc = np.array(per_cond[h])
        summary[str(h)] = dict(gap=round(float(g.mean()), 4),
                               split_half_tv=round(float(n.mean()), 4),
                               split_half_fid=round(float(f.mean()), 4),
                               ratio=round(float(g.mean() / n.mean()), 2),
                               n_columns=int(g.size),
                               cond_lo=round(float(pc.min()), 4),
                               cond_hi=round(float(pc.max()), 4))
        print("%-4d %10.4f %12.4f %6.2fx %10d  [%.4f, %.4f]"
              % (h, g.mean(), n.mean(), g.mean() / n.mean(), g.size,
                 pc.min(), pc.max()))
    print("=" * 78)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
