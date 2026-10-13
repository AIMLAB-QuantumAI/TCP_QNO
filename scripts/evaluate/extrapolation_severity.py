"""How the extrapolation gap grows with the severity of the shift.

The out-of-envelope dataset moves one, two, or three coordinates outside the
range the operator was trained on, and its manifest records which. The
headline number for that dataset is one average over all its conditions,
which hides the thing worth reporting: whether the advantage grows as the
shift gets worse, and whether there is a coordinate on which it does not hold
at all. This feeds the extrapolation-by-severity table.

The same per-condition scores are split two ways -- by how many coordinates
moved, and by which one -- and the mean fidelity per group is reported
together with the paired difference against each baseline. Groups are small
by construction, so the per-coordinate rows are reported with their condition
counts and are meant to be read as descriptive.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from qtro.data import SweepData
from qtro.device import resolve_device
from qtro.eval.paired import crossed_bootstrap
from qtro.eval.panel import parse_model_specs, score_run_directory


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True, help="out-of-envelope dataset directory")
    ap.add_argument("--split", required=True, help="split name inside the dataset")
    ap.add_argument("--models", required=True,
                    help="comma-separated name=run-directory entries")
    ap.add_argument("--control", default="qtro")
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260904)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = resolve_device(args.device)
    directory = Path(args.panel)
    manifest = json.loads((directory / "manifest.json").read_text(
        encoding="utf-8"))
    by_file = {s["file"]: s for s in manifest["shards"]}
    order = [p.name for p in sorted(directory.glob("shard_*.npz"))]
    outside = [by_file[name]["outside"] for name in order]

    h = args.horizon
    panel = SweepData(args.panel, horizons=[h], splits=(args.split,),
                      want_refs=True, require_targets=False)
    if len(panel) != len(order):
        raise SystemExit(f"{len(panel)} conditions but {len(order)} shards")
    print(f"panel {args.panel}: {len(panel)} conditions at h={h}")

    scores = {}
    for name, path in parse_model_specs(args.models).items():
        scored = score_run_directory(path, panel, device, (h,), ("fidelity",))
        scores[name] = {s: scored[s][h]["fidelity"]["occupancy_weighted"] for s in scored}
        print(f"  {name}: {len(scores[name])} seeds", flush=True)
    control = scores[args.control]

    groups: dict[str, list[int]] = defaultdict(list)
    for i, names in enumerate(outside):
        groups[f"{len(names)}_factor" if len(names) == 1
               else "multi_factor"].append(i)
        for coordinate in names:
            groups["coord:" + coordinate].append(i)

    rng = np.random.default_rng(args.seed)
    report = {"panel": args.panel, "horizon": h, "groups": {}}
    print(f"\n{'group':>22}{'n':>5}" +
          "".join(f"{k:>12}" for k in scores) + "   paired vs control")
    for label in sorted(groups, key=lambda k: (k.startswith("coord:"), k)):
        idx = np.array(groups[label])
        row = {"n_conditions": int(idx.size), "mean": {}, "paired": {}}
        line = f"{label:>22}{idx.size:>5}"
        for name, seeds in scores.items():
            value = float(np.mean([seeds[s][idx].mean() for s in seeds]))
            row["mean"][name] = value
            line += f"{value:>12.4f}"
        detail = []
        for name, seeds in scores.items():
            if name == args.control:
                continue
            shared = sorted(set(seeds) & set(control))
            delta = np.stack([seeds[s][idx] - control[s][idx]
                              for s in shared])
            stat = crossed_bootstrap(delta, n_resamples=args.resamples, rng=rng)
            low, high = stat["mean_ci"]
            row["paired"][name] = {
                "advantage": float(stat["mean_advantage"]),
                "ci_low": float(low), "ci_high": float(high),
                "significant": bool(low > 0 or high < 0)}
            mark = "*" if row["paired"][name]["significant"] else " "
            detail.append(f"{name} {stat['mean_advantage']:+.4f}{mark}")
        report["groups"][label] = row
        print(line + "   " + ", ".join(detail))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=1),
                                  encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
