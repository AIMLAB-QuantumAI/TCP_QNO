"""Communication observables and tail power on any ns-3 dataset.

The high-queue claim needs enough tail source columns per condition to be
measured at all, and that limit is about how often the queue actually reaches
the tail, not about any model. Bursty cross traffic changes that: the burst
dataset reaches the buffer ceiling where the long-lived-only dataset peaks
well below it, so the region the risk claim is about is populated. This
measures both things on whichever dataset it is given: how many tail columns
the dataset supplies, and the paired queue Wasserstein-1 and high-queue errors
on them. It feeds the tail-observable table.

The tail aggregation is the one used elsewhere: a column counts as a tail
source when its source cell sits above q_crit and it clears c_min, and a
condition with no such column contributes NaN rather than a zero.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from qtro.checkpoint import load_checkpoint
from qtro.data import SweepData
from qtro.device import resolve_device
from qtro.eval.paired import crossed_bootstrap
from qtro.eval.panel import parse_model_specs, score_checkpoint_directory

METRICS = ("queue_w1", "overflow_abs_error")


def tail_columns(panel: SweepData, device: str, c_min: int, grid) -> np.ndarray:
    """Effective tail source columns per condition at h=1."""
    tail = grid.tail_mask.to(device)
    counts = []
    for start in range(0, len(panel), 8):
        idx = torch.arange(start, min(start + 8, len(panel)))
        b = panel.batch(idx, device)
        c = b["ref_counts"][1]
        counts.append(((c >= c_min) & tail.unsqueeze(0)).sum(dim=1).cpu().numpy())
    return np.concatenate(counts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True, help="dataset directory")
    ap.add_argument("--split", required=True, help="split name inside the dataset")
    ap.add_argument("--models", required=True,
                    help="comma-separated name=run-directory entries")
    ap.add_argument("--control", default="qtro")
    ap.add_argument("--horizons", default="1,5,10,20")
    # The aggregation names are the ones configuration_column_aggregates
    # returns; 'macro' is accepted as another name for 'uniform'.
    ap.add_argument("--aggregation", default="tail_only",
                    choices=("tail_only", "uniform", "macro",
                             "occupancy_weighted"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.aggregation == "macro":
        args.aggregation = "uniform"

    device = resolve_device(args.device)
    horizons = tuple(int(h) for h in args.horizons.split(","))
    panel = SweepData(args.panel, horizons=list(horizons), splits=(args.split,),
                      want_refs=True, require_targets=False)
    rng = np.random.default_rng(args.seed)

    scores = {}
    grid = None
    for name, directory in parse_model_specs(args.models).items():
        model_path = sorted(Path(directory).glob("checkpoints/*.pt"))[0]
        _, grid, cfg, _ = load_checkpoint(model_path, device)
        scores[name] = score_checkpoint_directory(directory, panel, device, horizons,
                                                  METRICS)
        print(f"  {name}: {len(scores[name])} seeds", flush=True)

    cols = tail_columns(panel, device, cfg.data.c_min, grid)
    report = {"panel": args.panel, "split": args.split,
              "conditions": len(panel), "aggregation": args.aggregation,
              "tail_columns": {
                  "mean": float(cols.mean()), "median": float(np.median(cols)),
                  "conditions_at_or_above_20": int((cols >= 20).sum()),
                  "conditions": int(cols.size)},
              "mean": {}, "paired": {}}
    print(f"\ntail source columns per condition: mean {cols.mean():.2f}, "
          f"median {np.median(cols):.0f}, "
          f"{int((cols >= 20).sum())}/{cols.size} conditions reach the "
          f"predeclared twenty")

    for name, per_seed in scores.items():
        report["mean"][name] = {
            m: {str(h): float(np.nanmean(
                [per_seed[s][h][m][args.aggregation] for s in per_seed]))
                for h in horizons}
            for m in METRICS}

    print(f"\n{args.aggregation} means")
    print(f"{'model':>10}" + "".join(f"{m + ' h' + str(h):>22}"
                                     for m in METRICS for h in horizons))
    for name in scores:
        row = f"{name:>10}"
        for m in METRICS:
            for h in horizons:
                row += f"{report['mean'][name][m][str(h)]:>22.5f}"
        print(row)

    control = scores[args.control]
    print(f"\npaired against {args.control}, opponent minus control "
          f"(positive favours {args.control} on an error metric)")
    for name in scores:
        if name == args.control:
            continue
        report["paired"][name] = {}
        shared = sorted(set(control) & set(scores[name]))
        for m in METRICS:
            for h in horizons:
                delta = np.stack([
                    scores[name][s][h][m][args.aggregation]
                    - control[s][h][m][args.aggregation] for s in shared])
                stats = crossed_bootstrap(delta, n_resamples=args.resamples,
                                          rng=rng)
                low, high = stats["mean_ci"]
                base = float(np.nanmean(
                    [control[s][h][m][args.aggregation] for s in shared]))
                report["paired"][name][f"{m}_h{h}"] = {
                    "advantage": float(stats["mean_advantage"]),
                    "ci_low": float(low), "ci_high": float(high),
                    "significant": bool(low > 0 or high < 0),
                    "relative_improvement": (
                        float(stats["mean_advantage"] / base) if base else None),
                }
                r = report["paired"][name][f"{m}_h{h}"]
                mark = "*" if r["significant"] else " "
                rel = (f"{100 * r['relative_improvement']:+.1f}%"
                       if r["relative_improvement"] is not None else "--")
                print(f"  {name:>8} {m:>20} h={h:<3} "
                      f"{r['advantage']:+.5f} [{r['ci_low']:+.5f},"
                      f"{r['ci_high']:+.5f}]{mark}  {rel}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
