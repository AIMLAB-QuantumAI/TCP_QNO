"""Compare models on an ns-3 dataset, paired by seed and condition.

The models are trained on ns-3 transitions and scored on a disjoint ns-3
dataset in full. Seeds and conditions are crossed in the bootstrap, and the
split-half floor of the dataset is reported beside the models, because the
ranking only means something next to what the dataset's own two halves agree
on. This feeds the accuracy tables and the shift and extrapolation rows.

A positive effect favours the model over the control.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from qtro.data import SweepData
from qtro.device import resolve_device
from qtro.eval.paired import crossed_bootstrap
from qtro.eval.panel import parse_model_specs, score_run_directory, split_half_floor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True, help="dataset directory")
    ap.add_argument("--split", required=True, help="split name inside the dataset")
    ap.add_argument("--models", required=True,
                    help="comma-separated name=run-directory entries")
    ap.add_argument("--control", default="qtro")
    ap.add_argument("--horizons", default="1,5,10,20")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--bootstrap-seed", type=int, default=20260826)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = resolve_device(args.device)
    horizons = tuple(int(h) for h in args.horizons.split(",") if h.strip())
    panel = SweepData(args.panel, horizons=list(horizons),
                      splits=(args.split,), want_refs=True,
                      require_targets=False)
    print(f"panel {args.panel}: {len(panel)} configurations, "
          f"split {args.split!r}")

    scores = {}
    for name, path in parse_model_specs(args.models).items():
        print(f"{name}:")
        scores[name] = score_run_directory(path, panel, device, horizons,
                                           ("fidelity", "tv"))

    if args.control not in scores:
        raise SystemExit(f"control {args.control!r} not among {sorted(scores)}")
    control = scores[args.control]
    for name, seeds in scores.items():
        if not seeds:
            raise SystemExit(f"model {name!r} contributed no checkpoints")

    # Absolute scores first: the ranking only means something next to what the
    # panel's own two halves agree on.
    floor = split_half_floor(panel, horizons, c_min=32)

    print("\nabsolute, occupancy-weighted, mean over seeds and configurations")
    header = f"{'model':>10}" + "".join(f"{'fid h'+str(h):>10}" for h in horizons)
    print(header + "".join(f"{'TV h'+str(h):>10}" for h in horizons))
    for name, seeds in scores.items():
        f = [np.nanmean([seeds[s][h]["fidelity"]["occupancy_weighted"].mean()
                         for s in seeds]) for h in horizons]
        t = [np.nanmean([seeds[s][h]["tv"]["occupancy_weighted"].mean()
                         for s in seeds]) for h in horizons]
        print(f"{name:>10}" + "".join(f"{x:>10.4f}" for x in f)
              + "".join(f"{x:>10.4f}" for x in t))
    print(f"{'split-half':>10}"
          + "".join(f"{floor[h][0]:>10.4f}" for h in horizons)
          + "".join(f"{floor[h][1]:>10.4f}" for h in horizons))

    rng = np.random.default_rng(args.bootstrap_seed)
    report = {"panel": args.panel, "control": args.control,
              "n_configurations": len(panel), "n_resamples": args.resamples,
              "split_half": {f"h{h}": floor[h] for h in horizons},
              "comparisons": {}}
    print(f"\npaired against {args.control}, {len(panel)} configurations, "
          f"{args.resamples} crossed resamples")
    print(f"{'model':>10} {'metric':>14} {'effect':>10} {'95% CI':>22}"
          f" {'win':>6}  verdict")
    print("-" * 82)
    for name, seeds in scores.items():
        if name == args.control:
            continue
        shared = sorted(set(seeds) & set(control))
        entry = {}
        for h in horizons:
            for metric, sign in (("fidelity", 1.0), ("tv", -1.0)):
                delta = np.stack([sign * (seeds[s][h][metric]["occupancy_weighted"]
                                          - control[s][h][metric]["occupancy_weighted"])
                                  for s in shared])
                stat = crossed_bootstrap(delta, n_resamples=args.resamples,
                                         rng=rng)
                entry[f"h{h}_{metric}"] = stat
                lo, hi = stat["mean_ci"]
                verdict = (f"{name} better" if lo > 0 else
                           f"{args.control} better" if hi < 0 else "uncertain")
                print(f"{name:>10} {'h'+str(h)+'_'+metric:>14} "
                      f"{stat['mean_advantage']:>+10.5f} "
                      f"[{lo:>+9.5f}, {hi:>+9.5f}] {stat['win_rate']:>6.3f}"
                      f"  {verdict}")
        report["comparisons"][name] = entry

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=1, default=float),
                                  encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
