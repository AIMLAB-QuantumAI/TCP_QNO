"""Score the fluid and diffusion domain baselines beside the learned models.

The fluid / control-theoretic line and the diffusion-approximation line are
the prior art the learned operator is compared against. This script measures
them on the same held-out ns-3 dataset the learned models are scored on, so
the claim that fluid models describe mean trajectories rather than tail
probabilities becomes a measurement. It feeds the domain-baseline table.

The analytic models are not trained and have no seed axis. They are broadcast
across the control's seeds so the paired bootstrap still resamples the two
axes it is built to resample; the seed axis then carries only the learned
model's own variability.

A positive effect favours the model over the control.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from qtro.baselines.fluid import ANCHORS, MODES, FluidPropagator
from qtro.config import load_config
from qtro.data import SweepData
from qtro.device import resolve_device
from qtro.eval.paired import crossed_bootstrap
from qtro.eval.panel import parse_model_specs, score_kernels, score_run_directory
from qtro.grid import Grid

# Distributional scores first, then the moments. The moments are the check
# that makes the distributional comparison interpretable: a fluid model that
# tracks the mean queue and still loses on fidelity has lost for the reason
# claimed, whereas one that misses the mean too is simply mis-run.
METRICS = ("fidelity", "tv", "mean_queue_abs_error", "queue_w1", "cwnd_w1")


def analytic_per_configuration(mode: str, panel: SweepData, grid, cfg,
                               device: str, horizons: tuple[int, ...],
                               anchor: str = "balanced") -> dict:
    """Per-configuration scores for one analytic model. Nothing is trained."""
    propagator = FluidPropagator(grid, cfg, mode, anchor)
    return score_kernels(lambda b: propagator.kernels(b["u"], b["xi"], horizons),
                         panel, grid, cfg.data.c_min, device, horizons, METRICS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True, help="dataset directory")
    ap.add_argument("--split", required=True, help="split name inside the dataset")
    ap.add_argument("--config", default=None,
                    help="YAML config (default: the built-in values)")
    ap.add_argument("--models", required=True,
                    help="comma-separated name=run-directory entries, learned")
    ap.add_argument("--analytic", default=",".join(MODES),
                    help="comma-separated fluid/diffusion modes")
    ap.add_argument("--anchors", default=",".join(ANCHORS),
                    help="how the other sessions' load is set")
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
    cfg = load_config(args.config, [f"sim.device={device}"])
    grid = Grid(cfg.grid, cfg.net, device=device)
    print(f"panel {args.panel}: {len(panel)} configurations, "
          f"split {args.split!r}, M={grid.M}")

    scores: dict[str, dict] = {}
    for name, path in parse_model_specs(args.models).items():
        print(f"{name} (learned):")
        scores[name] = score_run_directory(path, panel, device, horizons, METRICS)
    if args.control not in scores:
        raise SystemExit(f"control {args.control!r} needs --models entries")
    control_seeds = sorted(scores[args.control])

    modes = [m.strip() for m in args.analytic.split(",") if m.strip()]
    anchors = [a.strip() for a in args.anchors.split(",") if a.strip()]
    analytic = []
    for mode in modes:
        # The identity ignores the anchor, so running it twice would only
        # duplicate a row.
        for anchor in (anchors[:1] if mode == "identity" else anchors):
            label = mode if mode == "identity" else f"{mode}/{anchor[:3]}"
            print(f"{label} (analytic, no training):", flush=True)
            value = analytic_per_configuration(mode, panel, grid, cfg,
                                               device, horizons, anchor)
            # No seed axis: the same prediction is paired against every seed
            # of the control, so resampled seed variability is the control's.
            scores[label] = {seed: value for seed in control_seeds}
            analytic.append(label)

    floor = {}
    for h in horizons:
        valid = panel.ref_counts[h] >= cfg.data.c_min
        floor[h] = (float(panel.noise_fidelity[h][valid].mean()),
                    float(panel.noise_tv[h][valid].mean()))

    def mean_over(seeds: dict, h: int, metric: str) -> float:
        return float(np.nanmean([seeds[s][h][metric]["occupancy_weighted"].mean()
                                 for s in seeds]))

    print("\nabsolute, occupancy-weighted, mean over seeds and configurations")
    header = f"{'model':>12}" + "".join(f"{'fid h' + str(h):>10}"
                                        for h in horizons)
    print(header + "".join(f"{'TV h' + str(h):>10}" for h in horizons))
    for name, seeds in scores.items():
        f = [mean_over(seeds, h, "fidelity") for h in horizons]
        t = [mean_over(seeds, h, "tv") for h in horizons]
        print(f"{name:>12}" + "".join(f"{x:>10.4f}" for x in f)
              + "".join(f"{x:>10.4f}" for x in t))
    print(f"{'split-half':>12}"
          + "".join(f"{floor[h][0]:>10.4f}" for h in horizons)
          + "".join(f"{floor[h][1]:>10.4f}" for h in horizons))

    print("")
    print("moment tracking: does the model get the mean right at all?")
    moment_header = f"{'model':>12}"
    for h in horizons:
        moment_header += f"{'|dq| h' + str(h):>11}{'qW1 h' + str(h):>10}"
    print(moment_header)
    for name, seeds in scores.items():
        row = f"{name:>12}"
        for h in horizons:
            dq = mean_over(seeds, h, "mean_queue_abs_error")
            w1 = mean_over(seeds, h, "queue_w1")
            row += f"{dq:>11.3f}{w1:>10.4f}"
        print(row)
    print("  |dq| is the mean-queue error in packets against the reference "
          "column; qW1 is the normalised queue Wasserstein distance.")

    rng = np.random.default_rng(args.bootstrap_seed)
    control = scores[args.control]
    report = {"panel": args.panel, "split": args.split,
              "control": args.control, "n_configurations": len(panel),
              "n_resamples": args.resamples, "analytic": analytic,
              "split_half": {f"h{h}": floor[h] for h in horizons},
              "comparisons": {}}
    print(f"\npaired against {args.control}, {len(panel)} configurations, "
          f"{args.resamples} crossed resamples")
    print(f"{'model':>12} {'metric':>14} {'effect':>10} {'95% CI':>22}"
          f" {'win':>6}  verdict")
    print("-" * 84)
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
                print(f"{name:>12} {'h' + str(h) + '_' + metric:>14} "
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
