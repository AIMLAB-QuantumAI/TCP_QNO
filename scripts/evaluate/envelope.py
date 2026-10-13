"""Decompose accuracy over the operating envelope.

Hollot et al. do not design against an average operating point. They design
against the worst corner of an envelope in the number of sessions, the
round-trip time and the link capacity, because a controller that is stable on
average and unstable at the load extreme is not a usable controller.

A mean over all conditions answers "is the model better on average" and hides
"where does it break". This script stratifies the same per-condition scores
by each physical axis, reports the level at which the model is weakest, and
reports whether the advantage over a baseline survives inside each level. It
feeds the envelope table.

Two different worst cases are reported, and they are not interchangeable:

* absolute worst level -- where the model's own accuracy is lowest. This is
  the operating point a control engineer has to live with.
* paired worst level -- where the advantage over the baseline is smallest.
  A pooled advantage can be carried entirely by a few easy configurations.

Uses the same crossed bootstrap as the pooled comparison, so a level's
interval is directly comparable to the pooled one; it is only wider because
the level holds fewer configurations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from qtro.data import SweepData
from qtro.device import resolve_device
from qtro.eval.paired import crossed_bootstrap
from qtro.eval.panel import parse_model_specs, score_run_directory

# (u, xi) laid out as generate_dataset.py writes them.
AXES = {
    "capacity_pps": ("xi", 0),
    "rtt_s": ("xi", 1),
    "n_flows": ("xi", 2),
    "p_err": ("xi", 3),
    "p_max": ("u", 0),
    "min_th": ("u", 1),
}
# Hollot's envelope is the plant, not the control knob. Both are reported,
# but only the plant axes answer "is the model usable across the network".
PLANT_AXES = ("capacity_pps", "rtt_s", "n_flows")


def axis_values(panel: SweepData, axis: str) -> np.ndarray:
    source, column = AXES[axis]
    table = panel.u if source == "u" else panel.xi
    return np.asarray(table[:, column].detach().cpu(), dtype=np.float64)


def stratify(scores: dict, control: str, panel: SweepData, horizons,
             resamples: int, rng: np.random.Generator) -> dict:
    out: dict[str, dict] = {}
    for axis in AXES:
        values = axis_values(panel, axis)
        levels = sorted(np.unique(values))
        if len(levels) < 2:
            continue
        entry: dict[str, dict] = {}
        for level in levels:
            mask = values == level
            block: dict[str, dict] = {"n_configurations": int(mask.sum())}
            for h in horizons:
                for metric in ("fidelity", "tv"):
                    absolute = {
                        name: float(np.nanmean(
                            [seeds[s][h][metric]["occupancy_weighted"][mask].mean()
                             for s in seeds]))
                        for name, seeds in scores.items()}
                    sign = 1.0 if metric == "fidelity" else -1.0
                    paired = {}
                    for name, seeds in scores.items():
                        if name == control:
                            continue
                        shared = sorted(set(seeds) & set(scores[control]))
                        delta = np.stack([
                            sign * (seeds[s][h][metric]["occupancy_weighted"][mask]
                                    - scores[control][s][h][metric]["occupancy_weighted"][mask])
                            for s in shared])
                        paired[name] = crossed_bootstrap(
                            delta, n_resamples=resamples, rng=rng)
                    block[f"h{h}_{metric}"] = {"absolute": absolute,
                                               "paired": paired}
            entry[f"{level:g}"] = block
        out[axis] = entry
    return out


def worst_levels(strata: dict, control: str, key: str) -> dict:
    """Absolute-worst and paired-worst level for one horizon and metric.

    The paired numbers are re-signed so that a positive value always means
    the control model is ahead, whichever metric is being reported.
    """
    higher_is_better = key.endswith("fidelity")
    out = {}
    for axis, levels in strata.items():
        def absolute(level: str) -> float:
            value = levels[level][key]["absolute"][control]
            return value if higher_is_better else -value

        worst_abs = min(levels, key=absolute)
        record = {"absolute_worst": {
            "level": worst_abs,
            "value": levels[worst_abs][key]["absolute"][control]}}
        for name in next(iter(levels.values()))[key]["paired"]:
            worst = min(
                levels,
                key=lambda level: -levels[level][key]["paired"][name]["mean_advantage"])
            stat = levels[worst][key]["paired"][name]
            record[f"paired_worst_vs_{name}"] = {
                "level": worst,
                "advantage": -stat["mean_advantage"],
                "ci": [-stat["mean_ci"][1], -stat["mean_ci"][0]],
                "n_configurations": levels[worst]["n_configurations"]}
        out[axis] = record
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True, help="dataset directory")
    ap.add_argument("--split", required=True, help="split name inside the dataset")
    ap.add_argument("--models", required=True,
                    help="comma-separated name=run-directory entries")
    ap.add_argument("--control", default="qtro")
    ap.add_argument("--horizons", default="1,5,10,20")
    ap.add_argument("--report-horizon", type=int, default=20)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--bootstrap-seed", type=int, default=20260826)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = resolve_device(args.device)
    horizons = tuple(int(h) for h in args.horizons.split(",") if h.strip())
    if args.report_horizon not in horizons:
        raise SystemExit("--report-horizon must be one of --horizons")
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
    for name, seeds in scores.items():
        if not seeds:
            raise SystemExit(f"model {name!r} contributed no checkpoints")

    rng = np.random.default_rng(args.bootstrap_seed)
    strata = stratify(scores, args.control, panel, horizons, args.resamples,
                      rng)
    if not strata:
        raise SystemExit("no axis in this panel has more than one level")

    others = [n for n in scores if n != args.control]
    for metric in ("fidelity", "tv"):
        key = f"h{args.report_horizon}_{metric}"
        print(f"\n=== h={args.report_horizon} {metric}, stratified "
              f"(control {args.control}; + favours {args.control}) ===")
        head = (f"{'axis':>13} {'level':>8} {'n':>3} "
                + "".join(f"{n:>9}" for n in scores))
        print(head + "".join(f"{'vs ' + n:>27}" for n in others))
        for axis, levels in strata.items():
            for level, block in levels.items():
                cell = block[key]
                row = (f"{axis:>13} {level:>8} {block['n_configurations']:>3} "
                       + "".join(f"{cell['absolute'][n]:>9.4f}"
                                 for n in scores))
                for name in others:
                    stat = cell["paired"][name]
                    lo, hi = stat["mean_ci"]
                    mark = "*" if (lo > 0 or hi < 0) else " "
                    row += (f"  {-stat['mean_advantage']:>+8.5f}"
                            f" [{-hi:>+7.4f},{-lo:>+7.4f}]{mark}")
                print(row)

    print(f"\n=== worst level per axis, h={args.report_horizon} fidelity ===")
    worst = worst_levels(strata, args.control,
                         f"h{args.report_horizon}_fidelity")
    for axis, record in worst.items():
        tag = "plant" if axis in PLANT_AXES else "control"
        print(f"{axis:>13} [{tag:>7}] weakest at {record['absolute_worst']['level']:>8}"
              f", {args.control} fidelity "
              f"{record['absolute_worst']['value']:.4f}")
        for name in others:
            entry = record[f"paired_worst_vs_{name}"]
            lo, hi = entry["ci"]
            verdict = ("holds" if lo > 0 else
                       "reversed" if hi < 0 else "uncertain")
            print(f"{'':>13}  smallest margin vs {name:>8} at {entry['level']:>8}"
                  f": {entry['advantage']:+.5f} [{lo:+.5f}, {hi:+.5f}]"
                  f" (n={entry['n_configurations']}) -> {verdict}")

    if args.out:
        report = {"panel": args.panel, "split": args.split,
                  "control": args.control, "n_configurations": len(panel),
                  "n_resamples": args.resamples, "horizons": list(horizons),
                  "plant_axes": list(PLANT_AXES), "strata": strata,
                  "worst_levels": {
                      f"h{h}_{m}": worst_levels(strata, args.control,
                                                f"h{h}_{m}")
                      for h in horizons for m in ("fidelity", "tv")}}
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=1, default=float),
                                  encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
