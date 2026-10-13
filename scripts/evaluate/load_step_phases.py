"""Split the load-step response into phases before comparing models.

Integrating the error over the whole post-step window treats it as
homogeneous, and it is not. Every step drives this 90-packet buffer to its
ceiling, so for part of the response the reference is very nearly a point
mass on the top queue bin and every predictor is trivially right about it.
Averaging that stretch together with the rest dilutes whatever separation
exists. This feeds the load-step phase table.

This re-reads the per-sample series already stored by
scripts/evaluate/load_steps.py and reports each phase separately:

  rise      from the step to the peak of the reference mean queue;
  peak      the saturated stretch, defined as samples whose reference mean
            queue is within 10% of the peak;
  recovery  everything after the queue leaves that band.

The phase boundaries come from the *reference*, never from a model, so no
model can be favoured by where the lines fall. Nothing is re-simulated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from qtro.eval.paired import crossed_bootstrap


def reference_mean_queue(comparison: dict, condition: str) -> np.ndarray:
    """Recover the shape of the reference response from the persistence entry.

    Persistence holds its start distribution fixed, so its per-sample mean
    queue error is |q0 - reference|, which peaks where the reference does.
    """
    rows = comparison[condition]["persistence"]
    return np.asarray(rows[0]["mean_queue_error"], dtype=np.float64)


def phases(curve: np.ndarray, band: float = 0.10):
    """Rise, peak and recovery index sets from the reference response."""
    peak = int(np.argmax(curve))
    threshold = curve[peak] * (1.0 - band)
    saturated = np.flatnonzero(curve >= threshold)
    first, last = int(saturated.min()), int(saturated.max())
    index = np.arange(curve.size)
    return {
        "rise": index[:first],
        "peak": index[first:last + 1],
        "recovery": index[last + 1:],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comparison", required=True,
                    help="JSON written by scripts/evaluate/load_steps.py --out")
    ap.add_argument("--metric", default="tv",
                    choices=("tv", "mean_queue_error"))
    ap.add_argument("--learned", default="qtro,cnto,seqpred,deeponet",
                    help="comma-separated learned-model labels as stored in the comparison")
    ap.add_argument("--control", default="qtro")
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    learned = tuple(n.strip() for n in args.learned.split(",") if n.strip())
    data = json.loads(Path(args.comparison).read_text(encoding="utf-8"))
    comparison = data["results"]
    conditions = sorted(comparison)
    rng = np.random.default_rng(args.seed)

    # Phase boundaries per condition, from the reference alone.
    bounds = {c: phases(reference_mean_queue(comparison, c)) for c in conditions}
    report = {"metric": args.metric, "control": args.control,
              "conditions": conditions, "phases": {}, "bounds": {
                  c: {k: [int(v.min()), int(v.max())] if v.size else None
                      for k, v in b.items()} for c, b in bounds.items()}}

    print(f"{'condition':>26}  {'rise':>10}{'peak':>10}{'recovery':>11}")
    for c in conditions:
        b = bounds[c]
        print(f"{c:>26}  {b['rise'].size:>10}{b['peak'].size:>10}"
              f"{b['recovery'].size:>11}")

    for phase in ("rise", "peak", "recovery", "all"):
        entry = {"mean": {}, "paired": {}}
        per_model = {}
        for name in learned + ("persistence", "split_half", "fluid_jump"):
            if name not in comparison[conditions[0]]:
                continue
            # (seeds, conditions) mean error inside the phase.
            rows = []
            n_seeds = len(comparison[conditions[0]][name])
            for s in range(n_seeds):
                per_condition = []
                for c in conditions:
                    series = np.asarray(
                        comparison[c][name][s][args.metric], dtype=np.float64)
                    idx = (np.arange(series.size) if phase == "all"
                           else bounds[c][phase])
                    idx = idx[idx < series.size]
                    per_condition.append(
                        float(series[idx].mean()) if idx.size else np.nan)
                rows.append(per_condition)
            per_model[name] = np.asarray(rows, dtype=np.float64)
            entry["mean"][name] = float(np.nanmean(per_model[name]))

        control = per_model[args.control]
        for name in learned:
            if name == args.control or name not in per_model:
                continue
            other = per_model[name]
            n = min(control.shape[0], other.shape[0])
            # Opponent minus control, so a positive value favours the control.
            delta = other[:n] - control[:n]
            stats = crossed_bootstrap(delta, n_resamples=args.resamples, rng=rng)
            low, high = stats["mean_ci"]
            entry["paired"][name] = {
                "advantage": float(stats["mean_advantage"]),
                "ci_low": float(low), "ci_high": float(high),
                "significant": bool(low > 0 or high < 0),
            }
        report["phases"][phase] = entry

    print(f"\nmean {args.metric} by phase")
    names = [n for n in ("split_half", "persistence", *learned, "fluid_jump")
             if n in report["phases"]["all"]["mean"]]
    print(f"{'phase':>10}" + "".join(f"{n:>13}" for n in names))
    for phase in ("rise", "peak", "recovery", "all"):
        row = f"{phase:>10}"
        for n in names:
            row += f"{report['phases'][phase]['mean'][n]:>13.4f}"
        print(row)

    print(f"\npaired against {args.control}, opponent minus control "
          f"(positive favours {args.control})")
    for phase in ("rise", "peak", "recovery", "all"):
        cells = []
        for name, r in report["phases"][phase]["paired"].items():
            mark = "*" if r["significant"] else " "
            cells.append(f"{name} {r['advantage']:+.4f} "
                         f"[{r['ci_low']:+.4f},{r['ci_high']:+.4f}]{mark}")
        print(f"  {phase:>9}  " + "   ".join(cells))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
