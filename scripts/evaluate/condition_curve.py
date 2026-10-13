"""Score the training-set-size curve on the held-out ns-3 dataset.

The question is not only whether QTRO wins at the full training set, but
whether the margin depends on how much of the (u, xi) space the training set
covered. A structural constraint removes hypotheses, so if that is what the
operator is buying, the margin should be largest where the data pins the
operator down least. This feeds the training-set-size figure.

Every size is scored at the same seeds, including the full-data point, so the
curve is not part three-seed and part five-seed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from qtro.data import SweepData
from qtro.device import resolve_device
from qtro.eval.paired import crossed_bootstrap
from qtro.eval.panel import parse_model_specs, score_checkpoint_directory

SUBSET_SIZES = (27, 54, 108)
FULL_SIZE = 216
DEFAULT_ARMS = "qtro=qtro,cnto=cnto,seqpred=seqpred,deeponet=deeponet"


def build_sizes(root, arms_spec, full_spec):
    """(n_conditions, {label: directory}) for the subsets and the full set."""
    arms = parse_model_specs(arms_spec)
    full = parse_model_specs(full_spec)
    if set(arms) != set(full):
        raise SystemExit(f"--arms {sorted(arms)} and --full {sorted(full)} "
                         f"name different models")
    sizes = [(n, {label: f"{root}/n{n:03d}/{directory}"
                  for label, directory in arms.items()})
             for n in SUBSET_SIZES]
    sizes.append((FULL_SIZE, dict(full)))
    return tuple(sizes)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True, help="dataset directory")
    ap.add_argument("--split", required=True, help="split name inside the dataset")
    ap.add_argument("--horizons", default="5,20")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--bootstrap-seed", type=int, default=20260902)
    ap.add_argument("--root", required=True,
                    help="root holding the nXXX subset runs")
    ap.add_argument("--arms", default=DEFAULT_ARMS,
                    help="comma-separated label=subdirectory entries")
    ap.add_argument("--full", required=True,
                    help="comma-separated label=run-directory entries for the full set")
    ap.add_argument("--control", default="qtro",
                    help="label every pairing is taken against")
    ap.add_argument("--opponents", default="cnto,seqpred,deeponet",
                    help="labels to pair against the control")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = resolve_device(args.device)
    SIZES = build_sizes(args.root, args.arms, args.full)
    opponents = tuple(o.strip() for o in args.opponents.split(",") if o.strip())
    labels = (args.control,) + tuple(o for o in opponents
                                     if o != args.control)
    opponents = tuple(o for o in opponents if o != args.control)

    horizons = tuple(int(h) for h in args.horizons.split(","))
    seeds = {int(s) for s in args.seeds.split(",")}
    panel = SweepData(args.panel, horizons=list(horizons), splits=(args.split,),
                      want_refs=True, require_targets=False)
    rng = np.random.default_rng(args.bootstrap_seed)

    report = {"panel": args.panel, "split": args.split,
              "conditions": len(panel), "seeds": sorted(seeds),
              "horizons": list(horizons), "sizes": {}}

    for n_conditions, models in SIZES:
        scores = {}
        for name, directory in models.items():
            path = Path(directory)
            if not path.exists():
                print(f"missing {directory}, skipping n={n_conditions}")
                scores = {}
                break
            scored = score_checkpoint_directory(path, panel, device, horizons,
                                                ("fidelity",), seeds)
            scores[name] = {s: {h: scored[s][h]["fidelity"]["occupancy_weighted"]
                                for h in horizons} for s in scored}
            print(f"  n={n_conditions} {name}: {len(scores[name])} seeds",
                  flush=True)
        if not scores or any(not v for v in scores.values()):
            continue

        entry = {"mean_fidelity": {}, "paired": {}}
        for name, per_seed in scores.items():
            entry["mean_fidelity"][name] = {
                str(h): float(np.mean([per_seed[s][h].mean()
                                       for s in per_seed]))
                for h in horizons}
        for opponent in opponents:
            shared = sorted(set(scores[args.control])
                            & set(scores[opponent]))
            entry["paired"][opponent] = {}
            for h in horizons:
                # Opponent minus control: a negative advantage favours it.
                delta = np.stack([scores[opponent][s][h]
                                  - scores[args.control][s][h]
                                  for s in shared])
                stats = crossed_bootstrap(delta, n_resamples=args.resamples,
                                          rng=rng)
                low, high = stats["mean_ci"]
                entry["paired"][opponent][str(h)] = {
                    "advantage": float(stats["mean_advantage"]),
                    "ci_low": float(low), "ci_high": float(high),
                    "significant": bool(low > 0 or high < 0),
                    "seeds": len(shared),
                }
        report["sizes"][str(n_conditions)] = entry

    print(f"\n{'n':>6}" + "".join(f"{m + ' h' + str(h):>16}"
                                  for m in labels
                                  for h in horizons))
    for n_conditions, _ in SIZES:
        e = report["sizes"].get(str(n_conditions))
        if not e:
            continue
        row = f"{n_conditions:>6}"
        for m in labels:
            for h in horizons:
                row += f"{e['mean_fidelity'][m][str(h)]:>16.4f}"
        print(row)

    print(f"\npaired, opponent minus {args.control} "
          f"(negative favours {args.control})")
    for n_conditions, _ in SIZES:
        e = report["sizes"].get(str(n_conditions))
        if not e:
            continue
        for opponent in opponents:
            cells = []
            for h in horizons:
                r = e["paired"][opponent][str(h)]
                mark = "*" if r["significant"] else " "
                cells.append(f"h={h} {r['advantage']:+.5f} "
                             f"[{r['ci_low']:+.5f},{r['ci_high']:+.5f}]{mark}")
            print(f"  n={n_conditions:>4} vs {opponent:<8} " + "   ".join(cells))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
