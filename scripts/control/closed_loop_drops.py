"""Packet drops per closed-loop episode, paired by seed.

Drops are the network-side cost of holding a short queue. The closed-loop
report already carries them per episode; this reports the per-model means
beside mean queue, p95 and mean window, and the paired drop differences
against the control. Negative favours the control.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np


def episodes_by_label(path) -> tuple[dict, dict]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    by = collections.defaultdict(dict)
    for e in d["episodes"]:
        by[e["label"]][e["seed"]] = e
    return d, by


def paired(by, a, b, field, resamples: int, seed: int):
    """Mean of (b - a) over the seeds both ran, with a percentile interval."""
    seeds = sorted(set(by[a]) & set(by[b]))
    diff = np.array([float(by[b][s][field]) - float(by[a][s][field])
                     for s in seeds])
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(diff, diff.size, replace=True).mean()
                     for _ in range(resamples)])
    return diff.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5), len(seeds)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True, help="JSON written by closed_loop.py")
    ap.add_argument("--control", default="qtro")
    ap.add_argument("--baselines", default="fixed,cnto,seqpred,deeponet")
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    baselines = [b.strip() for b in args.baselines.split(",") if b.strip()]

    d, by = episodes_by_label(args.report)
    print(f"horizon={d.get('horizon')}  forecast_s={d.get('forecast_s')}  "
          f"control_interval_s={d.get('control_interval_s')}  lam={d.get('lam')}  "
          f"lam_move={d.get('lam_move')}\n")
    print("%-10s %9s %11s %10s %10s"
          % ("model", "drops", "mean queue", "p95", "mean cwnd"))
    means = {}
    for lab in [args.control] + [b for b in baselines if b in by]:
        es = list(by[lab].values())
        means[lab] = {"drops": float(np.mean([e["drops"] for e in es])),
                      "mean_queue": float(np.mean([e["mean_queue"] for e in es])),
                      "p95_queue": float(np.mean([e["p95_queue"] for e in es])),
                      "mean_cwnd": float(np.mean([e["mean_cwnd"] for e in es]))}
        m = means[lab]
        print("%-10s %9.1f %11.2f %10.2f %10.2f"
              % (lab, m["drops"], m["mean_queue"], m["p95_queue"], m["mean_cwnd"]))

    print(f"\npaired differences in drops, (other - {args.control}), "
          f"negative favours {args.control}")
    paired_drops = {}
    for lab in baselines:
        if lab not in by:
            continue
        m, lo, hi, n = paired(by, args.control, lab, "drops", args.resamples, args.seed)
        paired_drops[lab] = {"mean": float(m), "ci_low": float(lo), "ci_high": float(hi),
                             "seeds": n, "significant": bool(not lo <= 0 <= hi)}
        print("   vs %-9s %+9.1f  [%+8.1f, %+8.1f]  seeds=%d %s"
              % (lab, m, lo, hi, n, "" if lo <= 0 <= hi else "excl0"))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"means": means, "paired_drops": paired_drops},
                                             indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
