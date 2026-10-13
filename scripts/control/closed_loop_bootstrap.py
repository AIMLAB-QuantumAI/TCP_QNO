"""Paired seed bootstrap over closed-loop episodes.

Positive favours the control: baseline minus control on every metric, all of
which are lower-is-better. This feeds the closed-loop table's intervals.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

METRICS = {"IAE": lambda e: e["tracking_mae"], "q95": lambda e: e["p95_queue"],
           "P": lambda e: e["tail_fraction"], "|qbar-q*|": lambda e: abs(e["mean_offset"]),
           "Forecast": lambda e: (e["prediction"] or {}).get("mean_queue_error", np.nan)}


def paired(episodes: list[dict], control: str, base: str, resamples: int,
           seed: int) -> dict:
    by = {(e["label"], e["seed"]): e for e in episodes}
    seeds = sorted({e["seed"] for e in episodes})
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(seeds), size=(resamples, len(seeds)))
    out = {}
    for name, f in METRICS.items():
        diff = np.array([f(by[(base, s)]) - f(by[(control, s)]) for s in seeds], float)
        if np.isnan(diff).any():
            continue
        boot = diff[idx].mean(1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out[name] = {"mean": float(diff.mean()), "ci_low": float(lo),
                     "ci_high": float(hi), "significant": bool(lo > 0 or hi < 0)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True, help="JSON written by closed_loop.py")
    ap.add_argument("--control", default="qtro")
    ap.add_argument("--baselines", default="fixed,cnto,seqpred,deeponet")
    ap.add_argument("--resamples", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    episodes = json.loads(Path(args.report).read_text(encoding="utf-8"))["episodes"]
    result = {}
    for base in [b.strip() for b in args.baselines.split(",") if b.strip()]:
        result[base] = paired(episodes, args.control, base, args.resamples, args.seed)
        print(f"{Path(args.report).name}  vs {base}:")
        for k, r in result[base].items():
            print(f"   {k:10s} {r['mean']:+8.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"
                  f"{'  *' if r['significant'] else ''}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
