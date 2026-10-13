"""Aggregate several seeds of the long-horizon report into the numbers quoted.

Each input is one ``long_horizon.py`` report. ``validity[h]`` carries finite,
min_element, max_colsum_error and the mean-queue range, or nonfinite_fraction
when the channel has blown up; ``drift[h]`` carries tv and mean_queue_error
against the ns-3 occupancy. A diverged channel stores null for tv, so every
read is guarded. The six sections are the statements the long-horizon
paragraph makes, each recomputed over every seed given.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HS = ["1", "5", "20", "50", "100", "200", "500", "1024"]


def val(d: dict, ch: str, sect: str, h: str, key: str) -> float | None:
    r = d["channels"].get(ch, {}).get(sect, {}).get(h)
    if not isinstance(r, dict):
        return None
    v = r.get(key)
    return None if v is None else float(v)


def rng(vals) -> tuple[float | None, float | None]:
    vals = [v for v in vals if v is not None]
    return (min(vals), max(vals)) if vals else (None, None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="long-horizon JSON reports, one per seed")
    ap.add_argument("--control", default="qtro",
                    help="channel label whose smallest entry and drift ranking are reported")
    ap.add_argument("--constrained", default="qtro,cnto,deeponet",
                    help="comma-separated labels of the structurally constrained channels")
    ap.add_argument("--learned", default="qtro,cnto,seqpred,deeponet",
                    help="comma-separated labels of the learned channels")
    ap.add_argument("--out", default=None, help="write the sections as JSON here too")
    args = ap.parse_args()
    constrained = [c.strip() for c in args.constrained.split(",") if c.strip()]
    learned = [c.strip() for c in args.learned.split(",") if c.strip()]

    runs = []
    for p in args.inputs:
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        runs.append((d.get("seed"), d))
    print(f"seeds {[s for s, _ in runs]}, panel {runs[0][1]['panel']}, "
          f"{runs[0][1]['conditions']} conditions\n")
    sections: dict[str, dict] = {}

    print("=" * 72 + "\n1. CONSTRAINED CHANNELS STAY VALID AT h = 1024\n" + "=" * 72)
    finite = {}
    for ch in constrained:
        fins = [d["channels"][ch]["validity"]["1024"].get("finite") for _s, d in runs]
        finite[ch] = bool(all(fins))
        print(f"  {ch:<10} finite on every seed: {all(fins)}")
    print(f"  -> all constrained channels valid on all seeds: {all(finite.values())}\n")
    sections["constrained_finite_at_1024"] = finite

    print("=" * 72 + "\n2. COLUMN MASS ERROR, h = 1 TO h = 1024\n" + "=" * 72)
    colsum = {}
    for h in ("1", "1024"):
        lo, hi = rng([val(d, ch, "validity", h, "max_colsum_error")
                      for _s, d in runs for ch in constrained])
        colsum[h] = [lo, hi]
        print(f"  h={h:<5} max column-sum error over seeds and channels: {lo:.3e} to {hi:.3e}")
    sections["max_colsum_error_range"] = colsum

    print("\n" + "=" * 72 + f"\n3. {args.control.upper()} SMALLEST ENTRY AT h = 1024\n" + "=" * 72)
    mins = [val(d, args.control, "validity", "1024", "min_element") for _s, d in runs]
    print(f"  per seed: {['%.3e' % m for m in mins]}")
    print(f"  worst (smallest) over seeds: {min(mins):.3e}   "
          f"all strictly positive: {all(m > 0 for m in mins)}")
    sections["control_min_element_at_1024"] = {"per_seed": mins, "worst": min(mins),
                                               "all_positive": bool(all(m > 0 for m in mins))}

    print("\n" + "=" * 72 + "\n4. SEQPRED WITHOUT REPAIR\n" + "=" * 72)
    raw = {}
    for h in ("20", "200"):
        q = [val(d, "seqpred_raw", "validity", h, "mean_queue_max") for _s, d in runs]
        nf = [val(d, "seqpred_raw", "validity", h, "nonfinite_fraction") for _s, d in runs]
        raw[h] = {"mean_queue_max": q, "nonfinite_fraction": nf}
        print(f"  h={h:<4} mean_queue_max "
              f"{['%.2e' % x if x is not None else 'n/a' for x in q]}")
        print(f"        nonfinite fraction "
              f"{['%.3f' % x if x is not None else 'n/a' for x in nf]}")
    sections["seqpred_raw"] = raw

    print("\n" + "=" * 72 + "\n5. DRIFT TV AT h = 1024\n" + "=" * 72)
    print(f"  {'channel':<14} {'min':<12} {'max':<12}")
    drift = {"tv": {}, "mean_queue_error": {}}
    for ch in learned + ["ns3_empirical"]:
        lo, hi = rng([val(d, ch, "drift", "1024", "tv") for _s, d in runs])
        if lo is not None:
            drift["tv"][ch] = [lo, hi]
            print(f"  {ch:<14} {lo:<12.4f} {hi:<12.4f}")
    print("\n  mean queue error at h=1024")
    for ch in learned:
        lo, hi = rng([val(d, ch, "drift", "1024", "mean_queue_error") for _s, d in runs])
        if lo is not None:
            drift["mean_queue_error"][ch] = [lo, hi]
            print(f"  {ch:<14} {lo:<12.2f} {hi:<12.2f}")
    sections["drift_at_1024"] = drift

    print("\n" + "=" * 72
          + f"\n6. IS {args.control.upper()} LOWEST AMONG LEARNED AT EVERY HORIZON\n"
          + "=" * 72)
    lowest = {}
    for s, d in runs:
        bad = []
        for h in HS:
            v = {ch: val(d, ch, "drift", h, "tv") for ch in learned}
            v = {k: x for k, x in v.items() if x is not None}
            if v and min(v, key=v.get) != args.control:
                bad.append((h, min(v, key=v.get)))
        lowest[str(s)] = bad
        print(f"  seed {s}: "
              f"{'lowest at every horizon' if not bad else 'NOT lowest at %s' % bad}")
    sections["control_not_lowest_at"] = lowest

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(sections, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
