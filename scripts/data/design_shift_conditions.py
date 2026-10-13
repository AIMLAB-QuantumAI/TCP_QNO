"""Design the two traffic- and topology-shift condition lists.

Both reuse the *same* held-out (u, xi) coordinates the main dataset is scored
on, and change only what the operator was never told about. That is what
makes the comparison paired: the difference between a condition here and its
twin in the main dataset is the shift and nothing else, so a drop in accuracy
cannot be attributed to having moved to easier or harder control settings.

Neither shift is representable in xi. The operator conditions on capacity,
RTT, session count and link-error rate, and none of those describe bursty
cross traffic or a second queue downstream. So no retraining is possible or
intended: these datasets are scored zero-shot, exactly as the out-of-envelope
dataset is, and they ask whether the ranking survives traffic and topology
the model has no coordinate for.

burst   Unresponsive Pareto on/off UDP sources sharing the bottleneck. Floyd
        and Jacobson motivate RED by burst absorption, and the diffusion
        approximations drive their queues with heavy-tailed arrivals. The
        stationary datasets offer only long-lived bulk TCP, which is the one
        class RED was not designed around.

tandem  Two RED bottlenecks in series with cross traffic on the second, after
        Misra et al., whose fluid-model validation covers a network of AQM
        routers rather than a single dumbbell. The emitted queue is the first
        bottleneck's, the one the RED action applies to.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# Burst levels. Two sources at 30% of the link with a 20% duty cycle offer
# about 12% mean load; four offer about 24%. Shape 1.5 is inside the
# long-range-dependent regime and is the value the self-similar traffic
# literature uses for on-times.
BURST_LEVELS = (
    {"burstFlows": 2, "burstRateFrac": 0.30, "burstOnMeanS": 0.10,
     "burstOffMeanS": 0.40, "burstShape": 1.5},
    {"burstFlows": 4, "burstRateFrac": 0.30, "burstOnMeanS": 0.10,
     "burstOffMeanS": 0.40, "burstShape": 1.5},
)

# Tandem levels, chosen against a constraint of the topology rather than by
# preference: in a tandem only the tightest link queues. Steady cross traffic
# on the second hop makes it tighter at all times, the tagged flow backs off
# before the first hop fills, and the queue this dataset measures collapses --
# the first attempt produced 16 occupied cells per condition against the main
# dataset's 72. Bursty cross traffic makes the binding link alternate instead.
#
# Measured on a probe run, first-hop / second-hop mean queue in packets:
#   ratio 1.0, cross 2   6.8 / 22.2, second hop busy 57% of the time
#   ratio 1.2, cross 2  17.7 / 17.0, busy 42%
#   ratio 1.2, cross 4   1.1 / 17.2, busy 44%
# The two kept are the ones whose first hop still carries a distribution
# comparable to the dumbbell's while the second is genuinely active.
TANDEM_LEVELS = (
    {"capacityRatio": 1.2, "crossFlows": 2},
    {"capacityRatio": 1.0, "crossFlows": 2},
)

SHARD = re.compile(
    r"shard_C(\d+)_R(\d+)_N(\d+)_E(\d+)_P(\d+)_T(\d+)\.npz")


def heldout_conditions(panel: Path) -> list[dict]:
    manifest = json.loads((panel / "manifest.json").read_text(encoding="utf-8"))
    return [s for s in manifest["shards"] if s.get("split") == "ns3_heldout"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True,
                    help="main dataset directory whose manifest holds the held-out conditions")
    ap.add_argument("--kind", choices=("burst", "tandem"), required=True)
    ap.add_argument("--replications", type=int, default=64)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = heldout_conditions(Path(args.panel))
    if not base:
        raise SystemExit(f"no held-out shards in {args.panel}")

    levels = BURST_LEVELS if args.kind == "burst" else TANDEM_LEVELS
    conditions = []
    for level in levels:
        for shard in base:
            condition = {
                "capacity_pps": shard["capacity_pps"],
                "rtt_prop_s": shard["rtt_prop_s"],
                "n_flows": shard["n_flows"],
                "p_err": shard["p_err"],
                "p_max": shard["p_max"],
                "min_th": shard["min_th"],
                "max_th_frac": shard["max_th_frac"],
                "buffer_pkts": shard["buffer_pkts"],
                "w_q": shard["w_q"],
                "replications": args.replications,
                "split": f"ns3_{args.kind}",
                "extra": dict(level),
            }
            conditions.append(condition)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(conditions, indent=2), encoding="utf-8")
    runs = sum(c["replications"] for c in conditions)
    print(f"{len(conditions)} conditions over {len(levels)} levels, "
          f"{runs} ns-3 runs")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
