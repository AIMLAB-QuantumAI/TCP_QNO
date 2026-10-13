"""Generate a transition-kernel dataset from ns-3 runs.

Each condition is run as an ensemble of independent ns-3 replications. The
tagged flow's (window, queue, phase) samples are digitised onto the grid, the
h-step transitions are counted across the ensemble, and the replications are
split in half: half A becomes the fitting target, half B the grading
reference, so the split-half agreement is the measurement floor every model
is reported against. One shard per condition is written in the format
``qtro.data.SweepData`` loads.

``--split-name`` says what a shard is for: ``train`` is the only split the
training script loads, so an evaluation dataset cannot be trained on by
accident. A condition may override the split and the replication count.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path

import numpy as np

from qtro.config import load_config
from qtro.grid import Grid
from qtro.ns3 import Ns3Runner, add_ns3_arguments, condition_flags, csv_rows, digitise


def condition_replications(condition: dict, default: int) -> int:
    return int(condition.get("replications", default))


def run_one(runner: Ns3Runner, condition: dict, run_seed: int, samples: int,
            warmup: float, timeout_s: int, scenario: str) -> np.ndarray | None:
    """The tagged flow's ``(samples + 1, 3)`` rows of cwnd, queue, phase, or None.

    A condition may carry an ``extra`` mapping of scenario-specific flags. The
    shift datasets use it for the burst and tandem settings; the stationary
    datasets have none.
    """
    flags = (f"{condition_flags(condition)} --samples={samples} --warmup={warmup} "
             f"--runSeed={run_seed}")
    for key, value in condition.get("extra", {}).items():
        flags += f" --{key}={value}"
    stdout = runner.run(runner.wrapper_argv(scenario, flags), timeout_s)
    if stdout is None:
        return None
    rows = csv_rows(stdout)
    if len(rows) < samples + 1:
        return None
    return np.array([[float(r[1]), float(r[2]), float(r[3])]
                     for r in rows[:samples + 1]], dtype=np.float64)


def kernel_from_cells(cells: np.ndarray, horizon: int, M: int
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Column-normalised h-step kernel and per-source counts.

    ``cells`` is ``(n_replications, n_samples)``. Transitions are pooled over
    replications and over time.
    """
    counts = np.zeros((M, M), dtype=np.int64)
    source = cells[:, :-horizon].reshape(-1)
    destination = cells[:, horizon:].reshape(-1)
    np.add.at(counts, (destination, source), 1)
    column_total = counts.sum(axis=0)
    kernel = np.divide(counts, np.maximum(column_total, 1),
                       dtype=np.float64)
    return kernel.astype(np.float16), column_total.astype(np.int32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None,
                    help="YAML config (default: the built-in values; the grid is "
                         "identical in every config)")
    ap.add_argument("--out", required=True, help="dataset directory to write")
    ap.add_argument("--conditions", required=True,
                    help="JSON list of conditions, as in configs/conditions/")
    ap.add_argument("--scenario", default="qtro-dumbbell",
                    help="ns-3 scratch program; the tandem shift dataset uses "
                         "qtro-multibottleneck")
    ap.add_argument("--replications", type=int, default=64,
                    help="ns-3 runs per condition, split into two halves")
    ap.add_argument("--samples", type=int, default=600)
    ap.add_argument("--warmup", type=float, default=8.0)
    ap.add_argument("--horizons", default="1,5,10,20")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout-s", type=int, default=1800)
    ap.add_argument(
        "--split-name", default="ns3_validation",
        help=("value written to the split field. The training script loads "
              "only 'train', so a dataset meant for training must say "
              "'train'; leave the default for an evaluation-only dataset."))
    add_ns3_arguments(ap)
    args = ap.parse_args()

    runner = Ns3Runner.from_args(args)
    cfg = load_config(args.config, ["sim.device=cpu"])
    grid = Grid(cfg.grid, cfg.net, device="cpu")
    horizons = [int(h) for h in args.horizons.split(",") if h.strip()]
    conditions = json.loads(Path(args.conditions).read_text(encoding="utf-8"))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    splits = {str(c.get("split", args.split_name)) for c in conditions}
    training = "train" in splits
    manifest = {"status": "running",
                "purpose": "training" if training else "external_validation",
                "simulator": "ns-3.42",
                "training_allowed": training,
                "evaluation_allowed": not training,
                "split_name": args.split_name, "splits": sorted(splits),
                "scenario": args.scenario + ".cc",
                "n_conditions": len(conditions), "replications": args.replications,
                "samples": args.samples, "warmup_s": args.warmup,
                "horizons": horizons, "shards": []}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                       encoding="utf-8")

    for index, condition in enumerate(conditions):
        replications = condition_replications(condition, args.replications)
        split_name = str(condition.get("split", args.split_name))
        trajectories: list[np.ndarray] = []
        with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
            futures = [pool.submit(run_one, runner, condition, seed + 1, args.samples,
                                   args.warmup, args.timeout_s, args.scenario)
                       for seed in range(replications)]
            for future in concurrent.futures.as_completed(futures):
                value = future.result()
                if value is not None:
                    trajectories.append(value)
        if len(trajectories) < 4:
            print(json.dumps({"event": "condition_failed", "index": index,
                              "runs": len(trajectories)}), flush=True)
            continue

        cells = np.stack([digitise(t, grid) for t in trajectories])
        half = len(cells) // 2
        payload: dict[str, np.ndarray] = {}
        for h in horizons:
            target, target_count = kernel_from_cells(cells[:half], h, grid.M)
            reference, reference_count = kernel_from_cells(cells[half:], h, grid.M)
            payload[f"target_h{h}"] = target[None]
            payload[f"count_h{h}"] = target_count[None]
            payload[f"ref_h{h}"] = reference[None]
            payload[f"count_ref_h{h}"] = reference_count[None]
            # The split-half agreement is the measurement noise floor: no model
            # can be scored above it, and reporting it stops a low score from
            # being read as model error when it is sampling error.
            valid = reference_count >= cfg.data.c_min
            a = target.astype(np.float64)[:, valid]
            b = reference.astype(np.float64)[:, valid]
            fidelity = np.zeros(grid.M, dtype=np.float32)
            tv = np.zeros(grid.M, dtype=np.float32)
            fidelity[valid] = (np.sqrt(a * b).sum(axis=0)) ** 2
            tv[valid] = 0.5 * np.abs(a - b).sum(axis=0)
            payload[f"noise_fidelity_h{h}"] = fidelity[None]
            payload[f"noise_tv_h{h}"] = tv[None]
        payload["u"] = np.array([[condition["p_max"], condition["min_th"]]],
                                dtype=np.float32)
        payload["xi"] = np.array([[condition["capacity_pps"],
                                   condition["rtt_prop_s"],
                                   float(condition["n_flows"]),
                                   condition["p_err"]]], dtype=np.float32)
        payload["split"] = np.array([split_name])

        # Every varying coordinate has to be in the name. Leaving the RED
        # action out collapsed three actions onto one filename and silently
        # kept only the last, losing two thirds of a dataset; leaving p_err
        # out did the same the first time a design varied it, and leaving the
        # scenario-specific flags out did it a third time, when the two burst
        # levels of the shift dataset shared every (u, xi) coordinate. The
        # guard below is what caught all three, so it stays even though the
        # name is now complete.
        extra = condition.get("extra", {})
        suffix = "".join(
            f"_{k}{v}" for k, v in sorted(extra.items())
        ).replace(".", "p")
        name = (f"shard_C{int(condition['capacity_pps'])}"
                f"_R{int(condition['rtt_prop_s'] * 1000)}"
                f"_N{condition['n_flows']}"
                f"_E{int(round(condition['p_err'] * 1e6)):05d}"
                f"_P{int(round(condition['p_max'] * 1000)):04d}"
                f"_T{int(round(condition['min_th']))}{suffix}.npz")
        if (out / name).exists():
            raise RuntimeError(
                f"{name} already exists; two conditions map to one filename "
                f"and the dataset would silently lose one of them")
        np.savez_compressed(out / name, **payload)
        manifest["shards"].append({"file": name, "runs": len(trajectories),
                                   "split": split_name, **condition})
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                           encoding="utf-8")
        print(json.dumps({"event": "condition_done", "index": index,
                          "file": name, "runs": len(trajectories),
                          "median_count_h1": int(np.median(
                              payload["count_ref_h1"]))}), flush=True)

    manifest["status"] = "complete"
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                       encoding="utf-8")
    print(json.dumps({"event": "complete", "shards": len(manifest["shards"])}))


if __name__ == "__main__":
    main()
