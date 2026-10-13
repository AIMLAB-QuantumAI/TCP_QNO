"""Measure what amortisation buys, against ns-3 and the domain solvers.

A new control or network query is answered by a forward evaluation rather
than a new solver run or a new simulation ensemble. This times that on the
decision the operator is for -- score K candidate RED settings at one
operating point and read a high-queue probability at h=20 -- and reports the
same decision's cost through ns-3 and through the two domain models. It
feeds the cost table.

Three costs are separated because they amortise differently:

* build    constructing P(u, xi) for the candidate, paid once per candidate;
* compose  raising it to the horizon, cacheable with the propagator;
* apply    pushing one distribution through the cached power, which is the
           only part paid again when the state changes inside an interval.

The ns-3 column is measured rather than estimated: the same binary and the
same replication count that produced the dataset.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from qtro.baselines.fluid import FluidPropagator
from qtro.checkpoint import load_checkpoint
from qtro.config import load_config
from qtro.data import SweepData
from qtro.device import resolve_device
from qtro.eval.panel import parse_model_specs
from qtro.losses import propagator_powers
from qtro.ns3 import Ns3Runner, add_ns3_arguments, condition_flags, csv_rows


def elapsed(fn, device: str, repeats: int) -> tuple[float, float]:
    """Return (median, min) seconds over `repeats` timed calls."""
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples), min(samples)


@torch.no_grad()
def time_model(model, u, xi, p0, horizon, device, repeats):
    """Time build, compose, and apply for one learned propagator."""
    for _ in range(5):                                  # warm the allocator
        propagator_powers(model.propagator(u, xi), [horizon])

    build_med, build_min = elapsed(
        lambda: model.propagator(u, xi), device, repeats)
    P = model.propagator(u, xi)
    compose_med, compose_min = elapsed(
        lambda: propagator_powers(P, [horizon]), device, repeats)
    Ph = propagator_powers(P, [horizon])[horizon]
    apply_med, apply_min = elapsed(
        lambda: torch.bmm(Ph, p0.unsqueeze(-1)).squeeze(-1), device, repeats)
    return {"build_s": build_med, "build_min_s": build_min,
            "compose_s": compose_med, "compose_min_s": compose_min,
            "apply_s": apply_med, "apply_min_s": apply_min,
            "total_s": build_med + compose_med + apply_med}


@torch.no_grad()
def time_domain(grid, cfg, mode, anchor, u, xi, horizon, device, repeats):
    """Time a domain solver.

    It has no build/apply split: every horizon is a fresh numerical solution,
    which is precisely the property the amortisation claim contrasts with.
    """
    prop = FluidPropagator(grid, cfg, mode=mode, anchor=anchor)
    for _ in range(2):
        prop.kernels(u, xi, [horizon])
    med, lo = elapsed(lambda: prop.kernels(u, xi, [horizon]), device, repeats)
    return {"total_s": med, "min_s": lo}


@torch.no_grad()
def scan_candidates(model, grid, cfg, u1, xi1, counts1, horizon, device,
                    sizes, repeats):
    """Cost of one decision against the number of candidates it compares.

    Amortisation is a claim about this slope, not about a single point. A
    fixed-overhead-dominated build means the marginal candidate is nearly
    free, which is the property multi-candidate AQM comparison needs.
    """
    rows = []
    for K in sizes:
        u = u1.expand(K, -1).contiguous()
        xi = xi1.expand(K, -1).contiguous()
        counts = counts1.expand(K, -1)
        p0 = (counts / counts.sum(dim=1, keepdim=True).clamp(min=1)).to(u)
        row = {"candidates": K}
        row.update(time_model(model, u, xi, p0, horizon, device, repeats))
        for label, (mode, anchor) in {
                "fluid": ("fluid", "balanced"),
                "diffusion": ("diffusion", "balanced")}.items():
            row[label + "_s"] = time_domain(
                grid, cfg, mode, anchor, u, xi, horizon, device,
                max(3, repeats // 10))["total_s"]
        rows.append(row)
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return rows


def time_ns3(runner: Ns3Runner, condition: dict, samples: int, warmup: float,
             replications: int, timeout_s: int) -> list[float]:
    """Time real ns-3 runs for one condition, serially."""
    walls = []
    for seed in range(1, replications + 1):
        flags = (f"{condition_flags(condition)} --samples={samples} "
                 f"--warmup={warmup} --runSeed={seed}")
        t0 = time.perf_counter()
        stdout = runner.run(runner.wrapper_argv("qtro-dumbbell", flags), timeout_s)
        wall = time.perf_counter() - t0
        rows = csv_rows(stdout or "")
        if len(rows) < samples + 1:
            raise SystemExit(
                f"ns-3 run seed {seed} returned {len(rows)} rows, want {samples + 1}")
        walls.append(wall)
    return walls


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True, help="dataset directory")
    ap.add_argument("--split", required=True, help="split name inside the dataset")
    ap.add_argument("--config", default=None,
                    help="YAML config (default: the built-in values)")
    ap.add_argument("--models", required=True,
                    help="comma-separated name=run-directory entries")
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--candidates", type=int, default=20,
                    help="RED settings scored per decision; 20 is the crossed grid")
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--devices", default="auto",
                    help="comma-separated devices to time, e.g. cuda,cpu")
    ap.add_argument("--ns3-replications", type=int, default=8,
                    help="timed ns-3 runs; the dataset itself used 64 or 128")
    ap.add_argument("--panel-replications", type=int, default=64)
    ap.add_argument("--ns3-samples", type=int, default=0,
                    help="0 takes the dataset manifest's own value")
    ap.add_argument("--ns3-warmup", type=float, default=0.0,
                    help="0 takes the dataset manifest's own value")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--scan", default="",
                    help="comma-separated candidate counts, e.g. 1,5,20,100")
    ap.add_argument("--skip-ns3", action="store_true")
    ap.add_argument("--out", required=True)
    add_ns3_arguments(ap)
    args = ap.parse_args()

    models = parse_model_specs(args.models)
    report = {"horizon": args.horizon, "candidates": args.candidates,
              "repeats": args.repeats, "devices": {}}

    for device in [resolve_device(d.strip()) for d in args.devices.split(",") if d.strip()]:
        if device.startswith("cuda") and not torch.cuda.is_available():
            print(f"skip {device}: unavailable")
            continue
        cfg = load_config(args.config, [f"sim.device={device}"])
        panel = SweepData(args.panel, horizons=[1], splits=(args.split,),
                          want_refs=True, require_targets=False)
        batch = panel.batch(torch.arange(1), device)

        # One operating point, K candidate controls: the MPC query.
        K = args.candidates
        u = batch["u"].expand(K, -1).contiguous()
        xi = batch["xi"].expand(K, -1).contiguous()
        counts = batch["ref_counts"][1].expand(K, -1)
        p0 = (counts / counts.sum(dim=1, keepdim=True).clamp(min=1)).to(u)

        entry = {"models": {}, "domain": {}}
        grid = None
        for name, run_dir in models.items():
            ck = sorted(Path(run_dir).glob("checkpoints/*.pt"))[0]
            model, grid, _, payload = load_checkpoint(ck, device)
            model.eval()
            entry["models"][name] = time_model(
                model, u, xi, p0, args.horizon, device, args.repeats)
            entry["models"][name]["parameters"] = payload["metadata"].get(
                "parameter_count")
            del model
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

        for label, (mode, anchor) in {
                "fluid": ("fluid", "balanced"),
                "diffusion": ("diffusion", "balanced")}.items():
            entry["domain"][label] = time_domain(
                grid, cfg, mode, anchor, u, xi, args.horizon, device,
                max(4, args.repeats // 10))

        if args.scan:
            sizes = [int(x) for x in args.scan.split(",")]
            first = next(iter(models))
            ck = sorted(Path(models[first]).glob("checkpoints/*.pt"))[0]
            model, grid, _, _ = load_checkpoint(ck, device)
            model.eval()
            entry["scan"] = scan_candidates(
                model, grid, cfg, batch["u"], batch["xi"],
                batch["ref_counts"][1], args.horizon, device, sizes,
                max(5, args.repeats // 2))
            del model
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

        report["devices"][device] = entry
        print(f"--- {device} ---")
        for name, v in entry["models"].items():
            print(f"  {name:>8} build {v['build_s'] * 1e3:8.3f} ms  "
                  f"compose {v['compose_s'] * 1e3:8.3f} ms  "
                  f"apply {v['apply_s'] * 1e6:8.1f} us  "
                  f"total {v['total_s'] * 1e3:8.3f} ms")
        for name, v in entry["domain"].items():
            print(f"  {name:>8} solve {v['total_s'] * 1e3:8.3f} ms")
        for row in entry.get("scan", []):
            print(f"  K={row['candidates']:>4}  model {row['total_s'] * 1e3:9.2f} ms"
                  f"  ({row['total_s'] / row['candidates'] * 1e3:7.3f} ms/cand)"
                  f"  fluid {row['fluid_s'] * 1e3:9.2f} ms"
                  f"  diffusion {row['diffusion_s'] * 1e3:9.2f} ms")

    if not args.skip_ns3:
        runner = Ns3Runner.from_args(args)
        # The manifest carries the exact arguments each shard was produced
        # with, so timing reproduces the dataset run rather than approximating it.
        manifest = json.loads(
            (Path(args.panel) / "manifest.json").read_text(encoding="utf-8"))
        shards = [s for s in manifest["shards"] if s.get("split") == args.split]
        if not shards:
            raise SystemExit(f"no shard in {args.panel} has split {args.split}")
        cond = shards[0]
        samples = args.ns3_samples or int(manifest["samples"])
        warmup = args.ns3_warmup or float(manifest["warmup_s"])
        replications = int(cond.get("replications", manifest["replications"]))
        walls = time_ns3(runner, cond, samples, warmup,
                         args.ns3_replications, args.timeout)
        per_run = statistics.median(walls)
        report["ns3"] = {
            "condition": cond,
            "samples": samples,
            "warmup_s": warmup,
            "timed_runs": len(walls),
            "per_run_s": per_run,
            "per_run_min_s": min(walls),
            "panel_replications": replications,
            "one_condition_serial_s": per_run * replications,
            "k_candidates_serial_s": per_run * replications * args.candidates,
        }
        print(f"--- ns-3 ---\n  {per_run:.2f} s per run, "
              f"{per_run * replications:.0f} s per condition at "
              f"{replications} replications, "
              f"{per_run * replications * args.candidates / 3600:.2f} h "
              f"for {args.candidates} candidates")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
