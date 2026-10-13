"""Roll every model through the ns-3 load step and score the distribution.

Takes the measured ns-3 ensemble at the sample before the step, propagates
each model forward one snapshot at a time while switching the session count
at the step, and compares the predicted state distribution against the
measured one at every sample. This feeds the load-step table.

Two design choices, neither cosmetic:

* The learned models are rolled by repeated application of their one-step
  kernel, which is what they are. The analytic models are solved directly to
  each time instead, which is what the fluid and diffusion literature does:
  composing a 10 ms binned map hides motion the ODE has.
* The analytic models are anchored at the fluid equilibrium rather than at
  flow balance. Balanced anchoring assumes the aggregate exactly fills the
  link at the observed queue, and a load step is precisely the moment when
  that is false; with it the fluid rollout holds the pre-step mean queue
  fixed, which is the anchoring speaking rather than a prediction.

Two reference rows make the numbers readable. Persistence freezes the
pre-step distribution and says how much of the score is available for free.
The split-half row compares one half of the ns-3 replications against the
other and says how much of the remaining error is the ensemble's own sampling
noise rather than model error; it is a diagnostic, not a lower bound, since a
smooth predictor can beat a noisy half.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from qtro.baselines.fluid import FluidPropagator
from qtro.checkpoint import load_checkpoint
from qtro.config import load_config
from qtro.device import resolve_device
from qtro.eval.panel import parse_model_specs
from qtro.grid import Grid


def empirical_distributions(cells: np.ndarray, M: int) -> np.ndarray:
    """``(time, M)`` occupancy of the replication ensemble at each sample."""
    out = np.zeros((cells.shape[1], M), dtype=np.float64)
    for t in range(cells.shape[1]):
        counts = np.bincount(cells[:, t], minlength=M)
        out[t] = counts / max(counts.sum(), 1)
    return out


def condition_tensors(condition: dict, n_flows: float, device: str):
    u = torch.tensor([[condition["p_max"], condition["min_th"]]],
                     dtype=torch.float32, device=device)
    xi = torch.tensor([[condition["capacity_pps"], condition["rtt_prop_s"],
                        float(n_flows), condition["p_err"]]],
                      dtype=torch.float32, device=device)
    return u, xi


def rollout(kernels: dict[int, np.ndarray], sessions: np.ndarray,
            start: np.ndarray, n_steps: int) -> np.ndarray:
    """Propagate ``start`` for ``n_steps``, switching kernel with the load."""
    out = np.zeros((n_steps + 1, start.size), dtype=np.float64)
    out[0] = start
    current = start
    for k in range(n_steps):
        matrix = kernels[int(sessions[k])]
        current = matrix @ current
        total = current.sum()
        if total > 0:
            current = current / total
        out[k + 1] = current
    return out


def score(predicted: np.ndarray, reference: np.ndarray, q: np.ndarray,
          tail: np.ndarray, dt: float) -> dict:
    """Per-sample error, integrated over the response."""
    tv = 0.5 * np.abs(predicted - reference).sum(axis=1)
    fidelity = np.square(np.sqrt(predicted * reference).sum(axis=1))
    mean_q_error = np.abs(predicted @ q - reference @ q)
    tail_error = np.abs(predicted @ tail - reference @ tail)
    return {
        "tv_auc": float(tv.sum() * dt),
        "mean_queue_error_auc": float(mean_q_error.sum() * dt),
        "high_queue_error_auc": float(tail_error.sum() * dt),
        "mean_fidelity": float(fidelity.mean()),
        "tv": tv.tolist(),
        "mean_queue_error": mean_q_error.tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--transient", required=True,
                    help="directory written by scripts/data/generate_load_steps.py")
    ap.add_argument("--config", default=None,
                    help="YAML config (default: the built-in values)")
    ap.add_argument("--models", required=True,
                    help="comma-separated name=run-directory entries, learned")
    ap.add_argument("--analytic", default="fluid_jump,diffusion")
    ap.add_argument("--horizon-samples", type=int, default=100,
                    help="samples scored after the step")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = resolve_device(args.device)
    root = Path(args.transient)
    report_in = json.loads((root / "results.json").read_text(encoding="utf-8"))
    cfg = load_config(args.config, [f"sim.device={device}"])
    grid = Grid(cfg.grid, cfg.net, device=device)
    q = grid.q_rep.detach().cpu().numpy().astype(np.float64)
    tail = grid.tail_mask.detach().cpu().numpy().astype(np.float64)

    checkpoints: dict[str, list[Path]] = {}
    for name, path in parse_model_specs(args.models).items():
        found = sorted(Path(path).glob("checkpoints/*.pt"))
        if not found:
            raise SystemExit(f"no checkpoints under {path}")
        checkpoints[name] = found
    analytic = [m.strip() for m in args.analytic.split(",") if m.strip()]

    results: dict[str, dict[str, list[dict]]] = {}
    for condition in report_in["conditions"]:
        name = condition["name"]
        payload = np.load(root / condition["cells_file"])
        cells = payload["cells"].astype(np.int64)
        sessions = payload["n_active"][0].astype(np.int64)
        reference = empirical_distributions(cells, grid.M)
        step = condition["step_index"]
        n_steps = min(args.horizon_samples, reference.shape[0] - step - 1)
        target = reference[step:step + n_steps + 1]
        start = reference[step]
        dt = condition["sample_interval_s"]
        loads = sorted(set(int(v) for v in sessions[step:step + n_steps]))
        print(f"{name}: step at {step}, {n_steps} samples, loads {loads}",
              flush=True)

        entry: dict[str, list[dict]] = {}
        # Persistence: the pre-step distribution held fixed.
        held = np.repeat(start[None], n_steps + 1, axis=0)
        entry["persistence"] = [score(held, target, q, tail, dt)]

        # Split-half of the ns-3 ensemble itself.
        half = cells.shape[0] // 2
        a = empirical_distributions(cells[:half], grid.M)[step:step + n_steps + 1]
        b = empirical_distributions(cells[half:], grid.M)[step:step + n_steps + 1]
        entry["split_half"] = [score(a, b, q, tail, dt)]

        for label, paths in checkpoints.items():
            rows = []
            for path in paths:
                model, _, _, _ = load_checkpoint(path, device)
                with torch.no_grad():
                    kernels = {}
                    for load in loads:
                        u, xi = condition_tensors(condition, load, device)
                        kernels[load] = model.propagator(
                            u, xi)[0].detach().cpu().numpy().astype(np.float64)
                rows.append(score(rollout(kernels, sessions[step:], start,
                                          n_steps),
                                  target, q, tail, dt))
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            entry[label] = rows

        if len(loads) != 1:
            raise SystemExit(
                f"{name}: the load changes inside the scored window "
                f"({loads}); the analytic solve-to-t path assumes one load")
        cpu_grid = Grid(cfg.grid, cfg.net, device="cpu")
        for mode in analytic:
            # Solved directly to each instant, on the CPU. Chunked because
            # 100 horizons of a 512x512 kernel is 105 MB held at once.
            propagator = FluidPropagator(cpu_grid, cfg, mode,
                                         anchor="equilibrium")
            u, xi = condition_tensors(condition, loads[0], "cpu")
            predicted = np.zeros((n_steps + 1, grid.M), dtype=np.float64)
            predicted[0] = start
            for begin in range(1, n_steps + 1, 25):
                wanted = range(begin, min(begin + 25, n_steps + 1))
                solved = propagator.kernels(u, xi, tuple(wanted))
                for k in wanted:
                    column = solved[k][0].detach().numpy().astype(np.float64)
                    predicted[k] = column @ start
            entry[mode] = [score(predicted, target, q, tail, dt)]
        results[name] = entry

    order = (["persistence"] + list(checkpoints) + analytic + ["split_half"])
    print(f"\nintegrated over {args.horizon_samples} samples after the step; "
          f"lower is better except fidelity")
    header = f"{'condition':>26}" + "".join(f"{m:>13}" for m in order)
    for metric in ("tv_auc", "mean_queue_error_auc", "high_queue_error_auc",
                   "mean_fidelity"):
        print(f"\n--- {metric} ---")
        print(header)
        for name, entry in results.items():
            row = f"{name:>26}"
            for model in order:
                values = [r[metric] for r in entry.get(model, [])]
                row += f"{np.mean(values):>13.4f}" if values else f"{'-':>13}"
            print(row)
        best = {}
        for name, entry in results.items():
            scores = {m: float(np.mean([r[metric] for r in entry[m]]))
                      for m in order if m in entry and m != "split_half"}
            pick = (max if metric == "mean_fidelity" else min)
            best[name] = pick(scores, key=scores.get)
        print(f"{'best (excl. split-half)':>26}" + "".join(
            f"{best[n]:>13}" for n in list(results)[:1]) + "  "
            + ", ".join(f"{n.split('_red_')[-1]}:{b}" for n, b in best.items()))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps({"transient": str(root), "results": results},
                       indent=1, default=float), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
