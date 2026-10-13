"""Row sums of the trained transport and the learned reset intensities.

The transport is column stochastic by construction; its row sums say how much
mass a destination cell can collect in one step, which bounds how far the
operator can concentrate probability, and the reset intensities say how much
mass the loss and timeout events move. Both are measured on the held-out
conditions for every seed of a training run. This feeds the row-sum bound
and the reset-intensity figures quoted in the mechanism discussion.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from qtro.checkpoint import load_checkpoint
from qtro.data import SweepData
from qtro.device import resolve_device
from qtro.eval.panel import resolve_checkpoint


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cw = np.cumsum(w)
    return float(v[np.searchsorted(cw, q * cw[-1])])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="training run directory (train_*.json)")
    ap.add_argument("--panel", required=True, help="dataset directory")
    ap.add_argument("--split", default="ns3_heldout")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", required=True,
                    help="directory for summary.json and per_condition.json")
    args = ap.parse_args()
    device = resolve_device(args.device)

    panel = SweepData(args.panel, horizons=[1], splits=(args.split,),
                      want_refs=False, require_targets=True)
    rows = []
    for manifest in sorted(Path(args.run).glob("train_*.json")):
        for record in json.loads(manifest.read_text(encoding="utf-8")):
            model, grid, cfg, _ = load_checkpoint(resolve_checkpoint(args.run, record), device)
            model = model.double()
            with torch.no_grad():
                for n in range(len(panel)):
                    u = panel.u[n:n + 1].double().to(device)
                    xi = panel.xi[n:n + 1].double().to(device)
                    cond = model._cond(u, xi)
                    transport = model.transport(cond)[0]
                    gamma = model.reset.intensities(cond)[0]
                    reset = model.reset(cond)[0]
                    count = panel.counts[1][n].double().to(device)
                    visited = count >= cfg.data.c_min
                    row_full = transport.sum(1)
                    top = torch.sort(row_full, descending=True).values.cumsum(0)
                    k = torch.arange(1, grid.M + 1, device=device, dtype=top.dtype)
                    # Any k rows of a Stinespring transport with d_env Kraus
                    # terms collect at most min(k * d_env, M) mass between them.
                    bound = torch.minimum(k * model.transport.d_env,
                                          k.new_tensor(float(grid.M)))
                    rows.append({
                        "seed": record["seed"], "condition": n,
                        "col_sum_err": float((transport.sum(0) - 1).abs().max()),
                        "g_max_full": float(row_full.max()),
                        "g_max_visited": float(transport[:, visited].sum(1).max()),
                        "n_visited": int(visited.sum()),
                        "subset_bound_slack": float((top - bound).max()),
                        "gR_max": float(reset.sum(1).max()),
                        "gamma_loss": gamma[:, 0].cpu().numpy().tolist(),
                        "gamma_timeout": gamma[:, 1].cpu().numpy().tolist(),
                        "count": count.cpu().numpy().tolist(),
                    })
            del model

    gmax_full = np.array([r["g_max_full"] for r in rows])
    gmax_vis = np.array([r["g_max_visited"] for r in rows])
    gt = np.concatenate([r["gamma_timeout"] for r in rows])
    gl = np.concatenate([r["gamma_loss"] for r in rows])
    w = np.concatenate([r["count"] for r in rows])
    vis = w >= 32
    summary = {
        "n_rows": len(rows),
        "col_sum_err_max": max(r["col_sum_err"] for r in rows),
        "subset_bound_slack_max": max(r["subset_bound_slack"] for r in rows),
        "g_max_full": {"max": float(gmax_full.max()),
                       "p99": float(np.quantile(gmax_full, 0.99)),
                       "median": float(np.median(gmax_full))},
        "g_max_visited": {"max": float(gmax_vis.max()),
                          "p99": float(np.quantile(gmax_vis, 0.99)),
                          "median": float(np.median(gmax_vis))},
        "gR_max": float(max(r["gR_max"] for r in rows)),
        "gamma_timeout": {"occupancy_p50": weighted_quantile(gt, w, 0.50),
                          "occupancy_p99": weighted_quantile(gt, w, 0.99),
                          "visited_max": float(gt[vis].max()),
                          "all_cells_max": float(gt.max()),
                          "all_cells_p99": float(np.quantile(gt, 0.99))},
        "gamma_loss": {"occupancy_p50": weighted_quantile(gl, w, 0.50),
                       "occupancy_p99": weighted_quantile(gl, w, 0.99),
                       "visited_max": float(gl[vis].max()),
                       "all_cells_max": float(gl.max())},
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    slim = [{k: v for k, v in r.items() if not isinstance(v, list)} for r in rows]
    (out / "per_condition.json").write_text(json.dumps(slim, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
