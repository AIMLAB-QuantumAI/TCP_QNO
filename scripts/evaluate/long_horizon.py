"""What the structural guarantee buys once the rollout is long.

The structural-validity table checks structure after a single application.
The claim actually made is stronger: every step of a recursive rollout
returns a probability law. Those differ, because an unconstrained kernel's
violation compounds under composition while a constrained one's cannot. This
measures both sides of that at horizons far past the ones the models trained
on, and feeds the long-horizon table.

Two questions, kept separate:

* validity   how min element, column mass, and the derived communication
             observables behave as h grows to 1024, in float64;
* invariance whether the kernel preserves the occupancy ns-3 actually spent
             its time in. This is the long-time statistic a dissipative-operator
             paper reports, and it needs no reference kernel at horizon h,
             because the dataset's own visitation counts supply the target.

Invariance is measured as drift rather than as a stationary law. Pushing the
observed occupancy forward, TV(P^h pihat, pihat), is well posed at every h and
needs no mixing assumption; power-iterating to a stationary vector does not
converge on these chains within 4096 steps and would report an unconverged
vector as if it were the invariant measure. The stationary comparison is kept
alongside, with its residual, as a secondary reading.

The measured ns-3 kernel is carried through as the floor. It is estimated from
the same trajectories, so the drift it shows is estimation error and finite
sampling rather than model error.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

from qtro.checkpoint import load_checkpoint
from qtro.data import SweepData
from qtro.device import resolve_device
from qtro.eval.metrics import normalised_axis_w1
from qtro.eval.panel import parse_model_specs

HORIZONS = (1, 5, 20, 50, 100, 200, 500, 1024)


def observables(P: torch.Tensor, q_bins: torch.Tensor, tail: torch.Tensor):
    """Structural and communication readouts of one batch of kernels.

    Every quantity is taken over columns, because a column is what a
    controller pushes a distribution through.
    """
    finite = torch.isfinite(P)
    if not bool(finite.all()):
        return {"finite": False,
                "nonfinite_fraction": float((~finite).float().mean())}
    colsum = P.sum(dim=1)
    mean_q = torch.einsum("bij,i->bj", P, q_bins)
    tail_p = torch.einsum("bij,i->bj", P, tail)
    return {
        "finite": True,
        "min_element": float(P.min()),
        "max_colsum_error": float((colsum - 1.0).abs().max()),
        "mean_queue_min": float(mean_q.min()),
        "mean_queue_max": float(mean_q.max()),
        "tail_prob_min": float(tail_p.min()),
        "tail_prob_max": float(tail_p.max()),
    }


def empirical_kernel(_model, b) -> torch.Tensor:
    """The measured ns-3 one-step kernel, completed to a stochastic matrix.

    Columns whose source cell ns-3 never left carry no measurement. Holding
    such a cell in place is the neutral completion: it adds no dynamics the
    data did not show, and it keeps the matrix stochastic so the same
    composition and drift measurements apply.
    """
    P = b["refs"][1].double().clone()
    mass = P.sum(dim=1, keepdim=True)
    empty = (mass <= 1e-12).squeeze(1)
    P = torch.where(mass > 1e-12, P / mass.clamp_min(1e-30), P)
    if bool(empty.any()):
        eye = torch.eye(P.shape[1], dtype=P.dtype, device=P.device)
        P[:, :, :] = torch.where(
            empty.unsqueeze(1), eye.unsqueeze(0).expand_as(P), P)
    return P


def stationary(P: torch.Tensor, iterations: int = 4096, tol: float = 1e-13):
    """Power-iterate a column-stochastic kernel to its invariant law.

    Returns (pi, converged_flag, residual, iterations). A kernel that is not
    stochastic has no such law; the caller checks that before asking.
    """
    B, M, _ = P.shape
    pi = torch.full((B, M), 1.0 / M, dtype=P.dtype, device=P.device)
    residual = float("inf")
    for step in range(iterations):
        nxt = torch.bmm(P, pi.unsqueeze(-1)).squeeze(-1)
        nxt = nxt.clamp_min(0)
        nxt = nxt / nxt.sum(dim=1, keepdim=True).clamp_min(1e-300)
        residual = float((nxt - pi).abs().sum(dim=1).max())
        pi = nxt
        if residual < tol:
            return pi, True, residual, step + 1
    return pi, False, residual, iterations


def compare(pi: torch.Tensor, ref: torch.Tensor, grid):
    """Fidelity, total variation, and normalised queue W1 between two laws.

    The W1 reuses the axis metric by presenting each distribution as a
    one-column kernel, so the number is on the same scale as the accuracy
    tables.
    """
    fid = (pi.clamp_min(0).sqrt() * ref.clamp_min(0).sqrt()).sum(dim=1) ** 2
    tv = 0.5 * (pi - ref).abs().sum(dim=1)
    _, q_bin, _ = grid.axis_bins()
    q_centres = 0.5 * (grid.q_edges[:-1] + grid.q_edges[1:])
    w1 = normalised_axis_w1(pi.unsqueeze(-1), ref.unsqueeze(-1),
                            q_bin, q_centres.to(pi)).squeeze(-1)
    return {"fidelity": fid, "tv": tv, "queue_w1": w1}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True, help="dataset directory")
    ap.add_argument("--split", required=True, help="split name inside the dataset")
    ap.add_argument("--models", required=True,
                    help="comma-separated name=run-directory entries")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0, help="checkpoint seed to use")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = resolve_device(args.device)
    panel = SweepData(args.panel, horizons=[1], splits=(args.split,),
                      want_refs=True, require_targets=False)
    n = len(panel)
    print(f"{n} conditions in {args.split}")

    specs = []
    for name, run_dir in parse_model_specs(args.models).items():
        cks = [p for p in sorted(Path(run_dir).glob("checkpoints/*.pt"))
               if re.search(rf"seed{args.seed}$", p.stem)]
        if not cks:
            cks = sorted(Path(run_dir).glob("checkpoints/*.pt"))[:1]
        specs.append((name, cks[0]))

    # Channels are held as (name, callable producing a float64 kernel).
    report = {"panel": args.panel, "split": args.split, "conditions": n,
              "horizons": list(HORIZONS), "seed": args.seed, "channels": {}}

    grid = None
    for name, ck in specs:
        model, grid, mcfg, _ = load_checkpoint(ck, device)
        # float64 throughout, as the structural checks are. Casting a float32
        # output up instead would report float32 rounding as structural drift:
        # QTRO's column sums leave the channel at ~1e-6 in float32 and at
        # ~1e-15 in float64, and its smallest entries underflow to exactly
        # zero in float32 while staying strictly positive in float64.
        model = model.double().eval()
        variants = {name: lambda m, b: m.propagator(b["u"].double(),
                                                    b["xi"].double())}
        if hasattr(model, "raw_propagator"):
            variants[name + "_raw"] = lambda m, b: m.raw_propagator(
                b["u"].double(), b["xi"].double())
        if not report["channels"]:
            # The measured kernel, carried once as the floor. A source cell
            # ns-3 never left has no measured column; holding it in place is
            # the neutral completion and is counted below.
            variants = {"ns3_empirical": empirical_kernel, **variants}

        for label, fn in variants.items():
            per_h = {h: [] for h in HORIZONS}
            drift = {h: [] for h in HORIZONS}
            stat_rows = []
            with torch.no_grad():
                for s in range(0, n, args.batch):
                    idx = torch.arange(s, min(s + args.batch, n))
                    b = panel.batch(idx, device)
                    q_bins = grid.q_rep.to(device).double()
                    tail = grid.tail_mask.to(device).double()

                    P = fn(model, b)
                    # Repeated squaring keeps every reported power exact.
                    cache = {1: P}
                    for h in HORIZONS:
                        if h in cache:
                            continue
                        bits, base = h, P
                        acc = torch.eye(P.shape[1], dtype=P.dtype,
                                        device=P.device).expand_as(P).clone()
                        while bits:
                            if bits & 1:
                                acc = torch.bmm(base, acc)
                            bits >>= 1
                            if bits:
                                base = torch.bmm(base, base)
                        cache[h] = acc
                    for h in HORIZONS:
                        per_h[h].append(observables(cache[h], q_bins, tail))

                    # Invariance: does the kernel leave the occupancy alone?
                    counts0 = b["ref_counts"][1].double()
                    occupancy = counts0 / counts0.sum(dim=1, keepdim=True).clamp_min(1)
                    ref_mean_q = (occupancy * q_bins).sum(dim=1)
                    for h in HORIZONS:
                        Ph = cache[h]
                        if not bool(torch.isfinite(Ph).all()):
                            drift[h].extend([None] * len(idx))
                            continue
                        pushed = torch.bmm(
                            Ph, occupancy.unsqueeze(-1)).squeeze(-1)
                        tv_h = 0.5 * (pushed - occupancy).abs().sum(dim=1)
                        mq_h = (pushed * q_bins).sum(dim=1)
                        for j in range(len(idx)):
                            drift[h].append({
                                "tv": float(tv_h[j]),
                                "mean_queue_error": float(
                                    (mq_h[j] - ref_mean_q[j]).abs()),
                            })

                    colsum = P.sum(dim=1)
                    ok = bool(torch.isfinite(P).all()) and float(
                        (colsum - 1.0).abs().max()) < 1e-6 and float(P.min()) > -1e-12
                    if ok:
                        pi, conv, res, steps = stationary(P.clamp_min(0))
                        counts = b["ref_counts"][1].double()
                        ref = counts / counts.sum(dim=1, keepdim=True).clamp_min(1)
                        m = compare(pi, ref, grid)
                        for j in range(len(idx)):
                            stat_rows.append({
                                "converged": bool(conv), "residual": res,
                                "iterations": steps,
                                "fidelity": float(m["fidelity"][j]),
                                "tv": float(m["tv"][j]),
                                "queue_w1": float(m["queue_w1"][j]),
                                "mean_queue": float((pi[j] * q_bins).sum()),
                                "ref_mean_queue": float((ref[j] * q_bins).sum()),
                            })
                    del cache, P
                    if device.startswith("cuda"):
                        torch.cuda.empty_cache()

            channel = {"checkpoint": str(ck), "validity": {}, "drift": {},
                       "stationary": {}}
            for h in HORIZONS:
                rows = [r for r in drift[h] if r is not None]
                channel["drift"][str(h)] = {
                    "conditions": len(rows),
                    "tv": float(np.mean([r["tv"] for r in rows])) if rows else None,
                    "mean_queue_error": float(np.mean(
                        [r["mean_queue_error"] for r in rows])) if rows else None,
                }
            for h in HORIZONS:
                rows = per_h[h]
                if not all(r["finite"] for r in rows):
                    channel["validity"][str(h)] = {
                        "finite": False,
                        "nonfinite_fraction": float(np.mean(
                            [r.get("nonfinite_fraction", 1.0) for r in rows])),
                    }
                    continue
                channel["validity"][str(h)] = {
                    "finite": True,
                    "min_element": min(r["min_element"] for r in rows),
                    "max_colsum_error": max(r["max_colsum_error"] for r in rows),
                    "mean_queue_min": min(r["mean_queue_min"] for r in rows),
                    "mean_queue_max": max(r["mean_queue_max"] for r in rows),
                    "tail_prob_min": min(r["tail_prob_min"] for r in rows),
                    "tail_prob_max": max(r["tail_prob_max"] for r in rows),
                }
            if stat_rows:
                channel["stationary"] = {
                    "conditions": len(stat_rows),
                    "converged": sum(r["converged"] for r in stat_rows),
                    "median_residual": float(np.median(
                        [r["residual"] for r in stat_rows])),
                    "fidelity": float(np.mean([r["fidelity"] for r in stat_rows])),
                    "tv": float(np.mean([r["tv"] for r in stat_rows])),
                    "queue_w1": float(np.mean([r["queue_w1"] for r in stat_rows])),
                    "mean_queue_abs_error": float(np.mean(
                        [abs(r["mean_queue"] - r["ref_mean_queue"])
                         for r in stat_rows])),
                    "per_condition": stat_rows,
                }
            else:
                channel["stationary"] = {"applicable": False,
                                         "reason": "kernel is not stochastic"}
            report["channels"][label] = channel

            v = channel["validity"]
            print(f"\n=== {label} ===")
            print(f"{'h':>6}{'min elem':>14}{'|colsum-1|':>13}"
                  f"{'mean queue':>22}{'P(tail)':>20}"
                  f"{'drift TV / pkts':>18}")
            for h in HORIZONS:
                r = v[str(h)]
                if not r["finite"]:
                    print(f"{h:>6}   non-finite in "
                          f"{r['nonfinite_fraction'] * 100:.1f}% of entries")
                    continue
                queue_range = "[%.2f, %.2f]" % (r["mean_queue_min"],
                                                r["mean_queue_max"])
                tail_range = "[%.3f, %.3f]" % (r["tail_prob_min"],
                                               r["tail_prob_max"])
                d = channel["drift"][str(h)]
                drift_text = ("--" if d["tv"] is None
                              else "%.4f / %5.2f" % (d["tv"],
                                                     d["mean_queue_error"]))
                print(f"{h:>6}{r['min_element']:>14.2e}"
                      f"{r['max_colsum_error']:>13.2e}"
                      f"{queue_range:>22}{tail_range:>20}{drift_text:>18}")
            st = channel["stationary"]
            if st.get("applicable", True):
                print(f"  stationary vs ns-3 occupancy: fid {st['fidelity']:.4f}  "
                      f"TV {st['tv']:.4f}  W1 {st['queue_w1']:.4f}  "
                      f"mean-queue err {st['mean_queue_abs_error']:.2f} pkts  "
                      f"({st['converged']}/{st['conditions']} converged, "
                      f"median residual {st['median_residual']:.2e})")
            else:
                print("  stationary law: not applicable, kernel is not stochastic")

        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
