"""Run the ns-3 closed loop with a predictor choosing the RED setting.

``qtro-closedloop`` publishes the router state once per control interval and
blocks for a RED setting. At each instant the controller digitises the observed
(window, queue, phase) to a grid cell, pushes that one-hot forward over the
control interval under each of the twenty crossed RED settings, and picks the
setting minimising the squared normalised distance of the predicted mean queue
from the target plus a risk term on the predicted 95th-percentile queue, plus
a small penalty on changing the setting. Every controller receives the same
objective, candidate set and observation. The router also publishes its active
flow count, and by default (``--known-load current``) every controller
conditions on it; ``--known-load initial`` withholds the load change from all
of them.

Baselines in the same loop: ``fixed`` holds the initial setting; ``--with-domain``
adds the fluid and diffusion solvers as controllers.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

from qtro.baselines.fluid import FluidPropagator
from qtro.checkpoint import load_checkpoint
from qtro.config import load_config
from qtro.device import resolve_device
from qtro.eval.panel import parse_model_specs
from qtro.grid import Grid
from qtro.ns3 import Ns3Runner, add_ns3_arguments, condition_flags

P_MAX_GRID = (0.05, 0.1, 0.2, 0.3)
MIN_TH_GRID = (16.0, 20.0, 26.0, 30.0, 34.0)


def candidates() -> list[tuple[float, float]]:
    return [(p, t) for p in P_MAX_GRID for t in MIN_TH_GRID]


class Controller:
    """Chooses a RED setting from a predicted one-interval-ahead distribution."""

    def __init__(self, model, grid, cfg, horizon, q_target, lam, device):
        self.model = model
        self.grid = grid
        self.cfg = cfg
        self.horizon = horizon
        self.q_target = float(q_target)
        self.lam = float(lam)
        self.device = device
        self.q = grid.q_rep.to(device).float()
        self.tail = grid.tail_mask.to(device).float()
        self.buffer = float(cfg.grid.buffer_pkts)
        self.u = torch.tensor([[p, t] for p, t in candidates()],
                              dtype=torch.float32, device=device)
        self._powers = None
        self._xi_key = None
        self.risk = "percentile"
        self.q95_limit = 40.0
        # Move suppression, standard in receding-horizon control. Without it
        # the controllers changed the RED setting in ten of twelve intervals,
        # and each change perturbs the queue the objective is trying to hold
        # steady: they bought about one packet of tracking and paid for it in
        # spread. The penalty is on the normalised distance from the setting
        # currently in force.
        self.lam_move = 0.0
        self.previous = None
        # Queue values sorted once; the percentile reads the predicted law
        # along this order.
        self._q_order = torch.argsort(self.q)

    def _kernel(self, xi_row):
        """Cache the one-step kernel per operating point; xi is fixed in a run.

        Only the one-step kernel is stored. Materialising every power up to the
        horizon would be (H, K, M, M), which is 2 GB at H=100, and it is not
        needed: the cost depends on one column, so pushing a single vector
        forward gives the same numbers at a thousandth of the memory.
        """
        key = tuple(float(v) for v in xi_row)
        if self._powers is not None and self._xi_key == key:
            return self._powers
        xi = torch.tensor([key], dtype=torch.float32,
                          device=self.device).expand(self.u.shape[0], -1)
        with torch.no_grad():
            self._powers = self.model.propagator(self.u, xi.contiguous())
        self._xi_key = key
        return self._powers

    def act(self, cell, xi_row):
        """Average predicted cost over the interval, not its endpoint.

        The realised metric is the queue averaged over the interval, so the
        predicted cost is the same average. Scoring the endpoint alone lets a
        candidate that overshoots and returns look identical to one that sits
        at the target throughout.
        """
        P = self._kernel(xi_row)                        # (K, M, M)
        v = torch.zeros(P.shape[0], P.shape[1], device=self.device,
                        dtype=P.dtype)
        v[:, cell] = 1.0
        mean_q = torch.zeros(P.shape[0], device=self.device, dtype=P.dtype)
        tail_p = torch.zeros_like(mean_q)
        mass = torch.zeros_like(P[:, :, 0])
        with torch.no_grad():
            for _ in range(self.horizon):
                v = torch.bmm(P, v.unsqueeze(-1)).squeeze(-1)
                mean_q += v @ self.q
                tail_p += v @ self.tail
                mass += v
        mean_q /= self.horizon
        tail_p /= self.horizon
        mass /= self.horizon

        if self.risk == "threshold":
            risk = tail_p
        else:
            # A soft constraint on the predicted 95th percentile queue.
            #
            # The threshold form is inert here. q_crit is 80% of the buffer
            # and the queue crosses it in under 1% of samples, so every model
            # predicts a high-queue probability near zero, the risk term
            # contributes nothing, and the objective collapses to tracking.
            # A percentile reads the same predicted distribution at a point
            # the system actually visits, so a model that gets the spread
            # right can be rewarded for it.
            q95 = self._percentile(mass, 0.95)
            risk = torch.clamp(q95 - self.q95_limit, min=0.0) / self.buffer
            risk = risk ** 2

        cost = ((mean_q - self.q_target) / self.buffer) ** 2 + self.lam * risk
        if self.lam_move > 0.0 and self.previous is not None:
            span = self.u.max(dim=0).values - self.u.min(dim=0).values
            move = ((self.u - self.u.new_tensor(self.previous))
                    / span.clamp_min(1e-9)).pow(2).sum(dim=1)
            cost = cost + self.lam_move * move
        best = int(torch.argmin(cost))
        self.last_prediction = (float(mean_q[best]), float(tail_p[best]))
        self.previous = candidates()[best]
        return candidates()[best], float(cost[best])

    def _percentile(self, mass, level):
        """Queue value at a probability level of the predicted law."""
        order = self._q_order
        cdf = mass[:, order].cumsum(dim=1)
        idx = torch.searchsorted(
            cdf.contiguous(),
            torch.full((mass.shape[0], 1), level, device=mass.device,
                       dtype=mass.dtype)).clamp(max=mass.shape[1] - 1)
        return self.q[order][idx.squeeze(1)]


class DomainController(Controller):
    """Same objective, kernels from a fluid or diffusion solver."""

    def __init__(self, mode, anchor, grid, cfg, horizon, q_target, lam, device):
        super().__init__(None, grid, cfg, horizon, q_target, lam, device)
        self.prop = FluidPropagator(grid, cfg, mode=mode, anchor=anchor)

    def _kernel(self, xi_row):
        key = tuple(float(v) for v in xi_row)
        if self._powers is not None and self._xi_key == key:
            return self._powers
        xi = torch.tensor([key], dtype=torch.float32,
                          device=self.device).expand(self.u.shape[0], -1)
        with torch.no_grad():
            self._powers = self.prop.kernels(self.u, xi.contiguous(), [1])[1]
        self._xi_key = key
        return self._powers


class FixedController:
    def __init__(self, setting):
        self.setting = setting

    def act(self, cell, xi_row):
        self.last_prediction = (float("nan"), float("nan"))
        return self.setting, 0.0


def interval_tracking_error(array, args):
    """Mean over control intervals of |interval mean queue - target|."""
    t = array[:, 0]
    if t.size == 0:
        return float("nan")
    edges = np.arange(t[0], t[-1] + args.control_interval, args.control_interval)
    errors = []
    for lo in edges[:-1]:
        window = (t >= lo) & (t < lo + args.control_interval)
        if window.sum() >= 5:
            errors.append(abs(float(array[window, 2].mean()) - args.q_target))
    return float(np.mean(errors)) if errors else float("nan")


def prediction_error(actions, array, args):
    """How well the controller's own forecast matched what then happened.

    A controller can pick badly for two different reasons: the objective can
    be wrong, or the prediction feeding it can be wrong. Only the second is
    about the operator. This measures it in the loop, by comparing what the
    chosen candidate was predicted to produce over the next interval against
    what the queue actually did over that interval.
    """
    if not actions:
        return None
    t = array[:, 0]
    errors_mean, errors_tail = [], []
    # The tail threshold of the 90-packet buffer, q_crit = 0.8 * 90.
    tail = 72.0
    for row in actions:
        if len(row) < 5 or not np.isfinite(row[3]):
            continue
        start = float(row[0])
        window = (t >= start) & (t < start + args.control_interval)
        if window.sum() < 5:
            continue
        realised_mean = float(array[window, 2].mean())
        realised_tail = float((array[window, 2] > tail).mean())
        errors_mean.append(abs(row[3] - realised_mean))
        errors_tail.append(abs(row[4] - realised_tail))
    if not errors_mean:
        return None
    return {"intervals": len(errors_mean),
            "mean_queue_error": float(np.mean(errors_mean)),
            "tail_probability_error": float(np.mean(errors_tail))}


def run_episode(runner: Ns3Runner, controller, condition, args, grid, label):
    """One ns-3 run with this controller in the loop."""
    flags = (
        f"{condition_flags(condition)} "
        f"--samples={args.samples} --warmup={args.warmup} "
        f"--runSeed={condition['run_seed']} "
        f"--controlIntervalS={args.control_interval} "
        f"--surgeFlows={condition.get('surge_flows', 0)} "
        f"--surgeStartS={condition.get('surge_start_s', -1.0)} "
        f"--surgeStopS={condition.get('surge_stop_s', -1.0)}"
    )
    xi_row = (condition["capacity_pps"], condition["rtt_prop_s"],
              float(condition["n_flows"]), condition["p_err"])
    process = runner.popen(runner.binary_argv("qtro-closedloop", flags))

    samples, actions = [], []
    try:
        # readline() rather than iterating the file object. Iteration uses an
        # internal read-ahead buffer that will not return the CTRL line until
        # the buffer fills, which deadlocks a request-response protocol: ns-3
        # is waiting for the action that is sitting unread in Python's buffer.
        while True:
            line = process.stdout.readline()
            if not line:
                break
            line = line.strip()
            if line.startswith("S,"):
                parts = line.split(",")
                samples.append((float(parts[1]), float(parts[2]),
                                float(parts[3]), int(parts[4]), int(parts[6])))
            elif line.startswith("CTRL,"):
                parts = line.split(",")
                q_inst, cwnd, phase = float(parts[3]), float(parts[4]), float(parts[5])
                cell = int(grid.digitize(
                    torch.tensor([cwnd]), torch.tensor([q_inst]),
                    torch.tensor([phase]))[0])
                # The router publishes its active flow count, so a
                # controller conditions on the load in force rather than the
                # one the run started with. Every controller gets it.
                row = xi_row
                # Field 8 is nActive; field 7 is the delivered-packet count.
                if args.known_load == "current" and len(parts) > 8:
                    row = (xi_row[0], xi_row[1], float(parts[8]), xi_row[3])
                (p_max, min_th), _ = controller.act(cell, row)
                predicted = getattr(controller, "last_prediction",
                                    (float("nan"), float("nan")))
                actions.append((float(parts[1]), p_max, min_th,
                                predicted[0], predicted[1]))
                process.stdin.write(f"{p_max},{min_th}\n")
                process.stdin.flush()
    except BrokenPipeError:
        pass
    finally:
        try:
            process.stdin.close()
        except Exception:
            pass
        process.wait(timeout=args.timeout)

    if not samples:
        return None
    array = np.array(samples, dtype=np.float64)
    queue = array[:, 2]
    tail = float(grid.q_crit)
    return {
        "label": label,
        "samples": len(samples),
        "mean_queue": float(queue.mean()),
        "p95_queue": float(np.percentile(queue, 95)),
        "tail_fraction": float((queue > tail).mean()),
        "tracking_rmse": float(np.sqrt(((queue - args.q_target) ** 2).mean())),
        "mean_offset": float(abs(queue.mean() - args.q_target)),
        # Tracking error integrated over control intervals rather than taken
        # on the whole-run mean.
        #
        # The whole-run mean lets errors in opposite directions cancel: under a
        # load that rises and falls, a single setting that sits five packets
        # low while the load is light and five packets high while it is heavy
        # averages to exactly the target and scores perfectly, while never
        # being near it. That is the metric hiding the thing a controller is
        # for. This is the absolute error a control paper reports.
        "tracking_mae": interval_tracking_error(array, args),
        "drops": int(array[-1, 4]),
        "mean_cwnd": float(array[:, 1].mean()),
        "n_actions": len(actions),
        "prediction": prediction_error(actions, array, args),
        "distinct_actions": len({(a[1], a[2]) for a in actions}),
        "actions": actions[:64],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None,
                    help="YAML config (default: the built-in values)")
    ap.add_argument("--models", required=True,
                    help="comma-separated name=run-directory entries")
    ap.add_argument("--with-domain", action="store_true",
                    help="also run the fluid and diffusion solvers in the loop")
    ap.add_argument("--horizon", type=int, default=500,
                    help="snapshots pushed forward per decision; 500 x 10 ms = one interval")
    ap.add_argument("--control-interval", type=float, default=5.0,
                    help="seconds between decisions")
    ap.add_argument("--q-target", type=float, default=20.0)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--risk", default="percentile",
                    choices=("percentile", "threshold"),
                    help=("threshold penalises predicted P(q > q_crit), which "
                          "is inert here because the queue crosses q_crit in "
                          "under 1%% of samples; percentile penalises the "
                          "predicted 95th percentile above --q95-limit"))
    ap.add_argument("--q95-limit", type=float, default=40.0)
    ap.add_argument("--init-p-max", type=float, default=0.05)
    ap.add_argument("--init-min-th", type=float, default=26.0,
                    help=("the setting the run starts from and that the fixed "
                          "controller holds. A well-tuned start leaves an "
                          "adaptive controller about one packet to win, which "
                          "is too little to separate anything; the operating "
                          "problem Hollot et al. pose is a RED configuration "
                          "that is not right for the load."))
    ap.add_argument("--lam-move", type=float, default=0.01,
                    help="move-suppression weight; 0 lets the controllers switch "
                         "setting almost every interval")
    ap.add_argument("--samples", type=int, default=6000)
    ap.add_argument("--warmup", type=float, default=8.0)
    ap.add_argument("--seeds", default="1,2,3,4,5,6,7,8")
    ap.add_argument("--surge-flows", type=int, default=6)
    ap.add_argument("--surge-start-s", type=float, default=5.0)
    ap.add_argument("--known-load", default="current",
                    choices=("current", "initial"),
                    help=("whether the controller is told the flow count now "
                          "in force or the one the run started with; a flow "
                          "table supplies the first"))
    ap.add_argument("--surge-stop-s", type=float, default=25.0,
                    help=("seconds when the extra sessions leave again. A load "
                          "that only rises lets one setting be best throughout, "
                          "and the controllers then agree on it in 97%% of "
                          "decisions, so the loop cannot separate them"))
    ap.add_argument("--capacity", type=float, default=2000.0)
    ap.add_argument("--rtt", type=float, default=0.030)
    ap.add_argument("--flows", type=int, default=2)
    ap.add_argument("--p-err", type=float, default=0.0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--seed-index", type=int, default=0,
                    help="checkpoint seed to place in the loop")
    ap.add_argument("--out", required=True)
    add_ns3_arguments(ap)
    args = ap.parse_args()

    device = resolve_device(args.device)
    runner = Ns3Runner.from_args(args)
    cfg = load_config(args.config, [f"sim.device={device}"])
    grid = Grid(cfg.grid, cfg.net, device=device)
    # Digitising happens on the observation coming back from ns-3, which is
    # a handful of scalars, so it stays on the host rather than paying a
    # transfer per control instant.
    host_grid = Grid(cfg.grid, cfg.net, device="cpu")

    controllers: dict[str, object] = {
        "fixed": FixedController((args.init_p_max, args.init_min_th)),
    }
    for name, run_dir in parse_model_specs(args.models).items():
        cks = [p for p in sorted(Path(run_dir).glob("checkpoints/*.pt"))
               if re.search(rf"seed{args.seed_index}$", p.stem)]
        if not cks:
            cks = sorted(Path(run_dir).glob("checkpoints/*.pt"))[:1]
        model, model_grid, model_cfg, _ = load_checkpoint(cks[0], device)
        model.eval()
        controllers[name] = Controller(model, model_grid, model_cfg,
                                       args.horizon, args.q_target, args.lam,
                                       device)
        controllers[name].risk = args.risk
        controllers[name].q95_limit = args.q95_limit
        controllers[name].lam_move = args.lam_move
    if args.with_domain:
        for label, (mode, anchor) in {
                "fluid": ("fluid", "balanced"),
                "diffusion": ("diffusion", "balanced")}.items():
            controllers[label] = DomainController(
                mode, anchor, grid, cfg, args.horizon, args.q_target,
                args.lam, device)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    # Every knob that changes what a comparison means is recorded, not just the
    # ones the summary happens to print. Two runs whose load pattern, movement
    # penalty, or risk term differ are not comparable.
    report = {"horizon": args.horizon, "control_interval_s": args.control_interval,
              "forecast_s": args.horizon * float(cfg.sim.dt_s),
              "q_target": args.q_target, "lam": args.lam,
              "lam_move": args.lam_move, "risk": args.risk,
              "q95_limit": args.q95_limit, "known_load": args.known_load,
              "candidates": len(candidates()), "seeds": seeds,
              "models": args.models, "samples": args.samples,
              "warmup_s": args.warmup,
              "capacity_pps": args.capacity, "rtt_prop_s": args.rtt,
              "n_flows": args.flows, "p_err": args.p_err,
              "init_p_max": args.init_p_max, "init_min_th": args.init_min_th,
              "surge_flows": args.surge_flows,
              "surge_start_s": args.surge_start_s,
              "surge_stop_s": args.surge_stop_s, "episodes": []}

    for seed in seeds:
        condition = {
            "capacity_pps": args.capacity, "rtt_prop_s": args.rtt,
            "n_flows": args.flows, "p_err": args.p_err,
            "p_max": args.init_p_max, "min_th": args.init_min_th,
            "max_th_frac": 0.5,
            "buffer_pkts": float(cfg.grid.buffer_pkts), "w_q": 1e-4,
            "run_seed": seed,
            "surge_flows": args.surge_flows,
            "surge_start_s": args.surge_start_s,
            "surge_stop_s": args.surge_stop_s,
        }
        for label, controller in controllers.items():
            # Each episode starts from the setting the scenario is configured
            # with, so move suppression cannot carry state across seeds.
            if hasattr(controller, "previous"):
                controller.previous = (condition["p_max"], condition["min_th"])
            result = run_episode(runner, controller, condition, args, host_grid, label)
            if result is None:
                print(f"  seed {seed} {label}: no samples", flush=True)
                continue
            result["seed"] = seed
            report["episodes"].append(result)
            print(f"  seed {seed} {label:>10}: mean q {result['mean_queue']:6.2f}  "
                  f"p95 {result['p95_queue']:6.1f}  "
                  f"tail {result['tail_fraction']:.4f}  "
                  f"drops {result['drops']:5d}  "
                  f"IAE {result['tracking_mae']:6.2f}  "
                  f"{result['distinct_actions']} distinct actions", flush=True)

    labels = list(controllers)
    print(f"\n{'controller':>12}{'mean q':>10}{'p95':>8}{'tail':>9}"
          f"{'drops':>9}{'IAE':>8}{'|offset|':>10}{'rmse':>8}")
    summary = {}
    for label in labels:
        rows = [e for e in report["episodes"] if e["label"] == label]
        if not rows:
            continue
        summary[label] = {
            k: float(np.mean([r[k] for r in rows]))
            for k in ("mean_queue", "p95_queue", "tail_fraction", "drops",
                      "tracking_rmse", "mean_offset", "tracking_mae",
                      "mean_cwnd")}
        s = summary[label]
        print(f"{label:>12}{s['mean_queue']:>10.2f}{s['p95_queue']:>8.1f}"
              f"{s['tail_fraction']:>9.4f}{s['drops']:>9.0f}"
              f"{s['tracking_mae']:>8.2f}{s['mean_offset']:>10.2f}"
              f"{s['tracking_rmse']:>8.2f}")
    report["summary"] = summary

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
