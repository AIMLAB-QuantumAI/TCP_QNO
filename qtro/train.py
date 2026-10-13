"""Training loop for every model family, with a pre-flight memory guard.

The guard is not defensive boilerplate. Measured on the 16 GB card the
reported runs used, a QTRO at L=8, d_E=4 needs 6.59 GiB of activations for
16 conditions and 26.28 GiB for 64, and the 64-condition case does not raise
OutOfMemoryError. It spills to shared host memory and takes 134 s per step
instead of 0.39 s, a factor of 300. A sweep that hits this does not fail, it
silently crawls, so the estimate is made before the first step and the run
refuses to start.
"""
from __future__ import annotations

import math
import time

import torch

from qtro.losses import dropped_columns, total_loss

# Activation bytes per (condition x layer x environment dimension), fitted to
# the measurement at (16, 8, 4): 6.59 * 2**30 / (16 * 8 * 4) = 1.38e7, and
# cross-checked at (4, 8, 4) -> 1.67 GiB measured against 1.65 predicted and
# (16, 2, 4) -> 1.88 GiB against 1.65. It is a fit, not an identity, so the
# guard leaves the headroom that vram_frac expresses.
BYTES_PER_CONFIG_LAYER_DENV = 1.38e7


def estimate_activation_bytes(batch_configs: int, n_layers: int, d_env: int) -> float:
    """Estimate the circuit activations held for the backward pass."""
    return BYTES_PER_CONFIG_LAYER_DENV * batch_configs * n_layers * d_env


def check_memory(cfg, device: str) -> dict:
    """Refuse to start a run whose activations will not fit.

    Returns the estimate when it passes; raises with the arithmetic when it
    does not, including the batch size that would fit, because the useful
    response to this error is to lower the batch rather than to read the code.
    """
    est = estimate_activation_bytes(cfg.train.batch_configs,
                                    cfg.model.n_layers, cfg.model.d_env)
    if not (isinstance(device, str) and device.startswith("cuda")
            and torch.cuda.is_available()):
        return {
            "estimated_bytes": est,
            "budget_bytes": None,
            "checked": False,
            "effective_batch_configs": cfg.train.batch_configs,
        }
    total = torch.cuda.get_device_properties(0).total_memory
    budget = cfg.train.vram_frac * total
    if est > budget:
        per_config = est / max(cfg.train.batch_configs, 1)
        fits = int(budget // per_config)
        raise RuntimeError(
            f"estimated activations {est / 2**30:.2f} GiB exceed "
            f"{cfg.train.vram_frac:.0%} of {total / 2**30:.1f} GiB VRAM "
            f"({budget / 2**30:.2f} GiB) at batch_configs="
            f"{cfg.train.batch_configs}, n_layers={cfg.model.n_layers}, "
            f"d_env={cfg.model.d_env}. Use train.batch_configs<={fits}. "
            f"Refusing to start: over-budget runs spill to shared memory and "
            f"slow down about 300x rather than failing.")
    return {
        "estimated_bytes": est,
        "budget_bytes": budget,
        "checked": True,
        "effective_batch_configs": cfg.train.batch_configs,
    }


def batch_index_generator(seed: int, n_data: int, batch_configs: int,
                          sample_offset_steps: int = 0) -> torch.Generator:
    """Build a reproducible minibatch stream, optionally after prior steps.

    A weight-only continuation creates a fresh optimizer because checkpoints
    do not contain Adam state. It must not also repeat the first phase's
    minibatches: advancing the CPU generator by the number of indices already
    consumed makes phase ``k`` use the same data stream as a monolithic run
    with the same seed and batch size.
    """
    if n_data < 1 or batch_configs < 1:
        raise ValueError("n_data and batch_configs must be positive")
    if sample_offset_steps < 0:
        raise ValueError("sample_offset_steps must be non-negative")
    gen = torch.Generator().manual_seed(seed)
    if sample_offset_steps:
        # Draw in one flat tensor: torch.randint consumes the generator in the
        # same order regardless of the output shape.
        torch.randint(n_data, (sample_offset_steps * batch_configs,),
                      generator=gen)
    return gen


def validated_lr_schedule(schedule: dict[int, float] | None) -> dict[int, float]:
    """Return a sorted, validated global-step learning-rate schedule."""
    out = {}
    for raw_step, raw_lr in (schedule or {}).items():
        step, lr = int(raw_step), float(raw_lr)
        if step != raw_step or step < 0:
            raise ValueError("learning-rate milestones must be non-negative integers")
        if not math.isfinite(lr) or lr <= 0.0:
            raise ValueError("scheduled learning rates must be finite and positive")
        if step in out:
            raise ValueError(f"duplicate learning-rate milestone {step}")
        out[step] = lr
    return dict(sorted(out.items()))


def parse_lr_milestones(text: str) -> dict[int, float]:
    """Parse ``STEP:LR,...`` without silently accepting duplicate steps."""
    schedule = {}
    if not text.strip():
        return schedule
    try:
        for item in text.split(","):
            step_text, separator, lr_text = item.strip().partition(":")
            if not separator:
                raise ValueError
            step, lr = int(step_text), float(lr_text)
            if step in schedule:
                raise ValueError(f"duplicate learning-rate milestone {step}")
            if step < 0 or not math.isfinite(lr) or lr <= 0.0:
                raise ValueError
            schedule[step] = lr
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "learning-rate milestones must look like "
            "2000:0.001,4000:0.0003 with unique non-negative steps and "
            "positive finite rates") from exc
    return dict(sorted(schedule.items()))


class ConsoleLog:
    """Prints scalars every ``every`` steps and text lines as they come."""

    def __init__(self, every: int):
        self.every = max(1, every)

    def text(self, name: str, value: str) -> None:
        print(f"[{name}] {value}", flush=True)

    def scalars(self, step: int, values: dict) -> None:
        if step % self.every == 0:
            body = " ".join(f"{k}={v:.6g}" for k, v in values.items())
            print(f"[step {step}] {body}", flush=True)


def train(model, data, grid, cfg, seed: int = 0, log=None,
          eval_every: int = 0, eval_fn=None,
          sample_offset_steps: int = 0,
          lr_schedule: dict[int, float] | None = None,
          max_wall_s: float | None = None) -> dict:
    """Fit `model` to the conditional column targets in `data`.

    `model` needs only `.propagator(u, xi) -> (B, M, M)`, so QTRO and the
    baselines train through the same path: any difference in the reported
    numbers then comes from the parameterisation and not from two separate
    training loops that drifted apart.
    """
    if max_wall_s is not None and (not math.isfinite(max_wall_s)
                                   or max_wall_s <= 0.0):
        raise ValueError("max_wall_s must be finite and positive")
    device = cfg.sim.device
    memory = check_memory(cfg, device)
    torch.manual_seed(seed)

    horizons = [h for h in cfg.data.horizons_train if h <= cfg.train.h0]
    if not horizons:
        raise ValueError(f"h0={cfg.train.h0} excludes every training horizon "
                         f"in {cfg.data.horizons_train}")
    missing = [h for h in horizons if h not in data.targets]
    if missing:
        raise KeyError(f"dataset lacks targets for horizons {missing}")

    schedule = validated_lr_schedule(lr_schedule)
    initial_lr = float(cfg.train.lr)
    for milestone, scheduled_lr in schedule.items():
        if milestone > sample_offset_steps:
            break
        initial_lr = scheduled_lr
    opt = torch.optim.Adam(model.parameters(), lr=initial_lr)
    gen = batch_index_generator(seed, len(data), cfg.train.batch_configs,
                                sample_offset_steps)
    history, t0 = [], time.perf_counter()

    # Report what the column threshold discards once, up front. Silent
    # truncation reads as full coverage.
    drop = dropped_columns(data.counts[horizons[0]], cfg.data.c_min)
    if log is not None:
        log.text("columns_dropped",
                 f"{drop['n_dropped']}/{drop['n_columns']} columns below "
                 f"c_min={cfg.data.c_min}, carrying "
                 f"{drop['occupancy_share_dropped']:.4%} of observed "
                 f"transitions")

    stopped_by_wall = False
    steps_run = 0
    for step in range(cfg.train.n_steps):
        global_step = sample_offset_steps + step
        if global_step in schedule:
            for group in opt.param_groups:
                group["lr"] = schedule[global_step]
        idx = torch.randint(len(data), (cfg.train.batch_configs,),
                            generator=gen)
        batch = data.batch(idx, device)
        P = model.propagator(batch["u"], batch["xi"])
        targets = {h: batch["targets"][h] for h in horizons}
        counts = {h: batch["counts"][h] for h in horizons}
        loss, parts = total_loss(P, targets, counts, grid, cfg,
                                 parameters=model.parameters())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        steps_run = step + 1
        parts["lr"] = float(opt.param_groups[0]["lr"])

        # `parts` is the loss of THIS step's random batch, not of a fixed set,
        # so the logged series is not a learning curve until it is smoothed or
        # replaced by an eval_fn pass.
        if log is not None:
            log.scalars(global_step, parts)
        if eval_every and eval_fn is not None and \
                (step % eval_every == 0 or step == cfg.train.n_steps - 1):
            metrics = eval_fn(model, global_step)
            history.append({"step": global_step, **parts, **metrics})
            if log is not None:
                log.scalars(global_step, metrics)
        elif step % max(1, cfg.train.n_steps // 20) == 0:
            history.append({"step": global_step, **parts})

        # Stop only at an update boundary. This keeps every saved model a
        # valid optimiser state and bounds wall-budget overshoot by one step.
        if max_wall_s is not None and time.perf_counter() - t0 >= max_wall_s:
            stopped_by_wall = True
            if not history or history[-1].get("step") != global_step:
                history.append({"step": global_step, **parts})
            break

    wall_s = time.perf_counter() - t0
    return {
        "seed": seed,
        "sample_offset_steps": sample_offset_steps,
        "completed_steps": sample_offset_steps + steps_run,
        "initial_lr": initial_lr,
        "lr_schedule": schedule,
        "wall_s": wall_s,
        "max_wall_s": max_wall_s,
        "stopped_by_wall": stopped_by_wall,
        "wall_overshoot_s": (None if max_wall_s is None else
                             max(0.0, wall_s - max_wall_s)),
        "memory": memory,
        "columns_dropped": drop,
        "history": history,
        "final": history[-1] if history else {},
    }
