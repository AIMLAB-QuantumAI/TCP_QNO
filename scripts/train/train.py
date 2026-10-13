"""Train one model on a transition dataset and save a checkpoint per seed, with
the resolved configuration, split indices and validation metrics it needs to
be scored later."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from qtro.checkpoint import load_payload, save_checkpoint
from qtro.config import load_config
from qtro.data import (
    SweepData,
    SweepSubset,
    manifest_sha256,
    nested_training_fraction,
    train_validation_split,
)
from qtro.device import resolve_device
from qtro.eval.validation import validation_metrics, validation_metrics_multihorizon
from qtro.grid import Grid
from qtro.registry import build_model, parameter_count
from qtro.train import ConsoleLog, parse_lr_milestones, train
from qtro.variants import MODEL_NAMES, parse_variant


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None,
                    help="YAML config (default: the built-in values)")
    ap.add_argument("--data", required=True, help="dataset directory of shard_*.npz files")
    ap.add_argument("--out", required=True, help="run directory for checkpoints/ and train_*.json")
    ap.add_argument("--model", choices=MODEL_NAMES, default="qtro")
    ap.add_argument(
        "--variant", default="4x8",
        help="depth models: layers x d_env, e.g. 4x8; baselines: rank, e.g. rank27")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument(
        "--max-wall-s", type=float, default=None,
        help=("stop after the first completed update at or beyond this "
              "training-loop wall time; --steps remains a safety ceiling"))
    ap.add_argument("--batch-configs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--h0", type=int, default=None)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--val-seed", type=int, default=20260811)
    ap.add_argument(
        "--val-horizons", default="",
        help=("comma-separated composed horizons for held-out validation, "
              "e.g. 1,5,10,20. Empty keeps only the one-step metric."))
    ap.add_argument(
        "--eval-every", type=int, default=0,
        help=("also record held-out validation every N steps, not only at "
              "the end. Zero disables. Requires --val-horizons."))
    ap.add_argument("--train-fraction", type=float, default=1.0,
                    help="nested fraction of the post-validation training set")
    ap.add_argument("--set", action="append", dest="overrides", default=[],
                    help="config override, e.g. train.lr=0.001")
    ap.add_argument(
        "--resume", default=None,
        help=("checkpoint whose weights initialise a deterministic continuation; "
              "Adam state is restarted because checkpoints store weights, not "
              "optimizer moments"))
    ap.add_argument(
        "--sample-offset", type=int, default=None,
        help=("number of prior minibatch steps to skip in the deterministic "
              "seed stream; with --resume this defaults to, and must match, "
              "the checkpoint's completed step count"))
    ap.add_argument(
        "--lr-milestones", default="",
        help=("global-step learning-rate changes as STEP:LR pairs, e.g. "
              "2000:0.001,4000:0.0003"))
    ap.add_argument("--device", default="auto", help="auto, cpu or cuda[:N]")
    args = ap.parse_args()

    if not 0.0 < args.train_fraction <= 1.0:
        ap.error("--train-fraction must lie in (0,1]")
    if not 0.0 <= args.val_fraction < 1.0:
        ap.error("--val-fraction must lie in [0,1)")
    if args.sample_offset is not None and args.sample_offset < 0:
        ap.error("--sample-offset must be non-negative")
    try:
        lr_schedule = parse_lr_milestones(args.lr_milestones)
    except ValueError as exc:
        ap.error(str(exc))

    device = resolve_device(args.device)
    cfg = load_config(args.config, [f"sim.device={device}", *args.overrides])
    if args.steps is not None:
        cfg.train.n_steps = args.steps
    if args.batch_configs is not None:
        cfg.train.batch_configs = args.batch_configs
    if args.lr is not None:
        cfg.train.lr = args.lr
    if args.h0 is not None:
        cfg.train.h0 = args.h0
    label, model_kwargs = parse_variant(args.model, args.variant, cfg)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    train_horizons = [h for h in cfg.data.horizons_train if h <= cfg.train.h0]
    try:
        base = SweepData(args.data, horizons=train_horizons, splits=("train",))
    except (FileNotFoundError, ValueError) as exc:
        ap.error(str(exc))
    gen = torch.Generator().manual_seed(args.val_seed)
    order = torch.randperm(len(base), generator=gen)
    val_idx, full_train_idx = train_validation_split(order, args.val_fraction)
    train_idx = nested_training_fraction(full_train_idx, args.train_fraction)
    train_data = SweepSubset(base, train_idx)
    val_data = SweepSubset(base, val_idx) if len(val_idx) else None
    # A second load for the evaluation half. Row selection inside SweepData
    # depends only on d["split"], never on the requested horizons, so both
    # loads enumerate the same configurations in the same order and val_idx
    # indexes them identically. The length check makes that assumption fail
    # loudly rather than silently scoring the wrong configurations.
    val_horizons = [int(x) for x in args.val_horizons.split(",") if x.strip()]
    val_ref_data = None
    if val_horizons and val_data is not None:
        ref_base = SweepData(args.data, horizons=val_horizons,
                             splits=("train",), want_refs=True,
                             require_targets=False)
        if len(ref_base) != len(base):
            raise ValueError(
                f"reference load selected {len(ref_base)} rows but the "
                f"training load selected {len(base)}; the two must index "
                f"identically for val_idx to be meaningful")
        val_ref_data = SweepSubset(ref_base, val_idx)
    elif args.eval_every:
        ap.error("--eval-every requires --val-horizons")
    scale = train_data.conditioning_scale()
    data_manifest_sha256 = manifest_sha256(args.data)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    resume_payload = None
    sample_offset = args.sample_offset
    if args.resume is not None:
        resume_payload = load_payload(args.resume)
        if resume_payload["model_name"] != args.model:
            raise ValueError(
                f"resume checkpoint is {resume_payload['model_name']}, "
                f"requested model is {args.model}")
        if len(seeds) != 1:
            raise ValueError("--resume accepts exactly one seed per run")
        resume_meta = resume_payload.get("metadata", {})
        prior_completed = resume_meta.get(
            "completed_steps",
            resume_payload.get("training", {}).get("completed_steps"))
        if prior_completed is None:
            raise ValueError(
                "resume checkpoint has no completed_steps for deterministic "
                "minibatch continuation")
        prior_completed = int(prior_completed)
        if sample_offset is None:
            sample_offset = prior_completed
        elif sample_offset != prior_completed:
            raise ValueError(
                f"--sample-offset={sample_offset} does not match resume "
                f"checkpoint completed_steps={prior_completed}")
        prior_batch = resume_payload.get("config", {}).get(
            "train", {}).get("batch_configs")
        if prior_batch is not None and int(prior_batch) != cfg.train.batch_configs:
            raise ValueError(
                "resume checkpoint uses a different batch_configs value; "
                "step-based RNG continuation would not reproduce the "
                "minibatch stream")
        if "seed" in resume_meta and int(resume_meta["seed"]) != seeds[0]:
            raise ValueError(
                f"resume checkpoint seed={resume_meta['seed']} does not "
                f"match requested seed={seeds[0]}")
        if "train_indices" in resume_meta and \
                resume_meta["train_indices"] != train_idx.tolist():
            raise ValueError("resume checkpoint uses a different training split")
        if "val_indices" in resume_meta and \
                resume_meta["val_indices"] != val_idx.tolist():
            raise ValueError("resume checkpoint uses a different validation split")
        if "conditioning_scale" in resume_meta and not torch.allclose(
                torch.as_tensor(resume_meta["conditioning_scale"]), scale):
            raise ValueError(
                "resume checkpoint uses a different conditioning scale")
        prior_manifest = resume_meta.get("training_manifest_sha256")
        if prior_manifest is not None and prior_manifest != data_manifest_sha256:
            raise ValueError(
                "resume checkpoint uses a different training-data manifest")
    elif sample_offset is None:
        sample_offset = 0

    for seed in seeds:
        torch.manual_seed(seed)
        grid = Grid(cfg.grid, cfg.net, device=device)
        model = build_model(args.model, grid, cfg, model_kwargs).to(device)
        if resume_payload is not None:
            model.load_state_dict(resume_payload["state_dict"])
        model.set_conditioning_scale(scale.to(device))
        start_record = {
            "event": "start", "model": args.model, "variant": label,
            "seed": seed, "parameters": parameter_count(model),
            "train_configs": len(train_data),
            "val_configs": 0 if val_data is None else len(val_data),
            "train_fraction": args.train_fraction,
            "conditioning_scale": scale.tolist(), "steps": cfg.train.n_steps,
            "max_wall_s": args.max_wall_s,
            "sample_offset_steps": sample_offset,
            "lr_schedule": lr_schedule, "device": device,
            "training_data": str(args.data),
            "training_manifest_sha256": data_manifest_sha256,
        }
        print(json.dumps(start_record), flush=True)

        trace: list[dict] = []

        def _trace_eval(trained_model, global_step: int) -> dict:
            """Record the composed-horizon metrics mid-training.

            `train` flattens whatever this returns into its history and hands
            it to `log.scalars`, so the nested payload goes to `trace` and
            only scalars come back.
            """
            was_training = trained_model.training
            trained_model.eval()
            try:
                metrics = validation_metrics_multihorizon(
                    trained_model, val_ref_data, grid, cfg, device,
                    horizons=val_horizons)
            finally:
                trained_model.train(was_training)
            trace.append({"step": int(global_step), "metrics": metrics})
            return {f"val_{key}_fidelity":
                    value["fidelity"]["occupancy_weighted"]
                    for key, value in metrics.items()}

        result = train(
            model, train_data, grid, cfg, seed=seed,
            log=ConsoleLog(max(1, cfg.train.n_steps // 20)),
            sample_offset_steps=sample_offset,
            lr_schedule=lr_schedule, max_wall_s=args.max_wall_s,
            eval_every=args.eval_every,
            eval_fn=(_trace_eval if args.eval_every and val_ref_data is not None
                     else None))
        val = (None if val_data is None else validation_metrics(
            model, val_data, grid, cfg, device))
        val_multi = None
        if val_ref_data is not None:
            val_multi = validation_metrics_multihorizon(
                model, val_ref_data, grid, cfg, device, horizons=val_horizons)
        metadata = {
            "variant": label,
            "seed": seed,
            "parameter_count": parameter_count(model),
            "train_indices": train_idx.tolist(),
            "val_indices": val_idx.tolist(),
            "val_seed": args.val_seed,
            "val_fraction": args.val_fraction,
            "train_fraction": args.train_fraction,
            "full_train_config_count": int(full_train_idx.numel()),
            "conditioning_scale": scale.tolist(),
            "validation": val,
            "validation_multihorizon": val_multi,
            "validation_trace": trace,
            "training_data": str(args.data),
            "training_manifest_sha256": data_manifest_sha256,
            "resumed_from": args.resume,
            "sample_offset_steps": sample_offset,
            "completed_steps": result["completed_steps"],
            "optimizer_resume": "restarted" if args.resume else "new",
            "lr_schedule": lr_schedule,
            "max_wall_s": args.max_wall_s,
            "stopped_by_wall": result["stopped_by_wall"],
            "wall_overshoot_s": result["wall_overshoot_s"],
        }
        path = out_dir / "checkpoints" / f"{args.model}_{label}_seed{seed}.pt"
        save_checkpoint(path, model, cfg, model_name=args.model,
                        model_kwargs=model_kwargs, metadata=metadata,
                        training=result)
        summary = {"checkpoint": str(path), **metadata,
                   "training_wall_s": result["wall_s"]}
        summaries.append(summary)
        print(json.dumps({
            "event": "complete", "checkpoint": str(path),
            "model": args.model, "variant": label, "seed": seed,
            "training_wall_s": result["wall_s"],
            "validation_fidelity": None if val is None else val["fidelity"],
        }), flush=True)

    summary_path = out_dir / f"train_{args.model}_{label}.json"
    # A multi-seed experiment is often launched as separate invocations.
    # Preserve the earlier seeds in that case instead of silently replacing
    # the experiment manifest with only the most recent invocation. The
    # checkpoint path is the stable run identity; a deliberate rerun of the
    # same seed replaces its old record.
    previous = []
    if summary_path.exists():
        loaded = json.loads(summary_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, list):
            raise ValueError(f"existing training manifest is not a list: {summary_path}")
        previous = loaded
    merged = {record["checkpoint"]: record for record in previous}
    merged.update({record["checkpoint"]: record for record in summaries})
    ordered = sorted(merged.values(), key=lambda record: (
        int(record.get("seed", 0)), str(record["checkpoint"])))
    summary_path.write_text(json.dumps(ordered, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
