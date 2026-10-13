"""Train every model at every training-set size, one process at a time.

Operator-learning papers report accuracy against the amount of training data
because that is where a structural constraint is supposed to pay: removing
hypotheses should matter most when the data cannot pin the operator down.
This varies the number of *conditions*, which is coverage of the (u, xi)
space the operator is conditioned on.

--train-fraction takes a deterministic nested prefix, so the 27-condition set
is a subset of the 54, which is a subset of the 108. Curves are then paired
across sizes rather than being three unrelated draws.

The full-data point is not retrained: the main training run already holds it
at train_fraction 1.0 with the same config, data and step count, and the
evaluation script takes that run as ``--full``.

Two processes sharing the GPU is not faster here, so this chains rather than
fans out.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

TRAIN = Path(__file__).resolve().with_name("train.py")
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "ns3_h20.yaml"
# 216 post-validation training conditions in the main dataset.
FRACTIONS = ((0.125, 27), (0.25, 54), (0.5, 108))
DEFAULT_MODELS = ("qtro=qtro:4x8,cnto=cnto:rank27,seqpred=seqpred:rank27,"
                  "deeponet=deeponet:rank19")


def parse_models(spec):
    """`dir=model:variant` triples, so two variants of one model can coexist."""
    out = []
    for entry in spec.split(","):
        name, _, rest = entry.partition("=")
        model, _, variant = rest.partition(":")
        if not (name and model and variant):
            raise SystemExit(f"--models entry must be dir=model:variant, "
                             f"got {entry!r}")
        out.append((name.strip(), model.strip(), variant.strip()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--data", required=True, help="training dataset directory")
    ap.add_argument("--out", required=True, help="root directory for n027/, n054/, n108/")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--models", default=DEFAULT_MODELS,
                    help="comma-separated dir=model:variant entries")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    models = parse_models(args.models)

    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    log = root / "driver.jsonl"
    wanted = len(args.seeds.split(","))

    for fraction, n_conditions in FRACTIONS:
        for name, model, variant in models:
            target = root / f"n{n_conditions:03d}" / name
            existing = list((target / "checkpoints").glob("*.pt")) if (
                target / "checkpoints").exists() else []
            if len(existing) >= wanted:
                print(f"skip n={n_conditions}/{name}: "
                      f"{len(existing)} checkpoints present", flush=True)
                continue
            command = [args.python, str(TRAIN),
                       "--config", args.config, "--data", args.data,
                       "--out", str(target), "--model", model,
                       "--variant", variant, "--seeds", args.seeds,
                       "--train-fraction", str(fraction)]
            started = time.time()
            print(f"=== n={n_conditions} conditions / {model} {variant} ===",
                  flush=True)
            result = subprocess.run(command)
            record = {"n_conditions": n_conditions, "train_fraction": fraction,
                      "arm": name, "model": model, "variant": variant,
                      "returncode": result.returncode,
                      "wall_s": round(time.time() - started, 1)}
            with log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            if result.returncode != 0:
                raise SystemExit(f"n={n_conditions}/{name} failed")

    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
