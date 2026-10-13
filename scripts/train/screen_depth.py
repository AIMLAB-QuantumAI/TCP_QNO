"""Circuit depth against environment dimension at a fixed parameter budget: the
four variants on the L x d_E = 32 diagonal, ranked on the composed horizons,
with the 500-step ranking recorded inside the same run.

Holding the product fixed keeps the parameter count within a few percent and
the activation memory equal (activations scale as L * d_E * B * M^2), so the
only thing that varies is how depth and width are traded against each other.
Each run records its own 500-step point through ``--eval-every``, so a
reversal between 500 steps and the full budget is visible inside one artifact
rather than inferred across two experiments.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

from qtro.eval.validation import rank_depth_variants

TRAIN = Path(__file__).resolve().with_name("train.py")
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "ns3_h5.yaml"
HORIZONS = ("h1", "h5", "h10", "h20")


def variant_grid() -> list[dict]:
    """The L * d_E = 32 diagonal.

    The axis is 15 bins wide and a depth-L Givens ladder displaces mass by at
    most 4L bins, so 8x4 (32 bins) and 4x8 (16 bins) have a bound that
    restricts nothing while 3x10 (12) and 2x16 (8) actually bind. The sweep
    therefore spans both regimes rather than only one.
    """
    return [{"variant": "8x4"}, {"variant": "4x8"},
            {"variant": "3x10"}, {"variant": "2x16"}]


def select_variants(only: str) -> list[dict]:
    """Filter the grid so the sweep can be split across processes.

    The transport block is launch-bound, not compute-bound: a depth-L ladder
    issues 9 sweeps twice per layer and each sweep is a handful of small
    kernels, so one process pins about four cores while leaving the GPU near
    idle. Running two streams roughly halves wall-clock. Splitting does not
    change any result -- each variant and seed is an independent run writing
    its own record.
    """
    grid = variant_grid()
    if not only:
        return grid
    wanted = [v.strip() for v in only.split(",") if v.strip()]
    known = {entry["variant"] for entry in grid}
    unknown = [v for v in wanted if v not in known]
    if unknown:
        raise ValueError(f"unknown variants {unknown}; grid is {sorted(known)}")
    return [entry for entry in grid if entry["variant"] in wanted]


def run(out_dir: Path, data: str, seeds: str, steps: int, config: str,
        python: str, dry_run: bool, only: str = "") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for entry in select_variants(only):
        cmd = [python, str(TRAIN), "--config", config,
               "--data", data, "--model", "qtro",
               "--variant", entry["variant"], "--seeds", seeds,
               "--steps", str(steps),
               "--val-horizons", "1,5,10,20", "--eval-every", "500",
               "--out", str(out_dir)]
        print(" ".join(cmd), flush=True)
        if not dry_run:
            subprocess.run(cmd, check=True)


def _trace_at(record: dict, step: int) -> dict | None:
    for entry in record.get("validation_trace") or []:
        if entry["step"] == step:
            return entry["metrics"]
    return None


def summarise(run_dir: Path) -> dict:
    records = []
    for path in sorted(run_dir.glob("train_*.json")):
        for record in json.loads(path.read_text(encoding="utf-8")):
            if record.get("validation_multihorizon"):
                records.append(record)
    if not records:
        raise ValueError(f"no rankable records in {run_dir}")
    ranked = rank_depth_variants(records)
    # The 500-step ranking is the one a short screen would have chosen on. If
    # it differs from the final ranking, that difference is the result.
    early = []
    for record in records:
        metrics = _trace_at(record, 500)
        if metrics:
            early.append({"variant": record["variant"], "seed": record["seed"],
                          "validation_multihorizon": metrics})
    return {"ranked": ranked,
            "ranked_at_500": rank_depth_variants(early) if early else None,
            "n_records": len(records), "records": records}


def horizon_table(records: list[dict], title: str) -> list[str]:
    """Per-horizon fidelity and TV, seed-averaged, one row per variant.

    The composite score decides the ranking; this is what a reader needs to
    see why, and it is the shape the paper reports at.
    """
    by_variant: dict[str, list[dict]] = {}
    for record in records:
        by_variant.setdefault(record["variant"], []).append(
            record["validation_multihorizon"])
    out = [f"### {title}", "",
           "| variant | seeds | "
           + " | ".join(f"fid {h}" for h in HORIZONS) + " | "
           + " | ".join(f"TV {h}" for h in HORIZONS) + " |",
           "|---|---:|" + "---:|" * (2 * len(HORIZONS))]
    scored = [row["variant"] for row in rank_depth_variants(records)]
    for name in scored:
        group = by_variant[name]
        cells = []
        for key in ("fidelity", "tv"):
            for h in HORIZONS:
                value = statistics.fmean(m[h][key]["occupancy_weighted"] for m in group)
                cells.append(f"{value:.5f}")
        out.append(f"| {name} | {len(group)} | " + " | ".join(cells) + " |")
    return out + [""]


def _table(rows: list[dict], title: str) -> list[str]:
    out = [f"### {title}", "",
           "| variant | score (fidelity - TV, h in 5/10/20) | seeds |",
           "|---|---:|---:|"]
    for row in rows:
        out.append(f"| {row['variant']} | {row['score']:.5f} "
                   f"| {row['n_seeds']} |")
    return out + [""]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="run directory shared by the four variants")
    ap.add_argument("--data", required=True, help="training dataset directory")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--summarise-only", action="store_true")
    ap.add_argument(
        "--variants", default="",
        help=("comma-separated subset of the grid to run, so the sweep "
              "can be split across concurrent processes. Empty runs all."))
    args = ap.parse_args()

    out = Path(args.out)
    if not args.summarise_only:
        run(out, args.data, args.seeds, args.steps, args.config,
            args.python, args.dry_run, only=args.variants)
        if args.dry_run:
            return
    summary = summarise(out)
    (out / "summary.json").write_text(
        json.dumps(summary, indent=1), encoding="utf-8")
    lines = _table(summary["ranked"], f"Final ({args.steps} steps)")
    if summary["ranked_at_500"]:
        lines += _table(summary["ranked_at_500"], "At 500 steps")
    lines += horizon_table(summary["records"], f"Per horizon ({args.steps} steps)")
    early = [{"variant": r["variant"], "validation_multihorizon": m}
             for r in summary["records"]
             if (m := _trace_at(r, 500)) is not None]
    if early:
        lines += horizon_table(early, "Per horizon (500 steps)")
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
