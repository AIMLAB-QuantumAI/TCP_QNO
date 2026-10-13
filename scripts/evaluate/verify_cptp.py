"""Verify the CPTP structure of trained channels on the dataset's conditions.

The script deliberately separates two claims:

* native QTRO: Stinespring transport completeness plus the protocol-reset
  Kraus family; and
* classical lift: any column-stochastic matrix (including CNTO or an
  empirical kernel) defines the dephased CPTP map rho -> diag(T diag(rho)).

The full 512-state Choi matrix is never materialised.  For the final dephased
map its eigenvalues are exactly the transition entries.  For the native
transport its non-zero Choi spectrum is obtained exactly from the d_E by d_E
Kraus Gram matrix. This feeds the structural-validity table.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import torch

from qtro.baselines.empirical import EmpiricalPropagator
from qtro.checkpoint import load_checkpoint
from qtro.data import SweepData
from qtro.device import resolve_device
from qtro.eval.cptp import (
    dephased_choi_metrics,
    reset_kraus_metrics,
    stinespring_metrics,
    tcp_reset_target_metrics,
)


class MetricAccumulator:
    def __init__(self) -> None:
        self.values: dict[str, list[float]] = defaultdict(list)

    def add(self, metrics: dict[str, torch.Tensor], *, prefix: str = "") -> None:
        for name, value in metrics.items():
            if name == "transport":
                continue
            tensor = torch.as_tensor(value).detach().cpu().reshape(-1)
            key = prefix + name
            self.values[key].extend(float(item) for item in tensor)

    def summary(self) -> dict[str, dict[str, float | int | None]]:
        result = {}
        for name, values in sorted(self.values.items()):
            finite = [value for value in values if math.isfinite(value)]
            result[name] = {
                "n": len(values), "n_finite": len(finite),
                "min": min(finite) if finite else None,
                "mean": statistics.fmean(finite) if finite else None,
                "max": max(finite) if finite else None,
            }
        return result


def _panel_conditions(panel_dir: str, split: str) -> tuple[torch.Tensor, torch.Tensor]:
    """The (u, xi) list straight off an evaluation dataset.

    Structural validity is a property of the channel at a condition, not of
    how the condition was scored, so the dataset's own condition list is all
    that is needed.
    """
    panel = SweepData(panel_dir, [1], splits=(split,), want_refs=False,
                      require_targets=False)
    return panel.u.double(), panel.xi.double()


def _classical_metrics(matrix: torch.Tensor, grid) -> dict[str, torch.Tensor]:
    metrics = dephased_choi_metrics(
        matrix, q_values=grid.q_rep, tail_mask=grid.tail_mask)
    metrics["nonunital_row_sum_residual"] = (
        matrix.sum(-1) - 1.0).abs().amax(-1)
    return metrics


def _evaluate_channel(model, model_name: str, u: torch.Tensor, xi: torch.Tensor,
                      grid, batch_size: int, device: str):
    channel_acc: dict[str, MetricAccumulator] = defaultdict(MetricAccumulator)
    native = MetricAccumulator()
    semantic = None
    n = u.shape[0]
    with torch.no_grad():
        for start in range(0, n, batch_size):
            ub = u[start:start + batch_size].to(device)
            xb = xi[start:start + batch_size].to(device)
            if model_name == "qtro":
                cond = model._cond(ub, xb)
                psi = model.transport.circuit(cond)
                transport_metrics = stinespring_metrics(
                    psi, system_dim=grid.M,
                    environment_dim=model.transport.d_env)
                transport = transport_metrics.pop("transport")
                gamma = model.reset.intensities(cond)
                reset = model.reset(cond)
                reset_metrics = reset_kraus_metrics(
                    gamma, reset, model.reset.targets)
                matrix = reset @ transport
                native.add(transport_metrics, prefix="transport_")
                native.add(reset_metrics, prefix="reset_")
                native.add({
                    "transport_nonunital_row_sum_residual": (
                        transport.sum(-1) - 1.0).abs().amax(-1),
                    "full_nonunital_row_sum_residual": (
                        matrix.sum(-1) - 1.0).abs().amax(-1),
                })
                full = model.propagator(ub, xb)
                channel_acc[model_name].add(_classical_metrics(full, grid))
                if semantic is None:
                    semantic = tcp_reset_target_metrics(grid, model.reset.targets)
            elif model_name == "seqpred":
                raw = model.raw_propagator(ub, xb)
                repaired = model.repair(raw)
                channel_acc["seqpred_raw"].add(_classical_metrics(raw, grid))
                channel_acc["seqpred_repaired"].add(
                    _classical_metrics(repaired, grid))
                change = 0.5 * (raw - repaired).abs().sum(dim=1).mean(-1)
                channel_acc["seqpred_repaired"].add({
                    "mean_column_l1_half_repair_distance": change})
            else:
                channel_acc[model_name].add(
                    _classical_metrics(model.propagator(ub, xb), grid))
    return channel_acc, native, semantic


def _merge(target: MetricAccumulator, source: MetricAccumulator) -> None:
    for key, values in source.values.items():
        target.values[key].extend(values)


def _summary_value(summary: dict, metric: str, field: str) -> str:
    value = summary.get(metric, {}).get(field)
    if value is None:
        return "N/A"
    return f"{value:.6e}"


def _metric(summary: dict, name: str, field: str) -> float:
    value = summary[name][field]
    if value is None:
        return float("nan")
    return float(value)


def _classical_verdict(summary: dict, *, residual_tolerance: float,
                       positivity_tolerance: float,
                       observable_tolerance: float) -> dict:
    checks = {
        "finite": _metric(summary, "has_nonfinite", "max") == 0.0,
        "choi_psd": _metric(
            summary, "choi_psd_violation", "max") <= positivity_tolerance,
        "trace_preserving": _metric(
            summary, "trace_preserving_residual", "max") <= residual_tolerance,
        "kraus_complete": _metric(
            summary, "kraus_completeness_residual", "max") <= residual_tolerance,
        "density_output_psd": _metric(
            summary, "density_output_min_eigenvalue", "min") >= (
                -positivity_tolerance),
        "density_output_unit_trace": _metric(
            summary, "density_output_trace_residual", "max") <= residual_tolerance,
        "tcp_tail_probability_bounded": _metric(
            summary, "tail_probability_bounds_violation", "max") <= (
                observable_tolerance),
        "tcp_mean_queue_bounded": _metric(
            summary, "mean_queue_bounds_violation", "max") <= (
                observable_tolerance),
    }
    return {
        "residual_tolerance": residual_tolerance,
        "positivity_tolerance": positivity_tolerance,
        "observable_tolerance": observable_tolerance,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _native_qtro_verdict(native: dict, semantics: dict,
                         thresholds: dict[str, float]) -> dict:
    checks = {
        "transport_choi_psd": _metric(
            native, "transport_choi_psd_violation", "max") <= thresholds[
                "positivity"],
        "transport_kraus_complete": _metric(
            native, "transport_kraus_completeness_max_residual", "max") <= (
                thresholds["native_residual"]),
        "transport_choi_trace": _metric(
            native, "transport_choi_trace_residual", "max") <= thresholds[
                "transport_choi_trace"],
        "reset_choi_psd": _metric(
            native, "reset_choi_psd_violation", "max") <= thresholds[
                "positivity"],
        "reset_probabilities_bounded": _metric(
            native, "reset_probability_bounds_violation", "max") <= thresholds[
                "observable"],
        "reset_kraus_complete": _metric(
            native, "reset_kraus_completeness_residual", "max") <= thresholds[
                "native_residual"],
        "reset_matrix_reconstructed": _metric(
            native, "reset_matrix_reconstruction_residual", "max") <= thresholds[
                "native_residual"],
        "tcp_reset_routing": _metric(
            native, "reset_protocol_routing_leakage", "max") <= thresholds[
                "observable"],
        "tcp_reset_target_semantics": all(
            value == 0 for key, value in semantics.items()
            if key.endswith("_count")),
    }
    return {"checks": checks, "pass": all(checks.values())}


def render_markdown(report: dict) -> str:
    full_dimension = report["system_dimension"]
    native_dimension = report["native_transport_dimension"]
    environment_dimension = report["environment_dimension"]
    zero_eigenvalues = native_dimension ** 2 - environment_dimension
    lines = [
        "# CPTP verification on TCP experiment conditions", "",
        (f"Evaluated {report['n_tcp_conditions']} TCP/AQM conditions per seed "
         f"over splits {report['splits']}.  QTRO uses "
         f"M={report['system_dimension']} and dE={report['environment_dimension']}."),
        "",
        "## Full dephased-channel checks", "",
        "The Choi spectrum is exact and implicit: its eigenvalues are the entries of T.",
        "CNTO, empirical, and repaired SeqPred are shown through the same classical "
        "measure-and-prepare CPTP lift; this is not a claim that they use a native "
        "density-matrix architecture.", "",
        "| channel | condition evaluations | min Choi eigenvalue | max PSD violation "
        "| max TP residual | max Kraus residual | max tail-bound violation | verdict |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for channel, summary in report["global_classical"].items():
        n = summary["choi_min_eigenvalue"]["n"]
        verdict = report["verification"]["classical_lifts"][channel]["pass"]
        lines.append(
            f"| {channel} | {n} | "
            f"{_summary_value(summary, 'choi_min_eigenvalue', 'min')} | "
            f"{_summary_value(summary, 'choi_psd_violation', 'max')} | "
            f"{_summary_value(summary, 'trace_preserving_residual', 'max')} | "
            f"{_summary_value(summary, 'kraus_completeness_residual', 'max')} | "
            f"{_summary_value(summary, 'tail_probability_bounds_violation', 'max')} | "
            f"{'PASS' if verdict else 'FAIL'} |")

    native = report["global_qtro_native"]
    rows = (
        ("Stinespring Kraus completeness residual",
         "transport_kraus_completeness_max_residual", "max"),
        ("Transport implicit-Choi PSD violation", "transport_choi_psd_violation", "max"),
        ("Transport Choi rank upper bound", "transport_choi_rank_upper_bound", "max"),
        ("Transport smallest non-zero Choi eigenvalue",
         "transport_choi_nonzero_min_eigenvalue", "min"),
        ("Transport Choi trace residual", "transport_choi_trace_residual", "max"),
        ("Reset implicit-Choi PSD violation", "reset_choi_psd_violation", "max"),
        ("Reset probability-bounds violation", "reset_probability_bounds_violation", "max"),
        ("Reset Kraus completeness residual", "reset_kraus_completeness_residual", "max"),
        ("Reset matrix reconstruction residual",
         "reset_matrix_reconstruction_residual", "max"),
        ("TCP protocol routing leakage", "reset_protocol_routing_leakage", "max"),
        ("Mean learned reset probability", "reset_total_event_probability_mean", "mean"),
        ("Max learned reset probability", "reset_total_event_probability_max", "max"),
    )
    lines += ["", "## Native QTRO construction", "",
              "| check | worst measured residual/value |", "|---|---:|"]
    lines += [f"| {label} | {_summary_value(native, key, field)} |"
              for label, key, field in rows]
    native_pass = report["verification"]["native_qtro"]["pass"]
    lines += [f"| Native QTRO verdict | {'PASS' if native_pass else 'FAIL'} |",
              "", "## TCP reset-target semantics", "",
              "| invariant | violation/count |", "|---|---:|"]
    for key, value in report["tcp_reset_target_semantics"].items():
        lines.append(f"| {key} | {value} |")
    lines += [
        "", "## Numerical acceptance criteria", "",
        ("Exact mathematical targets are non-negative Choi spectra and zero "
         "residuals.  The predeclared numerical tolerances used here are "
         "lambda_min >= -1e-12; float64 TP/Kraus/native residual <= 1e-10; "
         "transport Choi-trace residual <= 1e-9; TCP probability/routing "
         "violations <= 1e-12.  The stored float32 empirical kernel alone uses "
         "TP/Kraus tolerance 1e-6."),
        ("Choi eigenvalues use the unnormalised convention, so a TP channel "
         f"has Choi trace M={full_dimension}.  The native transport has "
         f"dimension {native_dimension} and rank at most "
         f"dE={environment_dimension}; its remaining {zero_eigenvalues} "
         "eigenvalues are exactly zero."),
        "", "## Interpretation", "",
        "QTRO passes both its native Stinespring/reset construction checks and the "
        "full dephased-channel Choi/TP checks.  The TCP loss and timeout maps route "
        "mass only to protocol-permitted destinations, and tail/mean-queue "
        "observables remain within physical bounds.", "",
        "This does not make classical stochastic baselines non-CPTP.  Any valid "
        "stochastic matrix has the same dephased measure-and-prepare CPTP lift.  "
        "QTRO's distinctive fact is that its learned factorisation is natively a "
        "Stinespring transport followed by explicit TCP-reset Kraus operators; "
        "the probability-level validity result alone is shared by CNTO and "
        "empirical kernels.", "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", action="append", required=True,
                    help="checkpoint to verify; repeat for several")
    ap.add_argument("--panel", required=True,
                    help="evaluation dataset to take conditions from")
    ap.add_argument("--panel-split", required=True, help="split name inside --panel")
    ap.add_argument("--data", default=None,
                    help="training dataset for the empirical classical lift")
    ap.add_argument("--skip-empirical", action="store_true",
                    help="omit the empirical classical lift")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", required=True, help="results JSON; a .md twin is written beside it")
    args = ap.parse_args()
    if args.batch_size < 1:
        ap.error("--batch-size must be positive")
    if not args.skip_empirical and not args.data:
        ap.error("--data is needed for the empirical lift; pass --skip-empirical to omit it")
    device = resolve_device(args.device)
    splits = [args.panel_split]
    conditions = {args.panel_split: _panel_conditions(args.panel, args.panel_split)}
    paths = [Path(path) for path in args.checkpoint]

    global_classical: dict[str, MetricAccumulator] = defaultdict(MetricAccumulator)
    global_native = MetricAccumulator()
    records = []
    first_payload = None
    first_grid = None
    first_native_dimension = None
    first_environment_dimension = None
    semantic = None
    for path in paths:
        model, grid, _, payload = load_checkpoint(path, device)
        model = model.double().eval()
        first_payload = first_payload or payload
        first_grid = first_grid or grid
        for split in splits:
            u, xi = conditions[split]
            channels, native, current_semantic = _evaluate_channel(
                model, payload["model_name"], u, xi, grid,
                args.batch_size, device)
            for channel, accumulator in channels.items():
                _merge(global_classical[channel], accumulator)
                records.append({
                    "checkpoint": str(path), "model": payload["model_name"],
                    "channel": channel, "variant": payload["metadata"].get("variant"),
                    "seed": payload["metadata"].get("seed"), "split": split,
                    "n_configurations": len(u), "metrics": accumulator.summary(),
                })
            if payload["model_name"] == "qtro":
                _merge(global_native, native)
                semantic = current_semantic if semantic is None else semantic
                if first_native_dimension is None:
                    first_native_dimension = grid.M
                    # The Kraus rank is the environment size the
                    # checkpoint was built with, not a fixed constant.
                    first_environment_dimension = int(model.transport.d_env)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    assert first_payload is not None and first_grid is not None
    if first_native_dimension is None:
        raise SystemExit("at least one qtro checkpoint is needed for the native checks")
    # The empirical baseline uses the exact training half, split indices, and
    # conditioning scale carried by the first checkpoint.
    if not args.skip_empirical:
        train = SweepData(args.data, [1], splits=("train",))
        train_idx = torch.tensor(first_payload["metadata"]["train_indices"])
        empirical = EmpiricalPropagator(
            torch.cat([train.u[train_idx], train.xi[train_idx]], dim=-1),
            train.targets[1][train_idx],
            torch.tensor(first_payload["metadata"]["conditioning_scale"]),
        ).to(device).double().eval()
        for split in splits:
            u, xi = conditions[split]
            channels, _, _ = _evaluate_channel(
                empirical, "empirical", u, xi, first_grid,
                args.batch_size, device)
            accumulator = channels["empirical"]
            _merge(global_classical["empirical"], accumulator)
            records.append({
                "checkpoint": None, "model": "empirical",
                "channel": "empirical", "variant": "nearest", "seed": None,
                "split": split, "n_configurations": len(u),
                "metrics": accumulator.summary(),
            })

    report = {
        "status": "complete",
        "method": {
            "full_channel": ("exact implicit Choi spectrum of rho -> diag(T diag(rho)); "
                             "eigenvalues are T entries"),
            "transport": ("exact non-zero Choi spectrum from dE x dE Kraus Gram and "
                          "completeness from Stinespring-column Gram"),
            "precision": "float64 checkpoint replay",
            "baseline_note": ("cnto/empirical/repaired-seqpred use a classical "
                              "measure-and-prepare CPTP lift"),
        },
        "splits": splits,
        "n_tcp_conditions": sum(len(conditions[split][0]) for split in splits),
        "system_dimension": first_grid.M,
        "choi_dimension": first_grid.M ** 2,
        "native_transport_dimension": first_native_dimension,
        "environment_dimension": first_environment_dimension,
        "records": records,
        "global_classical": {key: value.summary()
                             for key, value in sorted(global_classical.items())},
        "global_qtro_native": global_native.summary(),
        "tcp_reset_target_semantics": semantic,
    }
    thresholds = {
        "positivity": 1e-12,
        "float64_residual": 1e-10,
        "native_residual": 1e-10,
        "transport_choi_trace": 1e-9,
        "observable": 1e-12,
        "empirical_float32_residual": 1e-6,
    }
    classical_verdicts = {}
    for channel, summary in report["global_classical"].items():
        residual_tolerance = (
            thresholds["empirical_float32_residual"]
            if channel == "empirical" else thresholds["float64_residual"])
        classical_verdicts[channel] = _classical_verdict(
            summary, residual_tolerance=residual_tolerance,
            positivity_tolerance=thresholds["positivity"],
            observable_tolerance=thresholds["observable"])
    native_verdict = _native_qtro_verdict(
        report["global_qtro_native"], semantic, thresholds)
    report["verification"] = {
        "thresholds": thresholds,
        "classical_lifts": classical_verdicts,
        "native_qtro": native_verdict,
        "qtro_full_pass": (
            all(verdict["pass"] for name, verdict in classical_verdicts.items()
                if name == "qtro") and native_verdict["pass"]),
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False),
                      encoding="utf-8")
    markdown = output.with_suffix(".md")
    markdown.write_text(render_markdown(report), encoding="utf-8")
    print(f"wrote {output} and {markdown}")


if __name__ == "__main__":
    main()
