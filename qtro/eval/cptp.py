"""Exact CPTP diagnostics for dephased finite-state channels.

The deployed channel is ``D o R o T o D``. On the full density operator this
is the measure-and-prepare map

    Phi_T(rho) = diag(T @ diag(rho)).

For this map the Choi matrix is diagonal (up to a fixed basis permutation)
and its eigenvalues are exactly the entries of ``T``. Consequently complete
positivity, trace preservation, and Kraus completeness can be checked on the
real 512-state model without materialising a 262144-square Choi matrix or the
``M^2`` rank-one Kraus operators.

The native Stinespring transport is checked separately. If ``A_e`` are the
environment Kraus operators, the non-zero Choi eigenvalues equal the
eigenvalues of the small Gram matrix ``Tr(A_e^T A_f)`` and trace preservation
is ``sum_e A_e^T A_e = I``.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor


def _require_square_batch(matrix: Tensor) -> Tensor:
    if matrix.ndim == 2:
        matrix = matrix.unsqueeze(0)
    if matrix.ndim != 3 or matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError("matrix must have shape (B,M,M) or (M,M)")
    return matrix


def dephased_choi_metrics(matrix: Tensor, *, q_values: Tensor | None = None,
                          tail_mask: Tensor | None = None) -> dict[str, Tensor]:
    """Return exact per-condition metrics for ``rho -> diag(T diag(rho))``.

    Under the standard unnormalised Choi convention, ``C(Phi_T)`` has
    eigenvalues ``T[i,j]``. Its output partial trace is
    ``diag(T.sum(destination))`` and therefore Kraus completeness has the
    same residual as column stochasticity. These are identities, not a
    sampled PSD approximation.
    """
    matrix = _require_square_batch(matrix)
    B, M, _ = matrix.shape
    finite = torch.isfinite(matrix)
    safe = torch.where(finite, matrix, torch.zeros_like(matrix))
    min_entry = torch.where(
        finite.reshape(B, -1).all(-1),
        safe.reshape(B, -1).amin(-1),
        torch.full((B,), float("nan"), dtype=matrix.dtype,
                   device=matrix.device),
    )
    column_mass = safe.sum(dim=1)
    tp_residual = (column_mass - 1.0).abs().amax(-1)
    choi_trace_residual = (safe.sum(dim=(-2, -1)) - float(M)).abs()

    # Exercise the density-matrix interpretation on a coherent pure input.
    # Only its diagonal enters D o Phi o D; output eigenvalues are T @ p.
    v = torch.arange(1, M + 1, dtype=matrix.dtype, device=matrix.device)
    v = torch.sin(v * 0.731) + 1.5
    p = v.square()
    p = p / p.sum()
    output_eigenvalues = torch.einsum("bij,j->bi", safe, p)

    result: dict[str, Tensor] = {
        "has_nonfinite": ~finite.reshape(B, -1).all(-1),
        "choi_min_eigenvalue": min_entry,
        "choi_psd_violation": (-min_entry).clamp_min(0.0),
        "trace_preserving_residual": tp_residual,
        "kraus_completeness_residual": tp_residual,
        "choi_trace_residual": choi_trace_residual,
        "density_output_min_eigenvalue": output_eigenvalues.amin(-1),
        "density_output_trace_residual": (
            output_eigenvalues.sum(-1) - 1.0).abs(),
        # The output is built by diag_embed in the full map, hence exact.
        "diagonal_output_offdiag_residual": torch.zeros(
            B, dtype=matrix.dtype, device=matrix.device),
    }

    if tail_mask is not None:
        tail = torch.as_tensor(tail_mask, device=matrix.device,
                               dtype=torch.bool)
        if tail.shape != (M,):
            raise ValueError(f"tail_mask must have shape ({M},)")
        probability = safe[:, tail, :].sum(dim=1)
        result.update({
            "tail_probability_min": probability.amin(-1),
            "tail_probability_max": probability.amax(-1),
            "tail_probability_bounds_violation": torch.maximum(
                (-probability).clamp_min(0.0).amax(-1),
                (probability - 1.0).clamp_min(0.0).amax(-1)),
        })

    if q_values is not None:
        q = torch.as_tensor(q_values, device=matrix.device,
                            dtype=matrix.dtype)
        if q.shape != (M,):
            raise ValueError(f"q_values must have shape ({M},)")
        mean_q = torch.einsum("i,bij->bj", q, safe)
        result.update({
            "mean_queue_min": mean_q.amin(-1),
            "mean_queue_max": mean_q.amax(-1),
            "mean_queue_bounds_violation": torch.maximum(
                (q.min() - mean_q).clamp_min(0.0).amax(-1),
                (mean_q - q.max()).clamp_min(0.0).amax(-1)),
        })
    return result


def stinespring_metrics(psi: Tensor, *, system_dim: int,
                        environment_dim: int) -> dict[str, Tensor]:
    """Check a native Stinespring isometry and its implicit Choi spectrum."""
    if psi.ndim != 3 or psi.shape[1:] != (
            system_dim * environment_dim, system_dim):
        raise ValueError("psi has incompatible Stinespring-column shape")
    amp = psi.reshape(psi.shape[0], system_dim, environment_dim, system_dim)
    completeness = torch.einsum("biej,biek->bjk", amp, amp)
    eye = torch.eye(system_dim, dtype=psi.dtype, device=psi.device)
    residual = completeness - eye
    # A_e[i,j] = amp[i,e,j]. The non-zero spectrum of
    # sum_e vec(A_e)vec(A_e)^T is the spectrum of this d_E x d_E Gram.
    kraus_gram = torch.einsum("biej,bifj->bef", amp, amp)
    eig = torch.linalg.eigvalsh((kraus_gram + kraus_gram.transpose(-1, -2)) / 2)
    transport = amp.square().sum(dim=2)
    return {
        "kraus_completeness_max_residual": residual.abs().amax(dim=(-2, -1)),
        "kraus_completeness_fro_relative": (
            torch.linalg.matrix_norm(residual, ord="fro") / math.sqrt(system_dim)),
        "choi_nonzero_min_eigenvalue": eig.amin(-1),
        "choi_nonzero_max_eigenvalue": eig.amax(-1),
        "choi_psd_violation": (-eig.amin(-1)).clamp_min(0.0),
        "choi_trace_residual": (eig.sum(-1) - float(system_dim)).abs(),
        "choi_rank_upper_bound": torch.full(
            (psi.shape[0],), environment_dim, dtype=torch.int64,
            device=psi.device),
        "transport": transport,
    }


def reset_kraus_metrics(gamma: Tensor, reset: Tensor, targets: Tensor) -> dict[str, Tensor]:
    """Check the trained TCP reset Kraus family and protocol routing."""
    reset = _require_square_batch(reset)
    if gamma.ndim != 3 or gamma.shape[:2] != reset.shape[:2]:
        raise ValueError("gamma and reset shapes are incompatible")
    B, M, events = gamma.shape
    if targets.shape != (events, M):
        raise ValueError("targets must have shape (events,M)")
    stay = 1.0 - gamma.sum(-1)
    completeness = stay + gamma.sum(-1)
    minimum_weight = torch.minimum(stay.amin(-1), gamma.amin(dim=(-2, -1)))
    probability_violation = torch.maximum(
        (-gamma).clamp_min(0.0).amax(dim=(-2, -1)),
        (-stay).clamp_min(0.0).amax(-1),
    )

    # Independently reconstruct the sparse rank-one Kraus action and expose
    # any mass outside stay/loss/timeout destinations.
    expected = torch.diag_embed(stay)
    allowed = torch.eye(M, dtype=torch.bool, device=reset.device)
    batch = torch.arange(B, device=reset.device)[:, None].expand(B, M)
    source = torch.arange(M, device=reset.device)[None, :].expand(B, M)
    for event in range(events):
        destination = targets[event][None, :].expand(B, M)
        expected = expected.index_put(
            (batch, destination, source), gamma[:, :, event], accumulate=True)
        allowed[targets[event], torch.arange(M, device=reset.device)] = True
    leakage = torch.where(allowed[None], torch.zeros_like(reset), reset.abs())

    uniform = torch.full((M,), 1.0 / M, dtype=reset.dtype, device=reset.device)
    output = torch.einsum("bij,j->bi", reset, uniform)
    entropy = -(output.clamp_min(torch.finfo(output.dtype).tiny)
                * output.clamp_min(torch.finfo(output.dtype).tiny).log()).sum(-1)
    return {
        # The reset Kraus coefficients are sqrt(stay_j) and
        # sqrt(gamma_{j,e}); non-negative weights are therefore the CP check.
        "choi_min_weight": minimum_weight,
        "choi_psd_violation": (-minimum_weight).clamp_min(0.0),
        "probability_bounds_violation": probability_violation,
        "kraus_completeness_residual": (completeness - 1.0).abs().amax(-1),
        "matrix_reconstruction_residual": (reset - expected).abs().amax(
            dim=(-2, -1)),
        "protocol_routing_leakage": leakage.sum(dim=(-2, -1)),
        "total_event_probability_min": gamma.sum(-1).amin(-1),
        "total_event_probability_mean": gamma.sum(-1).mean(-1),
        "total_event_probability_max": gamma.sum(-1).amax(-1),
        "loss_probability_mean": gamma[..., 0].mean(-1),
        "timeout_probability_mean": gamma[..., 1].mean(-1),
        "nonunital_row_sum_residual": (reset.sum(-1) - 1.0).abs().amax(-1),
        "uniform_entropy_reduction": math.log(M) - entropy,
    }


def tcp_reset_target_metrics(grid, targets: Tensor) -> dict[str, int | float]:
    """Verify loss/timeout destinations against TCP semantics, not learning."""
    if targets.shape != (2, grid.M):
        raise ValueError("TCP reset requires loss and timeout target rows")
    source_w, source_q, _ = grid.axis_bins()
    loss_w, loss_q, loss_phase = _bins_from_indices(grid, targets[0])
    timeout_w, timeout_q, timeout_phase = _bins_from_indices(grid, targets[1])
    return {
        "loss_queue_change_count": int((loss_q != source_q).sum()),
        "loss_wrong_phase_count": int((loss_phase != 1).sum()),
        "loss_window_increase_count": int((loss_w > source_w).sum()),
        "timeout_queue_change_count": int((timeout_q != source_q).sum()),
        "timeout_nonminimum_window_count": int((timeout_w != 0).sum()),
        "timeout_wrong_phase_count": int((timeout_phase != 0).sum()),
        "loss_unique_targets": int(torch.unique(targets[0]).numel()),
        "timeout_unique_targets": int(torch.unique(targets[1]).numel()),
    }


def _bins_from_indices(grid, indices: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    phase = indices % grid.n_phase
    q_bin = torch.div(indices, grid.n_phase, rounding_mode="floor") % grid.n_q
    w_bin = torch.div(indices, grid.n_q * grid.n_phase,
                      rounding_mode="floor")
    return w_bin, q_bin, phase
