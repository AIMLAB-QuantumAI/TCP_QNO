"""Nearest-condition empirical transition baseline."""
from __future__ import annotations

import torch
from torch import Tensor, nn


class EmpiricalPropagator(nn.Module):
    """Plug-in one-step matrices fitted on the fitting half of the data.

    At an unobserved condition the closest training condition is used, in
    the feature metric defined by the training-derived conditioning scale.
    ``last_substitutions`` records how often that fallback happened, so an
    interpolation result cannot silently masquerade as a fitted value.
    """

    def __init__(self, features: Tensor, matrices: Tensor, scale: Tensor,
                 smoothing: float = 1.0e-7):
        super().__init__()
        if features.ndim != 2 or matrices.ndim != 3:
            raise ValueError("features must be (N,D) and matrices (N,M,M)")
        scale = torch.as_tensor(scale, dtype=torch.float32)
        self.register_buffer("features", features.float())
        self.register_buffer("scale", scale)
        P = matrices.float().clamp_min(0) + smoothing
        P = P / P.sum(dim=1, keepdim=True).clamp_min(1.0e-30)
        self.register_buffer("matrices", P)
        self.last_indices: Tensor | None = None
        self.last_distances: Tensor | None = None

    def propagator(self, u: Tensor, xi: Tensor) -> Tensor:
        query = torch.cat([u, xi], dim=-1)
        scale = self.scale.to(query)
        train = self.features.to(query)
        distance = torch.cdist(query / scale, train / scale)
        values, indices = distance.min(dim=1)
        self.last_indices = indices.detach().cpu()
        self.last_distances = values.detach().cpu()
        return self.matrices.to(query)[indices]

    def substitution_report(self) -> dict:
        if self.last_distances is None:
            return {"n_queries": 0, "n_substituted": 0}
        substituted = self.last_distances > 1.0e-7
        return {
            "n_queries": int(substituted.numel()),
            "n_substituted": int(substituted.sum()),
            "max_normalised_distance": float(self.last_distances.max()),
        }
