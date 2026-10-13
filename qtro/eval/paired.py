"""Crossed bootstrap over model seeds and network conditions.

The two sampling axes are model initialisation seed and network/AQM
condition. Resampling individual seed-condition cells would pretend that
cells sharing either axis were independent, so the two axes are resampled
independently and crossed in each replicate. A positive advantage always
favours the reference model.
"""
from __future__ import annotations

import numpy as np


def crossed_bootstrap(delta: np.ndarray, *, n_resamples: int,
                      rng: np.random.Generator,
                      confidence: float = 0.95) -> dict:
    """Paired mean and win rate of a ``seed x condition`` advantage matrix."""
    delta = np.asarray(delta, dtype=np.float64)
    if delta.ndim != 2 or not delta.shape[0] or not delta.shape[1]:
        raise ValueError(f"delta must be a non-empty 2D matrix, got {delta.shape}")
    finite = np.isfinite(delta)
    if not finite.any():
        raise ValueError("delta contains no finite paired observation")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")

    observed_values = delta[finite]
    observed_mean = float(observed_values.mean())
    observed_win = float((observed_values > 0).mean())
    observed_tie = float((observed_values == 0).mean())
    S, C = delta.shape
    boot_mean = np.empty(n_resamples, dtype=np.float64)
    boot_win = np.empty(n_resamples, dtype=np.float64)
    offset = 0
    while offset < n_resamples:
        size = min(512, n_resamples - offset)
        seed_idx = rng.integers(0, S, size=(size, S))
        config_idx = rng.integers(0, C, size=(size, C))
        sample = delta[seed_idx[:, :, None], config_idx[:, None, :]]
        sample_finite = np.isfinite(sample)
        denominator = sample_finite.sum(axis=(1, 2))
        boot_mean[offset:offset + size] = np.where(
            denominator > 0,
            np.where(sample_finite, sample, 0.0).sum(axis=(1, 2))
            / np.maximum(denominator, 1),
            np.nan,
        )
        boot_win[offset:offset + size] = np.where(
            denominator > 0,
            ((sample > 0) & sample_finite).sum(axis=(1, 2))
            / np.maximum(denominator, 1),
            np.nan,
        )
        offset += size

    alpha = (1.0 - confidence) / 2.0
    quantiles = [100.0 * alpha, 100.0 * (1.0 - alpha)]
    return {
        "mean_advantage": observed_mean,
        "mean_ci": np.nanpercentile(boot_mean, quantiles).tolist(),
        "win_rate": observed_win,
        "win_rate_ci": np.nanpercentile(boot_win, quantiles).tolist(),
        "tie_rate": observed_tie,
        "n_seed_config_pairs": int(finite.sum()),
        "n_seeds": S,
        "n_configurations": C,
    }
