"""Shared fixtures for the test suite: synthetic transition shards."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def write_shards(directory, n_conditions: int = 4, horizons=(1, 2), M: int = 512,
                 seed: int = 0, splits=("train",), banded: bool = False) -> list[Path]:
    """Write one shard with the keys and dtypes ``generate_dataset.py`` writes.

    ``banded=True`` makes the targets a band around the diagonal instead of
    uniform noise, which a locality-constrained model can actually fit. Every
    column carries at least 40 fitting and grading samples, so every column
    is graded at the default ``c_min``. Rows take their split from ``splits``
    in turn. Returns the written paths.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    def kernel(h: int) -> np.ndarray:
        if banded:
            i = np.arange(M)[:, None]
            j = np.arange(M)[None, :]
            band = np.exp(-((i - j) ** 2) / (2.0 * (1 + h) ** 2))
            t = np.repeat(band[None], n_conditions, axis=0).astype(np.float32)
        else:
            t = rng.random((n_conditions, M, M)).astype(np.float32)
        return (t / t.sum(axis=1, keepdims=True)).astype(np.float16)

    payload: dict[str, np.ndarray] = {}
    for h in horizons:
        payload[f"target_h{h}"] = kernel(h)
        payload[f"count_h{h}"] = rng.integers(40, 500, (n_conditions, M)).astype(np.int32)
        payload[f"ref_h{h}"] = kernel(h)
        payload[f"count_ref_h{h}"] = rng.integers(40, 500, (n_conditions, M)).astype(np.int32)
        payload[f"noise_fidelity_h{h}"] = rng.random((n_conditions, M)).astype(np.float32)
        payload[f"noise_tv_h{h}"] = rng.random((n_conditions, M)).astype(np.float32)
    payload["u"] = rng.random((n_conditions, 2)).astype(np.float32)
    payload["xi"] = np.tile(np.array([[2000.0, 0.03, 4.0, 0.0]], np.float32),
                            (n_conditions, 1))
    payload["split"] = np.array([splits[i % len(splits)] for i in range(n_conditions)])
    path = directory / "shard_C2000_R30_N4.npz"
    np.savez_compressed(path, **payload)
    return [path]
