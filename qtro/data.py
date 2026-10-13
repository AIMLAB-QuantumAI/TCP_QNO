"""Shards of ns-3 transition counts: the dataset the operator is fitted to and graded against.

A shard holds, per condition, the column-normalised h-step transition kernel
estimated from one half of the ns-3 replications (``target_h``), the same
estimate from the other half (``ref_h``), the per-column sample counts of
both halves, and the split-half agreement of the two, which is the
measurement floor every model is reported against.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


def fidelity(p: Tensor, q: Tensor, eps: float = 0.0) -> Tensor:
    """Squared Bhattacharyya coefficient of two distributions on the last axis.

    Exact by default. Measurement, and the split-half floor above all, must
    not be biased, and any eps inside the square roots biases it upward,
    which is the wrong direction for a quantity used as a ceiling.

    Training passes a positive eps. The Bhattacharyya coefficient has an
    unbounded derivative where one argument vanishes and the other does not,
    so a backpropagated call needs the guard even though a measured one must
    not have it. No single constant serves both uses, which is why the
    default is zero and the training value is a configuration field.
    """
    return (torch.sqrt(p.clamp(min=0) + eps)
            * torch.sqrt(q.clamp(min=0) + eps)).sum(-1) ** 2


def tv(p: Tensor, q: Tensor) -> Tensor:
    """Total variation distance, in [0, 1]."""
    return 0.5 * (p - q).abs().sum(-1)


class SweepData:
    """Shards of a dataset directory, concatenated and indexable.

    Held on the CPU in the float16 the shards store, and moved to the device
    a batch at a time: every training horizon of a few hundred conditions is
    gigabytes, which fits in host RAM but would crowd the model on a 16 GB
    card. ``horizons`` selects what to load, so an evaluation pass that wants
    h=10 and h=20 does not pay for the training targets.
    """

    def __init__(self, directory, horizons: list[int],
                 splits: tuple[str, ...] = ("train",), want_refs: bool = False,
                 require_targets: bool = True):
        directory = Path(directory)
        files = sorted(directory.glob("shard_*.npz"))
        if not files:
            raise FileNotFoundError(f"no shards in {directory}")

        u, xi, split, keep_any = [], [], [], False
        targets = {h: [] for h in horizons}
        counts = {h: [] for h in horizons}
        refs = {h: [] for h in horizons} if want_refs else {}
        ref_counts = {h: [] for h in horizons} if want_refs else {}
        noise = {h: [] for h in horizons}
        noise_tv = {h: [] for h in horizons}
        for f in files:
            d = np.load(f)
            sel = np.isin(d["split"], list(splits))
            if not sel.any():
                continue
            keep_any = True
            u.append(d["u"][sel])
            xi.append(d["xi"][sel])
            split.append(d["split"][sel])
            for h in horizons:
                key = f"target_h{h}"
                if key not in d:
                    if require_targets:
                        raise KeyError(
                            f"{f.name} has no {key}; it was generated with "
                            f"different horizons. Regenerate or ask for horizons "
                            f"the shards actually contain: "
                            f"{sorted(k for k in d.files if k.startswith('target_h'))}")
                else:
                    targets[h].append(d[key][sel])
                    counts[h].append(d[f"count_h{h}"][sel])
                noise[h].append(d[f"noise_fidelity_h{h}"][sel])
                tv_key = f"noise_tv_h{h}"
                noise_tv[h].append(
                    d[tv_key][sel] if tv_key in d
                    else np.full_like(d[f"noise_fidelity_h{h}"][sel], np.nan)
                )
                if want_refs:
                    refs[h].append(d[f"ref_h{h}"][sel])
                    ref_counts[h].append(d[f"count_ref_h{h}"][sel])
        if not keep_any:
            raise ValueError(f"no configuration in {directory} has split "
                             f"in {splits}")

        cat = lambda parts: torch.from_numpy(np.concatenate(parts, axis=0))
        self.u = cat(u).float()
        self.xi = cat(xi).float()
        self.split = np.concatenate(split, axis=0)
        self.horizons = list(horizons)
        self.targets = {h: cat(targets[h]) for h in horizons if targets[h]}
        self.counts = {h: cat(counts[h]).float() for h in horizons if counts[h]}
        self.noise_fidelity = {h: cat(noise[h]) for h in horizons}
        self.noise_tv = {h: cat(noise_tv[h]) for h in horizons}
        self.refs = {h: cat(refs[h]) for h in horizons} if want_refs else {}
        self.ref_counts = ({h: cat(ref_counts[h]).float() for h in horizons}
                           if want_refs else {})
        self.n = self.u.shape[0]

    def __len__(self) -> int:
        return self.n

    def batch(self, idx: Tensor, device: str) -> dict:
        """One batch, upcast to float32 on the device."""
        return {
            "u": self.u[idx].to(device),
            "xi": self.xi[idx].to(device),
            "targets": {h: value[idx].to(device).float()
                        for h, value in self.targets.items()},
            "counts": {h: value[idx].to(device)
                       for h, value in self.counts.items()},
            "refs": {h: self.refs[h][idx].to(device).float()
                     for h in self.refs},
            "ref_counts": {h: self.ref_counts[h][idx].to(device)
                           for h in self.ref_counts},
            "noise_fidelity": {h: self.noise_fidelity[h][idx].to(device)
                               for h in self.noise_fidelity},
            "noise_tv": {h: self.noise_tv[h][idx].to(device)
                         for h in self.noise_tv},
        }

    def conditioning_scale(self, idx: Tensor | None = None) -> Tensor:
        """Feature maxima measured on a training subset.

        The returned order is ``(p_max, min_th, capacity, rtt, n_flows,
        p_err)``. A one is used for an all-zero feature so division remains
        defined. Callers compute this on the training partition and store it
        in the checkpoint; validation or held-out values are never used.
        """
        features = torch.cat([self.u, self.xi], dim=-1)
        if idx is not None:
            features = features[idx]
        return features.abs().amax(dim=0).clamp_min(1.0e-12)


class SweepSubset:
    """Zero-copy index view used for deterministic train/validation splits."""

    def __init__(self, base: SweepData, indices: Tensor):
        self.base = base
        self.indices = torch.as_tensor(indices, dtype=torch.long)
        self.horizons = base.horizons
        # The training loop only needs these for key membership and the
        # per-column count report. Targets stay on the base and are selected
        # lazily by batch(), avoiding a multi-gigabyte copy.
        self.targets = base.targets
        self.counts = {h: v[self.indices] for h, v in base.counts.items()}
        self.u = base.u[self.indices]
        self.xi = base.xi[self.indices]

    def __len__(self) -> int:
        return int(self.indices.numel())

    def batch(self, idx: Tensor, device: str) -> dict:
        return self.base.batch(self.indices[idx], device)

    def conditioning_scale(self) -> Tensor:
        return self.base.conditioning_scale(self.indices)


def nested_training_fraction(indices: Tensor, fraction: float) -> Tensor:
    """Return a deterministic nested prefix of shuffled training indices."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("train_fraction must lie in (0, 1]")
    n = max(1, round(float(fraction) * len(indices)))
    return indices[:n]


def train_validation_split(order: Tensor, val_fraction: float) -> tuple[Tensor, Tensor]:
    """Split a shuffled order into validation and training indices.

    A zero validation fraction keeps every row for training, for a run whose
    model selection is already frozen and whose evaluation comes from a
    separate dataset.
    """
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must lie in [0, 1)")
    if val_fraction > 0.0 and len(order) < 2:
        raise ValueError(
            "a positive validation fraction requires at least two rows")
    n_val = 0 if val_fraction == 0.0 else min(
        len(order) - 1, max(1, round(val_fraction * len(order))))
    return order[:n_val], order[n_val:]


def manifest_sha256(directory) -> str | None:
    """Fingerprint the dataset contract without hashing multi-gigabyte shards."""
    path = Path(directory) / "manifest.json"
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()
