"""Resolved experiment configuration: dataclasses, YAML loading and dotted overrides."""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass

import yaml


@dataclass
class GridConfig:
    n_w: int = 16
    n_q: int = 16
    n_phase: int = 2
    w_min: float = 1.0
    w_max: float = 64.0
    buffer_pkts: float = 90.0
    q_crit_frac: float = 0.8
    beta: float = 0.5
    q_cluster_strength: float = 3.0
    fast_retransmit_w: float = 4.0


@dataclass
class NetConfig:
    """The nominal link the grid's round-trip time is derived from."""
    capacity_pps: float = 2000.0
    rtt_prop_s: float = 0.030


@dataclass
class AQMConfig:
    """RED settings that are not part of the control: the gentle-region
    threshold as a fraction of the buffer, and the queue-averaging weight."""
    max_th_frac: float = 0.5
    w_q: float = 1e-4


@dataclass
class SimConfig:
    """Sampling interval of the state, and the device models run on."""
    dt_s: float = 0.010
    device: str = "auto"


@dataclass
class ModelConfig:
    d_env: int = 4
    n_layers: int = 8
    d_cond: int = 8
    intensity_hidden: int = 64
    # Upper bound on the total reset probability a cell may carry per step.
    # One is the construction the paper reports; the reset ablation pins it
    # near zero so the intensities cannot move mass.
    reset_max_probability: float = 1.0
    # Qubit-native transport only: angle initialisation scale, and for the
    # uniformly controlled ladder how many of the other eleven qubits a pair's
    # rotation angle may read (-1 reads them all, which is QTRO parameter for
    # parameter; 0 gives every sweep a single angle).
    qubit_init_scale: float = 0.01
    qubit_controls: int = -1


@dataclass
class DataConfig:
    horizons_train: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    # Columns with fewer than c_min observed transitions are dropped from the
    # loss: a conditional over M cells estimated from a handful of samples is
    # noise, not a property of the dynamics. c_cap_quantile caps the weight of
    # the hottest columns so one cell cannot own the objective.
    c_min: int = 32
    c_cap_quantile: float = 0.9


@dataclass
class TrainConfig:
    lr: float = 1e-3
    n_steps: int = 20000
    batch_configs: int = 16
    h0: int = 5
    lambda_obs: float = 1.0
    alpha_tail: float = 10.0
    # Topology-aware marginal loss: one-dimensional Wasserstein-1 distances of
    # the queue and window marginals on the physical grid. Zero disables it.
    lambda_wasserstein: float = 0.0
    wasserstein_queue_weight: float = 1.0
    wasserstein_window_weight: float = 1.0
    # Smooth tail profile over several fractions of the queue buffer, beside
    # the single q_crit overflow term of the observable loss. Zero disables it.
    lambda_multitail: float = 0.0
    tail_threshold_fracs: list[float] = field(
        default_factory=lambda: [0.6, 0.7, 0.8, 0.9])
    lambda_reg: float = 1e-6
    # Training needs a positive eps inside the Bhattacharyya square roots and
    # measurement must not have one; no single constant serves both, so
    # qtro.data.fidelity defaults to zero and training passes this value.
    fidelity_eps: float = 1e-12
    # Source-column mixture for the distribution term: occupancy-weighted,
    # uniform and tail-source components, each normalised over reliable
    # (count >= c_min) columns. The observable term keeps occupancy weights,
    # because alpha_tail already targets its risk functional.
    dist_weight_occupancy: float = 1.0
    dist_weight_uniform: float = 0.0
    dist_weight_tail: float = 0.0
    # omega_h is proportional to h^{-horizon_weight_power}: zero weighs every
    # horizon equally, a positive value protects the identifiable one-step
    # kernel from the non-Markov multi-step targets.
    horizon_weight_power: float = 0.0
    # Activation memory scales as batch x n_layers x d_env, and a run past the
    # card spills to shared memory and slows about 300x instead of failing;
    # the guard refuses to start above this fraction of VRAM.
    vram_frac: float = 0.8
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])


@dataclass
class Config:
    grid: GridConfig = field(default_factory=GridConfig)
    net: NetConfig = field(default_factory=NetConfig)
    aqm: AQMConfig = field(default_factory=AQMConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def _apply_mapping(obj, mapping: dict) -> None:
    known = {f.name for f in fields(obj)}
    for key, value in mapping.items():
        if key not in known:
            raise KeyError(f"unknown config key: {key}")
        current = getattr(obj, key)
        if is_dataclass(current):
            _apply_mapping(current, value)
        else:
            setattr(obj, key, value)


_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off")


def _coerce(current, text: str):
    # bool before int: bool is a subclass of int, so type(current)(text) on a
    # bool field would call bool("false") == True.
    if isinstance(current, bool):
        low = text.strip().lower()
        if low not in _TRUE + _FALSE:
            raise ValueError(
                f"expected a boolean, got {text!r}; use one of "
                f"{', '.join(_TRUE + _FALSE)}")
        return low in _TRUE
    if isinstance(current, str):
        return text
    if isinstance(current, list):
        # Comma-separated, element type taken from the current first element.
        # Without this, type(current)(text) on a list field would call
        # list("1,2,3") and silently produce ['1', ',', '2', ...].
        if not current:
            raise ValueError("cannot infer element type of an empty list field")
        elem = type(current[0])
        items = [p.strip() for p in text.split(",") if p.strip()]
        if not items:
            raise ValueError(f"expected a comma-separated list, got {text!r}")
        return [elem(p) for p in items]
    return type(current)(text)


def _apply_override(cfg: Config, item: str) -> None:
    key, _, text = item.partition("=")
    obj = cfg
    parts = key.split(".")
    for part in parts[:-1]:
        if not hasattr(obj, part):
            raise KeyError(f"unknown config section: {part}")
        obj = getattr(obj, part)
    leaf = parts[-1]
    if not hasattr(obj, leaf):
        raise KeyError(f"unknown config key: {key}")
    setattr(obj, leaf, _coerce(getattr(obj, leaf), text))


def load_config(path: str | None, overrides: list[str] | None) -> Config:
    cfg = Config()
    if path is not None:
        with open(path, "r", encoding="utf-8") as fh:
            _apply_mapping(cfg, yaml.safe_load(fh) or {})
    for item in overrides or []:
        _apply_override(cfg, item)
    return cfg


def config_from_dict(mapping: dict) -> Config:
    """Reconstruct a resolved config stored inside a checkpoint."""
    cfg = Config()
    _apply_mapping(cfg, mapping)
    return cfg
