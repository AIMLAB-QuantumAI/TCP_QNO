"""Build a model from its registry name, which is also what a checkpoint stores."""
from __future__ import annotations

from qtro.baselines.cnto import CNTO
from qtro.baselines.deeponet import DeepONet
from qtro.baselines.seqpred import SeqPred
from qtro.model.qtro import N_COND_FEATURES, QTRO
from qtro.qubit.model import QubitQTRO
from qtro.qubit.ucr import UCRQTRO

MODEL_NAMES = ("qtro", "cnto", "seqpred", "deeponet",
               "qubit_hea", "qubit_dense", "qubit_ucr")
# Names stored in checkpoints written before the paper's terms were adopted.
LEGACY_NAMES = {"qkfp": "qtro", "softmax": "cnto"}


def canonical_name(name: str) -> str:
    return LEGACY_NAMES.get(name, name)


def build_model(name: str, grid, cfg, model_kwargs: dict | None = None):
    kwargs = dict(model_kwargs or {})
    name = canonical_name(name)
    if name == "qtro":
        return QTRO(grid, cfg.model)
    if name == "cnto":
        return CNTO(grid, N_COND_FEATURES, rank=int(kwargs.get("rank", 8)),
                    d_cond=cfg.model.d_cond, hidden=cfg.model.intensity_hidden)
    if name == "deeponet":
        return DeepONet(grid, N_COND_FEATURES, rank=int(kwargs.get("rank", 42)),
                        hidden=int(kwargs.get("hidden", 320)),
                        buffer_pkts=cfg.grid.buffer_pkts, w_max=cfg.grid.w_max)
    if name == "seqpred":
        return SeqPred(grid, N_COND_FEATURES, rank=int(kwargs.get("rank", 8)),
                       d_cond=cfg.model.d_cond, hidden=cfg.model.intensity_hidden)
    if name in ("qubit_hea", "qubit_dense"):
        return QubitQTRO(grid, cfg.model, ansatz=name.removeprefix("qubit_"))
    if name == "qubit_ucr":
        return UCRQTRO(grid, cfg.model)
    raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")


def parameter_count(model) -> int:
    return sum(p.numel() for p in model.parameters())
