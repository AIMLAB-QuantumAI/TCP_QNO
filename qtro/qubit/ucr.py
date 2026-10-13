"""The Givens ladder read as a qubit circuit, with its control count as a dial.

QTRO's transport rotates cell-adjacent pairs of the joint index. On the
twelve qubits that binary-encode that index, an even pair ``(2m, 2m+1)``
along an axis differs in that axis's low bit only, so an even sweep is one
``R_Y`` on that bit whose angle depends on the other eleven bits: a
uniformly controlled rotation with eleven controls and ``2**11`` angles. An
odd sweep pairs ``(2m+1, 2m+2)``; shifting the axis register down by one maps
those pairs onto the even ones, so it is the same controlled rotation
conjugated by the register's decrement, a fixed permutation of basis states
like the CNOT entangler of the hardware-efficient block.

That reading makes the number of controls a dial. With ``k`` controls the
rotation on a pair reads ``k`` of the other bits and shares its angle with
every pair that agrees on them: ``2**k`` angles per sweep. At ``k = 11`` the
family is QTRO, parameter for parameter, and loads its checkpoints; at
``k = 0`` every sweep is a single ``R_Y`` on one qubit and the block has as
few angles as ``hea``. Which bits count first is fixed by relevance to the
pair: its own axis (where it sits on that axis, coarsest bit first), then
the phase, then the other physical axis, then the environment.

Training and inference reuse the Givens sweep, which is the efficient way to
apply a uniformly controlled rotation: every pair's angle is gathered from
the parameter slots by a fixed index map. ``simulator.apply_ucr`` applies the
same gate from qubit primitives, and the tests check that the two agree.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from qtro.config import ModelConfig
from qtro.model.qtro import N_COND_FEATURES, QTRO
from qtro.model.transport import TransportBlock
from qtro.quantum.circuit import ConditionedCircuit
from qtro.quantum.givens import apply_sweep
from qtro.quantum.pairs import build_sweeps
from qtro.quantum.qubit import AXIS_OF, registers

# The other registers in the order their bits become controls.
OTHERS = {"window": ("phase", "queue", "env"),
          "queue": ("phase", "window", "env"),
          "phase": ("window", "queue", "env"),
          "env": ("phase", "window", "queue")}


def control_order(regs: dict[str, list[int]],
                  sweep: str) -> tuple[int, list[int]]:
    """Target bit of a sweep and its control bits, most relevant first."""
    axis = AXIS_OF[sweep.split("_")[0]]
    own = regs[axis]
    order = list(reversed(own[1:]))            # own axis, coarsest bit first
    for name in OTHERS[axis]:
        order += list(reversed(regs[name]))
    return own[0], order


def sweep_controls(grid, d_env: int, n_controls: int) -> list[dict]:
    """Per sweep: its target, its controls, and the slot of every pair.

    ``n_controls < 0`` means every other bit. ``slot`` is the integer the
    controls spell, most significant first, read off the pair's lower index
    (for an odd sweep that is the index the decrement maps onto an even
    pair, so its own-axis bits give the pair's position ``m``). ``index`` is
    ``slot`` compacted to ``0 .. n_slots - 1`` in increasing order, which for
    every sweep is the layout ``build_sweeps`` enumerates its pairs in.
    """
    regs = registers(grid, d_env)
    out = []
    for name, lo, hi in build_sweeps(grid, d_env):
        target, order = control_order(regs, name)
        k = len(order) if n_controls < 0 else min(n_controls, len(order))
        controls = sorted(order[:k], reverse=True)
        slot = torch.zeros_like(lo)
        for bit in controls:
            slot = slot * 2 + ((lo >> bit) & 1)
        uniq, index = torch.unique(slot, return_inverse=True)
        out.append({"name": name, "target": target, "controls": controls,
                    "lo": lo, "hi": hi, "slot": slot, "index": index,
                    "n_slots": int(uniq.numel())})
    return out


def angle_map(grid, d_env: int, n_controls: int) -> tuple[Tensor, list[int]]:
    """Pair -> parameter index over all sweeps, and the slot count per sweep."""
    maps, counts, offset = [], [], 0
    for sw in sweep_controls(grid, d_env, n_controls):
        maps.append(sw["index"] + offset)
        counts.append(sw["n_slots"])
        offset += sw["n_slots"]
    return torch.cat(maps), counts


class UCRLadder(ConditionedCircuit):
    """``ConditionedCircuit`` whose angles are read through ``angle_map``."""

    def __init__(self, grid, cfg: ModelConfig, n_cond_features: int):
        super().__init__(grid, cfg, n_cond_features)
        self.n_controls = int(cfg.qubit_controls)
        amap, counts = angle_map(grid, cfg.d_env, self.n_controls)
        self.register_buffer("angle_map", amap.to(grid.device),
                             persistent=False)
        self.slot_counts = counts
        self.n_slots = sum(counts)
        scale = float(cfg.qubit_init_scale)
        self.trainable = nn.ParameterList([
            nn.Parameter(scale * torch.randn(self.n_slots))
            for _ in range(cfg.n_layers)
        ])
        self.cond_heads = nn.ParameterList([
            nn.Parameter(scale * torch.randn(self.n_slots, cfg.d_cond))
            for _ in range(cfg.n_layers)
        ])

    def forward(self, cond: Tensor,
                source_indices: Tensor | None = None) -> Tensor:
        B = cond.shape[0]
        h = self.encoder(cond)
        psi = self._basis_columns(
            B, cond.device, cond.dtype, source_indices=source_indices)
        for layer in range(self.n_layers):
            cond_angles = (h @ self.cond_heads[layer].T)[:, self.angle_map]
            train_angles = self.trainable[layer][self.angle_map]
            train_angles = train_angles.unsqueeze(0).expand(B, -1)
            for angles in (cond_angles, train_angles):
                for k in range(self.n_sweeps):
                    start, count = self.offsets[k]
                    psi = apply_sweep(
                        psi, getattr(self, f"lo_{k}"), getattr(self, f"hi_{k}"),
                        angles[:, start:start + count],
                        getattr(self, f"untouched_{k}"),
                    )
        return psi


class UCRTransportBlock(TransportBlock):
    def __init__(self, grid, cfg: ModelConfig, n_cond_features: int):
        super().__init__(grid, cfg, n_cond_features)
        self.circuit = UCRLadder(grid, cfg, n_cond_features)


class UCRQTRO(QTRO):
    """QTRO with the ladder's angles tied through ``k`` control bits.

    Everything but ``transport.circuit`` is the parent's, and at ``k = -1``
    the state dict is QTRO's key for key and shape for shape.
    """

    def __init__(self, grid, cfg: ModelConfig):
        super().__init__(grid, cfg)
        self.transport = UCRTransportBlock(grid, cfg, N_COND_FEATURES)
        self.ansatz = "ucr"
