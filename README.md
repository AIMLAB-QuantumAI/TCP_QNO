# TCP_QNO

Code for *Quantum Transport–Reset Operator Learning for Predictive Control in
TCP Dynamics* (QTRO). The paper is under review; a citation will be added once
it is published.

QTRO predicts the probability law of a TCP sender's (window, queue, phase)
state several steps ahead, conditioned on the RED setting and the link. It is
a diagonal-preserving CPTP channel on nine system and three environment
qubits: a Givens-ladder transport traced over the environment, followed by a
reset whose destinations are the ones TCP's loss and timeout prescribe.
Validity holds by construction at every step of a recursive rollout.

## Layout

| path | contents |
|---|---|
| `qtro/` | the operator (`model/`), its circuit (`quantum/`), the qubit-native variant (`qubit/`), baselines, evaluation, the ns-3 runner |
| `scripts/data/` | design and generate the ns-3 datasets |
| `scripts/train/` | train one model; the depth screen and the coverage curve |
| `scripts/evaluate/` | every accuracy, structure and cost result |
| `scripts/control/` | the closed loop and the state probes |
| `scripts/qubit/` | drivers for the qubit-native arms |
| `configs/` | training recipes and the condition lists of every dataset |
| `ns3/` | the three ns-3.42 programs (GPL-2.0) |
| `docs/REPRODUCING.md` | the command behind each reported result |

## Install

```sh
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # or /cpu
pip install -e ".[dev]"
pytest
```

Python 3.12 or newer. The reported runs used Python 3.12, PyTorch 2.11.0 with
CUDA 12.8 and a 16 GB GPU; training QTRO at the paper's recipe takes about
13 minutes per seed there. Everything runs on CPU with `--device cpu`, slower.

## Data

Datasets are generated with ns-3 (`ns3/README.md`), not shipped. Each is a
directory of `shard_*.npz` files plus a `manifest.json`; `qtro.data.SweepData`
loads them. The condition lists that define the paper's datasets are in
`configs/conditions/`.

## Reproducing the paper

`docs/REPRODUCING.md` lists, result by result, the data generation, training
and evaluation commands.

## Licence

MIT for everything outside `ns3/`; the ns-3 programs are GPL-2.0-only.
