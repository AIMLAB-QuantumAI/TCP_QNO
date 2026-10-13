#!/usr/bin/env bash
# Further seeds for the arms the control-count table reports.
#
# Every arm already has seed 0 from run_control_count.sh / run_ansatz.sh; the
# training manifest merges seeds, so the same --out is reused. The arms are
# the control counts k = 10, 8, 4, 0 and k = -1 (every control, QTRO's own
# ladder) at depth 4, the hardware-efficient ansatz at depth 32, the two
# depth pairs (k = 8 at depth 8, k = 4 at depth 16) and the dense ansatz at
# depth 8, in the order of how much each matters to the table and how long
# it takes.
#
# Usage: DATA=path/to/main_dataset [PYTHON=python] [OUT=runs] [SEEDS=1,2,3,4] \
#        scripts/qubit/run_seeds.sh
set -eu
PYTHON="${PYTHON:-python}"
DATA="${DATA:?set DATA to the main training dataset directory}"
OUT="${OUT:-runs}"
HERE="$(cd "$(dirname "$0")" && pwd)"
TRAIN="$HERE/../train/train.py"
CFG="$HERE/../../configs/ns3_h20.yaml"
SEEDS="${SEEDS:-1,2,3,4}"
mkdir -p "$OUT"

train() {   # train <out-name> <model> <variant> [extra args...]
  local name="$1" model="$2" variant="$3"; shift 3
  echo "===== $name seeds=$SEEDS ====="; date "+%H:%M:%S"
  local rc=0
  "$PYTHON" -u "$TRAIN" --config "$CFG" --data "$DATA" \
    --model "$model" --variant "$variant" --seeds "$SEEDS" --steps 8000 \
    --out "$OUT/$name" "$@" < /dev/null > "$OUT/${name}_seeds.log" 2>&1 || rc=$?
  echo "EXIT=$rc"
  if [ "$rc" -ne 0 ]; then tail -n 12 "$OUT/${name}_seeds.log"; fi
  grep -c '"event": "complete"' "$OUT/${name}_seeds.log" || true
}

train ucr_k10 qubit_ucr 4x8 --set model.qubit_controls=10
train ucr_k8  qubit_ucr 4x8 --set model.qubit_controls=8
train ucr_k4  qubit_ucr 4x8 --set model.qubit_controls=4
train ucr_k0  qubit_ucr 4x8 --set model.qubit_controls=0
train ucr_kall qubit_ucr 4x8 --set model.qubit_controls=-1
train qubit_hea_L32 qubit_hea 32x8
train ucr_k8_L8   qubit_ucr 8x8  --set model.qubit_controls=8
train ucr_k4_L16  qubit_ucr 16x8 --set model.qubit_controls=4
train qubit_dense_L8 qubit_dense 8x8

date "+%H:%M:%S"
echo "SEEDS_DONE"
