#!/usr/bin/env bash
# The control dial of the uniformly controlled rotation ladder (qtro/qubit/ucr.py).
#
# Every arm is QTRO's recipe -- L4 x d_E 8, seed 0, 8000 steps -- with one
# change: how many of the other eleven bits a pair's rotation angle may read.
# k = -1 reads them all and is QTRO parameter for parameter, so it should land
# on QTRO's numbers; k = 0 gives every sweep a single angle and has fewer
# parameters than hea L32. The arms run most informative first.
#
# Usage: DATA=path/to/main_dataset [PYTHON=python] [OUT=runs] \
#        scripts/qubit/run_control_count.sh [k ...]
#   default k list: -1 8 4 0 6 10 2 9 7 5 3 1
#   LAYERS=16 scripts/qubit/run_control_count.sh 4 6
#        depth instead of control; arms are then named ucr_k<k>_L<layers>
set -eu
PYTHON="${PYTHON:-python}"
DATA="${DATA:?set DATA to the main training dataset directory}"
OUT="${OUT:-runs}"
HERE="$(cd "$(dirname "$0")" && pwd)"
TRAIN="$HERE/../train/train.py"
CFG="$HERE/../../configs/ns3_h20.yaml"
LAYERS="${LAYERS:-4}"
KS="${*:--1 8 4 0 6 10 2 9 7 5 3 1}"
mkdir -p "$OUT"

for k in $KS; do
  if [ "$k" = "-1" ]; then name=ucr_kall; else name="ucr_k$k"; fi
  if [ "$LAYERS" != "4" ]; then name="${name}_L${LAYERS}"; fi
  echo "===== $name ====="; date "+%H:%M:%S"
  rc=0
  "$PYTHON" -u "$TRAIN" --config "$CFG" --data "$DATA" \
    --model qubit_ucr --variant "${LAYERS}x8" --seeds 0 --steps 8000 \
    --set model.qubit_controls="$k" \
    --out "$OUT/$name" < /dev/null > "$OUT/$name.log" 2>&1 || rc=$?
  echo "EXIT=$rc"
  if [ "$rc" -ne 0 ]; then tail -n 12 "$OUT/$name.log"; fi
  grep '"event": "complete"' "$OUT/$name.log" | tail -n 1 | cut -c1-220 || true
done

date "+%H:%M:%S"
echo "CONTROL_COUNT_DONE"
