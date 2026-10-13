#!/usr/bin/env bash
# Train the hardware-efficient and dense-SU(4) qubit arms on the main dataset.
#
# Usage: DATA=path/to/main_dataset [PYTHON=python] [OUT=runs] \
#        scripts/qubit/run_ansatz.sh "<ansatz:L> ..." "<seeds>"
#   e.g. scripts/qubit/run_ansatz.sh "hea:8 hea:32 dense:8" "0,1,2"
#
# Each arm writes its full, unbuffered log to $OUT/<arm>.log, so an arm that
# dies mid-run leaves its traceback and its last step behind.
set -eu
PYTHON="${PYTHON:-python}"
DATA="${DATA:?set DATA to the main training dataset directory}"
OUT="${OUT:-runs}"
HERE="$(cd "$(dirname "$0")" && pwd)"
TRAIN="$HERE/../train/train.py"
CFG="$HERE/../../configs/ns3_h20.yaml"
ARMS="${1:?give the arms as \"<ansatz:L> ...\", e.g. \"hea:8 hea:32 dense:8\"}"
SEEDS="${2:?give the seeds as a comma list, e.g. 0,1,2}"
mkdir -p "$OUT"

for arm in $ARMS; do
  ansatz="${arm%%:*}"; L="${arm##*:}"
  name="qubit_${ansatz}_L${L}"
  echo "===== $name seeds=${SEEDS} ====="; date "+%H:%M:%S"
  rc=0
  "$PYTHON" -u "$TRAIN" --config "$CFG" --data "$DATA" \
    --model "qubit_${ansatz}" --variant "${L}x8" --seeds "$SEEDS" \
    --steps 8000 --out "$OUT/$name" < /dev/null > "$OUT/$name.log" 2>&1 || rc=$?
  echo "EXIT=$rc"
  if [ "$rc" -ne 0 ]; then
    echo "--- last lines of $OUT/$name.log ---"
    tail -n 15 "$OUT/$name.log"
  fi
  grep '"event": "complete"' "$OUT/$name.log" | tail -n 1 || true
done

date "+%H:%M:%S"
echo "ANSATZ_DONE"
