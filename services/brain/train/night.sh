#!/bin/sh
# Finish version 1 unattended: train on against greedy, check win/loss-only
# reward, freeze the plain fly and tag v1, then run the wiring controls.
#
#   train/night.sh BEST_RUN RATE CONTROL_GAMES [SEEDS]
#
# Two runs at a time. Timestamps go to artifacts/runs/night.log. For a dry run:
# NO_TAG=1 skips the git tag, A3_GAMES and EVAL_GAMES shrink the other steps.
set -u
cd "$(dirname "$0")/.." || exit 1
BEST=$1
RATE=$2
GAMES=$3
SEEDS=${4:-10}
A3=${A3_GAMES:-4096}
EVAL=${EVAL_GAMES:-100}
PY=../../.venv/bin/python
RUNS=${FLY_ARTIFACTS_DIR:-../../artifacts}/runs
LOG=$RUNS/night.log
mkdir -p "$RUNS"
# a subshell, so the waits below don't wait for caffeinate too
(caffeinate -i -w $$ &)

train() {
    name=$1
    shift
    echo "$(date '+%H:%M') start $name" >> "$LOG"
    "$PY" -m train.selfplay --run "$name" --rate "$RATE" --parallel 256 "$@" > "$RUNS/$name.out" 2>&1
    echo "$(date '+%H:%M') done  $name (exit $?)" >> "$LOG"
}

echo "$(date '+%H:%M') night: best=$BEST rate=$RATE control games=$GAMES seeds=$SEEDS" >> "$LOG"

train a3-greedy --init "$BEST" --opponent greedy --games "$A3" --eval-every $((A3 / 4)) --eval-games "$EVAL" &
train a3-terminal --reward terminal --games "$A3" --eval-every $((A3 / 4)) --eval-games "$EVAL" &
wait

if "$PY" -m train.freeze --runs "$BEST" a3-greedy --name plain --games $((EVAL * 3)) >> "$LOG" 2>&1; then
    [ "${NO_TAG:-0}" = 1 ] || git tag v1 >> "$LOG" 2>&1
    echo "$(date '+%H:%M') froze plain" >> "$LOG"
else
    echo "$(date '+%H:%M') freeze failed" >> "$LOG"
fi

running=0
seed=0
while [ "$seed" -lt "$SEEDS" ]; do
    for control in real degree random compartments; do
        train "c-$control-$seed" --control "$control" --seed "$seed" --games "$GAMES" \
            --eval-every "$GAMES" --eval-games "$EVAL" --eval-against random &
        running=$((running + 1))
        if [ "$running" -ge 2 ]; then
            wait
            running=0
        fi
    done
    seed=$((seed + 1))
done
wait

"$PY" -m train.report --pattern "c-*" > "$RUNS/controls-report.txt" 2>&1
echo "$(date '+%H:%M') all done" >> "$LOG"
