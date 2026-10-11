#!/bin/bash
# usage: sweepq.sh <worktree> <tag> <device-env> ; runs every quantized case cold and appends one line each
W=$1; TAG=$2; DEV=$3
cd "$W" || exit 1
: > "sweepq-$TAG.txt"
for c in 509 621 622 633 629 623 619 618 611 520 518 512 511 510 497 491 277; do
  rm -rf "$W/home-$TAG"; mkdir -p "$W/home-$TAG"
  START=$(date +%s.%N)
  if [ "$DEV" = cpu ]; then
    CASE=$c HOME="$W/home-$TAG" RECIPE_FORCE_CPU=1 timeout 900 "$W/target/release/recipe" reproq.rs > "runq-$TAG.out" 2> "runq-$TAG.err"
  else
    CASE=$c HOME="$W/home-$TAG" RECIPE_DEVICE=$DEV timeout 900 "$W/target/release/recipe" reproq.rs > "runq-$TAG.out" 2> "runq-$TAG.err"
  fi
  CODE=$?
  END=$(date +%s.%N)
  echo "case=$c exit=$CODE wall=$(echo "$END - $START" | bc) load=$(cut -d' ' -f1 /proc/loadavg) artifacts=$(ls "$W/home-$TAG/.cache/recipe/native" | wc -l) $(rg -o 'RESULT.*|panicked at.*' "runq-$TAG.err" | head -1) $(rg -n 'panicked at' -A1 "runq-$TAG.err" | tail -1 | cut -c1-120)" >> "sweepq-$TAG.txt"
done
echo done >> "sweepq-$TAG.txt"
